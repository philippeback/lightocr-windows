# Software Design and Implementation (SDI) — `lighton-ocr`

**Document Status:** Living Architecture & Implementation Specification  
**Target System:** `lighton-ocr` (`lightonai/LightOnOCR-3-4B` on Windows x64 + AMD ROCm `gfx1151`)  

---

## 1. Executive Summary & System Taxonomy

`lighton-ocr` is a local optical character recognition (OCR) and document transcription engine built around **`lightonai/LightOnOCR-3-4B`**, a 4-billion-parameter multimodal vision-language model based on the **Qwen3.5** hybrid architecture (`Qwen3_5ForConditionalGeneration`).

Running this model locally on **Windows x64 (`win_amd64`)** with **AMD Strix Halo (`gfx1151` / Radeon 8060S Graphics)** via **ROCm 10.1.0** exposes a dense intersection of package-management, multimodal tensor routing, custom kernel dispatch, hardware memory-bandwidth, and autoregressive decoding invariants. If any single invariant is violated, the system fails either overtly (dependency resolution errors, missing attributes) or silently (CPU fallback, blind text-only decoding outputting `"The\n\nThe..."`, or 10x latency inflation from runaway token loops).

### 1.1 Hardware & Runtime Environment Matrix

| Layer | Component | Specification / Version | Architectural Significance |
| :--- | :--- | :--- | :--- |
| **APU / GPU** | AMD Radeon(TM) 8060S Graphics | `gfx1151` (28 CUs, 2.83 GHz) | RDNA 3.5 integrated compute engine (`Compute 0` reaches 90–94% utilization during inference) |
| **Memory Subsystem** | Unified Memory Architecture (UMA) | 15.8 GB Dedicated + 42.3 GB Shared | Bandwidth-shared LPDDR5X; sensitive to weight precision (`bfloat16` vs. quantized GGUF) |
| **OS & Driver** | Windows 11 x64 (`win32` / `AMD64`) | AMD Driver `32.0.31041.5005` | Native Windows HIP runtime (no WSL2 virtualization layer) |
| **Python Runtime** | CPython | `3.13.13` (`>=3.13,<3.14`) | Matches prebuilt `cp313-cp313-win_amd64` ROCm wheels |
| **ROCm / HIP SDK** | `rocm[libraries,device-gfx1151]` | `10.1.0` (HIP `7.16.26385`) | Ships HIP runtime, MIOpen, hipBLASLt, and `gfx1151` device bitcode/kernels in `.venv` |
| **PyTorch Stack** | `torch[device-gfx1151]` | `2.14.0+rocm10.1.0` | Pulls `amd-torch-device-gfx1151` and `amd-torch-device-gfx115x` fatbins |
| **Vision Stack** | `torchvision[device-gfx1151]` | `0.29.0a0+rocm10.1.0` | Alpha pre-release tag requires explicit pre-release opt-in during resolution |
| **Model Framework** | `transformers` / `accelerate` | `5.20.2` / `1.15.0` | Implements `Qwen3_5ForConditionalGeneration` and `hub_kernels` fallback dispatch |

---

## 2. End-to-End Architecture & Dataflow

```mermaid
flowchart TD
    subgraph Client["Application Layer"]
        A["example_usage.py"] -->|"image_path"| B["LightOnOCR.transcribe()"]
    end

    subgraph Init["Initialization & Cache Layer (LightOnOCR.__init__)"]
        C["Try local_files_only=True"] -->|"Cache Hit (0 HTTP calls)"| D["Loaded Processor & Model"]
        C -->|"OSError (Cache Miss)"| E["Fallback local_files_only=False (HF Hub Download)"]
        E --> D
    end

    subgraphPre["Preprocessing & Multimodal Packing"]
        B --> F["_preprocess(): RGB convert + LANCZOS thumbnail <=1024px"]
        F --> G["processor.apply_chat_template(messages, add_generation_prompt=True)"]
        G --> H["AutoProcessor(text, images) -> input_ids, attention_mask, pixel_values, image_grid_thw"]
    end

    subgraph Model["Qwen3_5ForConditionalGeneration (bfloat16 on ROCm GPU)"]
        H --> I["Vision Encoder (2D Axial RoPE + Patch Merger)"]
        I -->|"Visual Embeddings"| J["Multimodal Token Scatter into <|image_pad|>"]
        J --> K["Hybrid Decoder Prefill (Full Attention + torch_chunk_gated_delta_rule)"]
        K --> L["Autoregressive Decode Loop (torch_recurrent_gated_delta_rule + causal_conv1d_update)"]
    end

    subgraph Guard["Decoding Guardrails & Post-Processing"]
        L -->|"Per-step input_ids"| M["_RepeatTokenStoppingCriteria (24x single / 16x pair)"]
        M -->|"Stop / EOS"| N["Slice prompt_len + processor.decode()"]
        N --> O["Regex Block Cap (█ <= 40) + rstrip()"]
    end
```

---

## 3. Dependency Resolution & Package Management Architecture (`uv` + ROCm)

### 3.1 Imperative vs. Declarative State Mismatch
`uv` operates in two distinct modes:
1. **Imperative (`uv pip install ...`):** Mutates `.venv/Lib/site-packages` directly without recording index URLs, extras, or version constraints in [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml) or `uv.lock`.
2. **Declarative (`uv lock` / `uv sync` / `uv run`):** Computes a deterministic dependency graph from [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml), writes `uv.lock`, and reconciles `.venv` to match `uv.lock` before executing commands.

**Consequence:** If ROCm wheels are installed via `uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ ...` while [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml) still declares generic `"torch>=2.0"`, the very next `uv run` or `uv sync` resolves `torch` from PyPI (`2.14.1+cpu`), **uninstalling the ROCm build and silently reverting the environment to CPU-only execution**.

### 3.2 Three Resolution Pathologies on AMD's ROCm Wheel Index

When migrating the AMD ROCm installation into [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml), three non-obvious resolver failures occur unless explicitly handled:

#### Pathology 1: Pre-release Transitive Dependency Rejection
* **Mechanism:** `torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0` is an alpha version (`0.29.0a0`) that declares `Requires-Dist: amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0 ; extra == "device-gfx1151"`.
* **Resolver Behavior:** By PEP 440 / `uv` rules, pre-release versions of transitive dependencies are excluded unless pre-releases are globally enabled. Without it, `uv` reports:
  > `Because there is no version of amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0 ... we can conclude that your requirements are unsatisfiable.`
* **Fix:** Set `prerelease = "allow"` under `[tool.uv]`.

#### Pathology 2: Linux-Only `triton` Metadata Leak in `torch`
* **Mechanism:** On `https://stable.repo.amd.com/rocm/whl-next/`, `torch==2.14.0+rocm10.1.0` publishes both Linux (`manylinux`) and Windows (`win_amd64`) wheels. During universal lockfile generation (`uv lock`), `uv` reads the wheel metadata for `torch==2.14.0+rocm10.1.0`, which declares an unconditional dependency on `triton==3.8.0+git669b31ac.rocm10.1.0` (missing a `; sys_platform == 'linux'` marker).
* **Resolver Behavior:** Because AMD's repo does not publish `win_amd64` wheels for `triton==3.8.0+git669b31ac.rocm10.1.0`, `uv lock` aborts on Windows:
  > `Because triton==3.8.0+git669b31ac.rocm10.1.0 has no platform_machine == 'AMD64' and sys_platform == 'win32'-compatible wheels and torch==2.14.0+rocm10.1.0 depends on triton==3.8.0+git669b31ac.rocm10.1.0, we can conclude that torch==2.14.0+rocm10.1.0 cannot be used.`
* **Fix:** Override the broken metadata in `[tool.uv]` using `override-dependencies = ["triton; sys_platform == 'linux'"]` and scope lockfile resolution to Windows x64 via `environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]`.

#### Pathology 3: Index Scoping for Transitive ROCm SDK Packages
* **Mechanism:** `rocm[libraries,device-gfx1151]==10.1.0` and `torch[device-gfx1151]==2.14.0+rocm10.1.0` pull in transitive packages from the AMD index:
  * `rocm-sdk-core==10.1.0`
  * `rocm-sdk-libraries==10.1.0`
  * `rocm-sdk-device-gfx1151==10.1.0`
  * `rocm-bootstrap==0.3.0`
  * `amd-torch-device-gfx1151==2.14.0+rocm10.1.0`
  * `amd-torch-device-gfx115x==2.14.0+rocm10.1.0`
  * `amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0`
* **Fix:** Define `[[tool.uv.index]]` pointing to `https://stable.repo.amd.com/rocm/whl-next/` **without** `explicit = true`, allowing all transitive ROCm/AMD packages to resolve automatically from the AMD index while standard packages (`transformers`, `accelerate`, `Pillow`, `pytest`) resolve from PyPI.

---

## 4. Multimodal Model Loading & Tensor Routing Architecture

### 4.1 The `AutoModelForCausalLM` vs. `Qwen3_5ForConditionalGeneration` Failure Mode

`lightonai/LightOnOCR-3-4B` is a **multimodal conditional generation model** whose checkpoint contains two distinct weight namespaces:
1. **Vision Tower & Multimodal Projector (`visual.*`):** Encodes 2D image patches using 2D axial rotary embeddings (`Qwen3_5VisionRotaryEmbedding`) and projects them into the language model's hidden dimension.
2. **Hybrid Text Decoder (`model.layers.*`, `lm_head.*`):** 426 weight tensors comprising full self-attention layers interleaved with `Qwen3_5GatedDeltaNet` linear-attention layers.

#### What Happens When Loaded with `AutoModelForCausalLM` (Broken State)
1. `AutoModelForCausalLM.from_pretrained("lightonai/LightOnOCR-3-4B")` instantiates the **text-only** class (`Qwen3_5ForCausalLM`).
2. It loads only the 426 text-decoder weights (`Loading weights: 100%|...| 426/426`) and silently ignores the vision encoder weights.
3. Because `Qwen3_5ForCausalLM.forward()` does not accept `pixel_values`, `image_grid_thw`, or `mm_token_type_ids`, the original code popped those keys before calling `generate()`:
   ```python
   # BROKEN ANTI-PATTERN (removed):
   unused_keys = ['mm_token_type_ids', 'pixel_values', 'image_grid_thw']
   for key in unused_keys:
       inputs.pop(key, None)
   ```
4. Stripping `pixel_values` and `image_grid_thw` meant **the image was completely discarded**. The text decoder received only raw `<|image_pad|>` token IDs whose unreplaced token embeddings are uninformative, causing the decoder to collapse into an unconditioned attractor loop:
   ```text
   The

   The

   The
   ...
   ```

#### Correct Multimodal Pipeline (Implemented in [`lightocr/engine.py`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py))
1. **Model Class:** Load via `Qwen3_5ForConditionalGeneration.from_pretrained(...)` so both the vision tower and language decoder are initialized.
2. **Numerical Precision (`torch.bfloat16`):** Use `dtype=torch.bfloat16` rather than `torch.float16`. `bfloat16` preserves the 8-bit exponent range of `float32`, preventing activation/exponentiation overflow in the vision tower and Gated DeltaNet recurrent decay (`exp(cum_decay)`), while `transformers 5.x` standardizes on the `dtype=` keyword argument (deprecating `torch_dtype=`).
3. **Chat Template Formatting:** `LightOnOCR-3-4B` is trained with Qwen3.5 multimodal chat markers (`<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>...<|im_end|>\n<|im_start|>assistant\n`). Inputs must be constructed via:
   ```python
   messages = [
       {
           "role": "user",
           "content": [
               {"type": "image", "image": image},
               {"type": "text", "text": prompt},  # "" for markdown OCR, "grounding" for layout boxes
           ],
       }
   ]
   text_prompt = self.processor.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
   )
   inputs = self.processor(
       text=[text_prompt],
       images=[image],
       padding=True,
       return_tensors="pt",
   ).to(self.model.device)
   ```
4. **Prompt Slicing:** `model.generate()` returns the concatenated `[prompt_ids, generated_ids]`. Slicing `output_ids[0, inputs["input_ids"].shape[1]:]` isolates the assistant's transcription.

### 4.2 Zero-Latency Offline Cache Loading (`local_files_only` Fast-Path)
By default, `AutoProcessor.from_pretrained(model_id)` and `PreTrainedModel.from_pretrained(model_id)` issue 7+ synchronous HTTP `GET`/`HEAD` requests to `https://huggingface.co/api/models/...` on every startup to check for remote revision updates and `additional_chat_templates`.

To eliminate network latency and allow fully offline operation once weights are cached:
* [`LightOnOCR.__init__`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py#L46-L86) attempts `from_pretrained(..., local_files_only=True)` first when `local_files_only=None`.
* If all files exist in `~/.cache/huggingface/hub`, initialization completes with **zero HTTP requests**.
* If `OSError` is raised (cache miss on first run), it logs a message and falls back to `local_files_only=False`.

---

## 5. Kernel Dispatch & Hardware Performance Architecture (ROCm PyTorch vs. `llama.cpp`)

### 5.1 Hybrid Architecture of `LightOnOCR-3-4B` (`Qwen3.5`)
Unlike traditional transformer VLMs that use standard softmax self-attention in every layer, `Qwen3.5` interleaves standard attention layers with **`Qwen3_5GatedDeltaNet`** linear-attention layers. Each `Qwen3_5GatedDeltaNet` layer relies on four specialized primitives:

| Primitive | Execution Phase | Purpose | Hub / Native Package |
| :--- | :--- | :--- | :--- |
| `causal_conv1d_fn` | Prefill ($L > 1$) | Depthwise 1D causal convolution over projected $Q, K, V$ | `causal-conv1d` / `kernels-community/mamba-ssm` |
| `chunk_gated_delta_rule` | Prefill ($L > 1$) | Chunk-parallel Gated Delta Rule state update & output | `flash-linear-attention` (`fla`) / `kernels-community/fla` |
| `causal_conv1d_update` | Decode ($L = 1$) | Single-step rolling state update for depthwise 1D conv | `causal-conv1d` / `kernels-community/mamba-ssm` |
| `fused_recurrent_gated_delta_rule` | Decode ($L = 1$) | Single-step recurrent state update $S_t \in \mathbb{R}^{H \times D_k \times D_v}$ | `flash-linear-attention` (`fla`) / `kernels-community/fla` |

### 5.2 Why Optimized Kernels Fall Back on Windows ROCm
In `transformers/models/qwen3_5/modeling_qwen3_5.py`, these four functions are decorated with `@use_kernel_func_from_hub_with_fallback(func_name, package)` (defined in `transformers/integrations/hub_kernels.py`), which resolves implementations in three tiers:
1. **Tier 1 — Hugging Face `kernels` Hub (`kernels-community/fla`, `kernels-community/mamba-ssm`):** Only registered for `"cuda"` (Linux CUDA) and `"xpu"` in `_KERNEL_MAPPING`, not Windows ROCm.
2. **Tier 2 — Local Python Packages (`fla` / `flash-linear-attention` and `causal_conv1d`):**
   * `flash-linear-attention` writes its kernels in **Triton** (`import triton`). Because AMD's `https://stable.repo.amd.com/rocm/whl-next/` repository currently publishes `triton` wheels only for Linux (`manylinux`), `fla` cannot execute Triton kernels on Windows ROCm.
   * `causal-conv1d` is a compiled C++/CUDA/HIP extension without prebuilt Windows `gfx1151` wheels.
3. **Tier 3 — Reference PyTorch Implementation (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `torch_causal_conv1d_update`):**
   * Executes standard PyTorch tensor operations **on the AMD Radeon 8060S GPU via ROCm/HIP**.

### 5.3 Why Reference PyTorch on ROCm is Slower than `llama.cpp`

During autoregressive decoding (`sequence_length == 1`), `torch_recurrent_gated_delta_rule` in `modeling_qwen3_5.py` executes the following per layer, per generated token:

```python
# Upcasts bfloat16 -> float32 and allocates contiguous copies on every token:
query, key, value, beta, decay = [
    x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
    for x in (query, key, value, beta, decay)
]
...
for i in range(sequence_length):
    q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
    decay_t = decay[:, :, i].exp()[..., None, None]
    last_recurrent_state = last_recurrent_state * decay_t
    beta_t = beta[:, :, i].unsqueeze(-1)
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

Comparing the two execution engines explains the latency gap:

| Dimension | `transformers` on Windows ROCm (Current) | `llama.cpp` Hosted Server |
| :--- | :--- | :--- |
| **Weight Precision & Memory Traffic** | **16-bit (`bfloat16`)** — ~8 GB weights read from unified memory on every token step | **Quantized (`Q4_K_M` / `Q8_0`)** — ~2.5 to 4.5 GB read per token (**2–3.5x less memory bandwidth** on LPDDR5X UMA) |
| **Gated DeltaNet / SSM Step** | **Un-fused PyTorch ops (~15 HIP kernel launches/layer)** with `bfloat16 -> float32 -> bfloat16` round-trips and intermediate VRAM reads/writes of the recurrent state $S_t$ | **Single fused C++/HIP/Vulkan kernel** keeping recurrent state $S_t$ in registers/LDS without FP32 intermediate allocations |
| **Dispatch Overhead** | Python interpreter loop + PyTorch ATen/HIP dispatcher per op | Native C++ token loop with static graph / minimal host-device launch latency |

### 5.4 MIOpen Find-Database Warning (`gfx115128.HIP.fdb.txt`)
When PyTorch invokes `nn.Conv1d` on ROCm during the first forward pass, MIOpen probes `.venv\Lib\site-packages\_rocm_sdk_libraries\bin\gfx115128.HIP.fdb.txt` (`gfx1151` architecture with `28` Compute Units = `gfx115128`). Because the ROCm 10.1.0 SDK package omits this pre-tuned text database file, MIOpen prints:
```text
MIOpen: Warning [ParseAndLoadDb] File is unreadable: "...\gfx115128.HIP.fdb.txt"
```
and falls back to its internal kernel heuristics. Setting `os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")` in [`lightocr/engine.py`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py#L6) before `import torch` restricts MIOpen logging to errors only.

---

## 6. Autoregressive Decoding Pathologies & Guardrail Engineering

### 6.1 Anatomy of the Progress-Bar Repetition Loop
When transcribing terminal screenshots containing ASCII/Unicode progress bars (e.g., `4 / 6 lines [██████████...`), greedy decoding enters a **degenerate repetition attractor**:
1. The visual feature map shows a dense horizontal bar of identical solid blocks.
2. Each emitted `█` (`\u2588`) increases the conditional probability $P(x_t = \text{'█'} \mid x_{<t} = \text{'██...█'})$ in the recurrent state $S_t$ of the Gated DeltaNet layers.
3. Once inside this attractor, the probability of emitting `]` or `<|im_end|>` drops below `█`, so the model emits `█` until `max_new_tokens=2048` is exhausted.
4. **Latency Impact:** On a screenshot with only ~150 tokens of real code/text, generating 1,900 extra tokens of `█` **inflates total inference time by >10x** (spending >90% of GPU compute time on useless block characters).

### 6.2 `_RepeatTokenStoppingCriteria` Design
To terminate degenerate loops immediately without harming legitimate text or code indentation, [`_RepeatTokenStoppingCriteria`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py#L22-L44) monitors the tail of `input_ids` after `prompt_length` at every decoding step:

1. **Single-Token Run-Length Invariant (`max_single_repeats = 24`):**
   $$\forall i \in [L - 24, L - 1], \quad \text{token}_i = \text{token}_{L-1} \implies \text{STOP}$$
   Catches single-token repetition loops (e.g., `█`, whitespace runs, newline loops).
2. **Period-2 Token Pair Invariant (`max_pair_repeats = 16`):**
   $$\forall j \in [0, 15], \quad (\text{token}_{L - 32 + 2j}, \text{token}_{L - 32 + 2j + 1}) = (\text{token}_{L - 2}, \text{token}_{L - 1}) \implies \text{STOP}$$
   Catches alternating 2-token oscillation loops (e.g., `"The\n\n"`, `" | | |"`).
3. **Post-Decode Regex Normalization:**
   ```python
   text = re.sub(r"(█)\1{39,}", lambda m: m.group(1) * 40, text)
   return text.rstrip()
   ```
   Even if a single token encodes multiple `█` characters, the output progress bar is capped cleanly at 40 blocks and trailing blank lines are stripped.

---

## 7. Module & File Reference

| File | Responsibility |
| :--- | :--- |
| [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml) | Declarative `uv` configuration pinning ROCm 10.1.0 `gfx1151` packages, `amd-rocm` index, `prerelease = "allow"`, Windows x64 `environments`, and Linux-scoped `triton` override. |
| [`lightocr/engine.py`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py) | Core [`LightOnOCR`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py#L47-L163) wrapper: silences MIOpen/hub_kernels noise, implements offline-first model loading (`Qwen3_5ForConditionalGeneration` in `bfloat16`), Pillow 10+ `LANCZOS` image preprocessing, multimodal chat template formatting, and [`_RepeatTokenStoppingCriteria`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/engine.py#L22-L44). |
| [`lightocr/utils.py`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/utils.py) | Path validation ([`validate_image_path`](file:///C:/Dev/github/philippeback/lighton-ocr/lightocr/utils.py#L5-L9)) and scoped `lightocr` package logger (preventing root logger `INFO:httpx2:` spam). |
| [`rocm-check.py`](file:///C:/Dev/github/philippeback/lighton-ocr/rocm-check.py) | Diagnostic CLI script verifying PyTorch version (`2.14.0+rocm10.1.0`), ROCm/HIP version (`7.16.26385`), and GPU device (`AMD Radeon(TM) 8060S Graphics`). |
| [`example_usage.py`](file:///C:/Dev/github/philippeback/lighton-ocr/example_usage.py) | End-to-end OCR transcription entrypoint. |
| [`doc/getting-amd-torch-to-work.md`](file:///C:/Dev/github/philippeback/lighton-ocr/doc/getting-amd-torch-to-work.md) | Focused operational guide for setting up and verifying AMD ROCm PyTorch with `uv`. |

---

## 8. Future Performance Optimization Roadmap (Windows ROCm)

1. **In-Place / Custom Fused `torch_recurrent_gated_delta_rule`:**
   Eliminating the per-token `transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)` copies and `torch.zeros_like(value)` allocations in `modeling_qwen3_5.py` (or compiling the recurrent step via a custom HIP/ATen extension) can reduce per-token kernel launches in `Qwen3_5GatedDeltaNet` from ~15 to 2–3.
2. **Weight-Only Quantization (`torchao` int8 / int4):**
   Because the Radeon 8060S (`gfx1151`) is memory-bandwidth bound during `batch_size=1` autoregressive decoding over LPDDR5X unified memory, quantizing linear projections from `bfloat16` (16 bits/weight) to `int8` or `int4` cuts memory read traffic by 2–4x, narrowing the gap with `llama.cpp` GGUF models.
3. **Windows Triton Support on ROCm:**
   Once AMD publishes `win_amd64` wheels for `triton` on `https://stable.repo.amd.com/rocm/whl-next/`, removing the `override-dependencies` constraint and adding `flash-linear-attention` to [`pyproject.toml`](file:///C:/Dev/github/philippeback/lighton-ocr/pyproject.toml) will automatically activate Tier-2 fused Triton kernels for both `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule`.
