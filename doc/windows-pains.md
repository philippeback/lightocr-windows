# Windows + AMD ROCm (`gfx1151`) Pains & Field Notes

A collection of real-world friction points ("paper cuts" and silent traps) encountered while getting `lightonai/LightOnOCR-3-4B` running natively on Windows 11 x64 with an AMD Radeon 8060S (`gfx1151`) GPU, `uv`, and Hugging Face `transformers`.

---

## 1. Global `python` vs. `uv` Virtual Environment Confusion
Running `python -c "import torch..."` directly in PowerShell hits the system Python (`2.14.1+cpu`) instead of `.venv`, reporting `GPU Available: False`.
* **Takeaway:** Always run via `uv run python ...` (or `uv run python rocm-check.py`) so `uv` executes inside `.venv`.

## 2. `uv pip install` Silently Gets Wiped Out by `uv run` / `uv sync`
AMD's official documentation tells you to run imperative `uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ ...` commands.
* **The Pain:** `uv pip install` does not update `pyproject.toml` or `uv.lock`. The very next time you run `uv run` or `uv sync`, `uv` sees `"torch>=2.0"` in `pyproject.toml`, resolves it against PyPI, and **silently replaces your 400 MB of ROCm wheels with `torch==2.14.1+cpu`**.
* **The Fix:** Put the `amd-rocm` index and exact `gfx1151` package pins directly into `pyproject.toml`.

## 3. PowerShell Bracket Escaping vs. `uv` Pre-Release Rejection
When installing `torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0`:
* In PowerShell, square brackets `[` `]` can be treated as wildcard globbing unless quoted carefully.
* Even when quoted, `uv` rejects `torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0` because its transitive dependency `amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0` is an alpha pre-release (`0.29.0a0`). Without `prerelease = "allow"` in `pyproject.toml` (or `--prerelease=allow` on the CLI), the whole install fails and leaves the old CPU `torch` in place.

## 4. Linux-Only `triton` Metadata Breaking Windows `uv lock`
When `uv lock` resolves `torch==2.14.0+rocm10.1.0` from AMD's `whl-next` index, the package metadata declares a hard dependency on `triton==3.8.0+git669b31ac.rocm10.1.0` without a `; sys_platform == 'linux'` marker.
* **The Pain:** AMD only publishes Linux wheels for `triton` on that index—there is no `win_amd64` wheel. So `uv lock` refuses to lock the project on Windows.
* **The Fix:** Force `uv` to treat `triton` as Linux-only in `pyproject.toml`:
  ```toml
  [tool.uv]
  environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]
  override-dependencies = [
      "triton; sys_platform == 'linux'",
  ]
  ```

## 5. Missing Windows Wheels for `flash-linear-attention` & `causal-conv1d`
`LightOnOCR-3-4B` uses Qwen3.5's hybrid `GatedDeltaNet` architecture, which looks for `causal-conv1d` and `flash-linear-attention` (`fla`) for fused kernels:
```text
[transformers] `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed...
[transformers] `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed...
```
* **The Pain:**
  * `flash-linear-attention` relies on **Triton** kernels, and Triton isn't shipped on Windows ROCm yet.
  * `causal-conv1d` has no prebuilt Windows ROCm `gfx1151` wheels (and building from source on Windows requires full HIP/MSVC CUDA-porting toolchains).
* **The Consequence:** Windows ROCm falls back to the pure-PyTorch reference implementation (`torch_recurrent_gated_delta_rule`). It still runs on the Radeon 8060S GPU (~90–94% `Compute 0` utilization), but launches ~15 small GPU kernels with `bfloat16 -> float32 -> bfloat16` casts per layer per token, making per-token decoding noticeably slower than a quantized `llama.cpp` server with fused C++ kernels.

## 6. MIOpen Missing Find-Database Warning (`gfx115128.HIP.fdb.txt`)
On the first convolution call, ROCm's MIOpen library logs:
```text
MIOpen: Warning [ParseAndLoadDb] File is unreadable: "...\.venv\Lib\site-packages\_rocm_sdk_libraries\bin\gfx115128.HIP.fdb.txt"
```
* **The Pain:** `_rocm_sdk_libraries` 10.1.0 on Windows does not bundle the pre-tuned `.fdb.txt` file for the 28-CU `gfx1151` (`gfx115128`).
* **The Fix:** It falls back to runtime heuristics automatically; setting `os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")` before `import torch` silences the warning.

## 7. Multimodal "Blind Model" Trap (`"The\n\nThe..."`)
Loading `lightonai/LightOnOCR-3-4B` with `AutoModelForCausalLM` instead of `Qwen3_5ForConditionalGeneration` only loads the 426 text-decoder weights and drops the vision tower. Popping `pixel_values` and `image_grid_thw` to silence `forward()` keyword errors leaves the model completely blind, causing it to repeat `"The\n\nThe..."` endlessly.

## 8. Runaway Progress-Bar Token Loops (`████...`)
When transcribing terminal screenshots containing ASCII/Unicode progress bars (`4 / 6 lines [████...`), greedy decoding gets trapped in an attractor state emitting `█` (`\u2588`) for the remaining ~1,900 tokens up to `max_new_tokens=2048`.
* **The Pain:** Even though the actual text on screen is only ~150 tokens, generating 1,900 extra `█` tokens via the un-fused Windows ROCm PyTorch fallback makes inference take **10x longer** than necessary.
* **The Fix:** Adding `_RepeatTokenStoppingCriteria` (stopping after 24 identical single tokens or 16 repeated 2-token pairs) cuts off the runaway progress bar immediately.

## 9. Chatty Hugging Face Hub & Root Logger Spam on Startup
* Calling `logging.basicConfig(level=logging.INFO)` on the root logger causes `httpx2` to spam `INFO:httpx2:HTTP Request: GET https://huggingface.co/...` on every startup.
* Even when the model weights are already downloaded locally, `from_pretrained()` makes 7+ online HTTP checks unless `local_files_only=True` (or `$env:HF_HUB_OFFLINE = "1"`) is used.
