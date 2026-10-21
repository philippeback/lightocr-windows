# LightOn-OCR (`lighton-ocr`)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-2.14.0%2Brocm10.1.0-EE4C2C.svg)](https://rocm.docs.amd.com/)
[![AMD Architecture](https://img.shields.io/badge/AMD_GPU-gfx1151_(Radeon_8060S)-ED1C24.svg)](https://www.amd.com/)
[![Model](https://img.shields.io/badge/HuggingFace-lightonai%2FLightOnOCR--3--4B-FFD21E.svg)](https://huggingface.co/lightonai/LightOnOCR-3-4B)
[![Managed with uv](https://img.shields.io/badge/uv-managed-de5fe9.svg)](https://docs.astral.sh/uv/)

Local Optical Character Recognition (OCR) and document transcription engine powered by [**`lightonai/LightOnOCR-3-4B`**](https://huggingface.co/lightonai/LightOnOCR-3-4B) (`Qwen3_5ForConditionalGeneration`), configured out-of-the-box for **Windows x64 + AMD ROCm 10.1.0 (`gfx1151` / Radeon 8060S Graphics)**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Installation](#installation)
- [Verifying AMD ROCm GPU Detection](#verifying-amd-rocm-gpu-detection)
- [Quickstart & Usage](#quickstart--usage)
  - [Plain Markdown Transcription](#plain-markdown-transcription)
  - [Visual Layout Grounding](#visual-layout-grounding)
  - [Offline Mode](#offline-mode)
- [Project Structure](#project-structure)
- [Architecture & Documentation](#architecture--documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`lighton-ocr` provides a clean Python interface (`LightOnOCR`) around the 4-billion-parameter `LightOnOCR-3-4B` vision-language model for transcribing printed documents, code screenshots, terminal outputs, and handwritten text locally on AMD Radeon GPUs under Windows.

It solves common pitfalls when running Qwen3.5-based multimodal models on Windows ROCm with `uv`, including declarative lockfile resolution for `gfx1151` wheels, offline-first Hugging Face cache loading, `bfloat16` multimodal tensor routing, and repetition-loop stopping criteria on visual progress bars.

---

## Key Features

- **Native Windows AMD ROCm (`gfx1151`) Acceleration:** Pre-configured `pyproject.toml` for AMD's `rocm==10.1.0` and `torch==2.14.0+rocm10.1.0` wheels without WSL2 overhead.
- **Zero-Network Offline Startup:** Automatically loads cached model and processor weights (`local_files_only=True` fast-path) with zero Hugging Face Hub HTTP calls after initial download.
- **Repetition-Loop Guardrails:** Includes `_RepeatTokenStoppingCriteria` to detect and stop runaway single-token or 2-token loops (such as ASCII/Unicode progress bars `████...`) immediately.
- **Clean Runtime Output:** Silences benign MIOpen find-db warnings (`gfx115128.HIP.fdb.txt`), `transformers` kernel fallback notices, and root-logger `httpx2` noise.
- **Dual Prompt Modes:** Supports both clean Markdown transcription (`prompt=""`) and structured bounding-box layout grounding (`prompt="grounding"`).

---

## Hardware & Software Requirements

| Requirement | Minimum / Target Specification |
| :--- | :--- |
| **Operating System** | Windows 11 x64 (`win32` / `AMD64`) |
| **GPU** | AMD Radeon(TM) 8060S Graphics (`gfx1151` / Ryzen AI Max / Strix Halo) or compatible ROCm GPU |
| **VRAM / Unified Memory** | >= 12 GB Dedicated GPU Memory (~8–11 GB used in `bfloat16`) |
| **Python** | Python `3.13.x` (`>=3.13,<3.14`) |
| **Package Manager** | [`uv`](https://docs.astral.sh/uv/) |

---

## Installation

1. **Clone the repository:**

   ```powershell
   git clone https://github.com/philippeback/lightocr-windows.git
   cd lightocr-windows
   ```

2. **Lock and synchronize the `uv` virtual environment:**

   ```powershell
   uv lock
   uv sync
   ```

   > **Important:** Always use `uv sync` and `uv run` rather than imperative `uv pip install` commands so `uv` preserves the AMD ROCm index wheels defined in [`pyproject.toml`](./pyproject.toml).

---

## Verifying AMD ROCm GPU Detection

Run the included diagnostic script via `uv run`:

```powershell
uv run python rocm-check.py
```

Expected output on an AMD Radeon 8060S (`gfx1151`) system:

```text
PyTorch: 2.14.0+rocm10.1.0
ROCm/HIP: 7.16.26385
GPU Available: True
Device: AMD Radeon(TM) 8060S Graphics
```

---

## Quickstart & Usage

### Plain Markdown Transcription

Run the example script:

```powershell
uv run python example_usage.py
```

Or use `LightOnOCR` directly in your Python code:

```python
from lightocr import LightOnOCR

ocr = LightOnOCR()
text = ocr.transcribe("path/to/image.png")
print(text)
```

### Visual Layout Grounding

Pass `prompt="grounding"` to extract text with structural layout bounding boxes:

```python
from lightocr import LightOnOCR

ocr = LightOnOCR()
grounded_output = ocr.transcribe("path/to/document.png", prompt="grounding")
print(grounded_output)
```

### Offline Mode

Once the weights have been downloaded once to your local Hugging Face cache (`~/.cache/huggingface/hub`), `LightOnOCR()` automatically loads them offline without making any HTTP requests. To strictly forbid network fallback even on a cache miss:

```python
ocr = LightOnOCR(local_files_only=True)
```

---

## Project Structure

```text
lighton-ocr/
├── doc/
│   └── getting-amd-torch-to-work.md  # Guide for uv + AMD ROCm gfx1151 setup
├── lightocr/
│   ├── __init__.py                   # Package export (LightOnOCR)
│   ├── engine.py                     # Core multimodal OCR inference engine & stopping criteria
│   └── utils.py                      # File validation & scoped package logger
├── example_usage.py                  # CLI example script
├── pyproject.toml                    # Declarative uv configuration & AMD ROCm wheel index
├── rocm-check.py                     # GPU & ROCm/HIP verification utility
├── sdi.md                            # Software Design and Implementation architecture doc
└── README.md                         # Project documentation
```

---

## Architecture & Documentation

- **[Software Design and Implementation (`sdi.md`)](./sdi.md):** Deep-dive architectural reference covering hardware taxonomy, `uv` dependency resolution internals, `Qwen3_5ForConditionalGeneration` multimodal routing, `Qwen3_5GatedDeltaNet` kernel dispatch on Windows ROCm vs. `llama.cpp`, and autoregressive stopping criteria mathematics.
- **[Getting AMD ROCm PyTorch to Work (`doc/getting-amd-torch-to-work.md`)](./doc/getting-amd-torch-to-work.md):** Step-by-step operational guide for configuring `uv` with AMD's `whl-next` index on Windows.

---

## Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| `GPU Available: False` / `PyTorch: 2.14.1+cpu` | Ran global `python` instead of `.venv`, or overwrote `.venv` with PyPI `torch` | Prefix commands with `uv run python ...` and run `uv sync` to restore ROCm wheels. |
| `uv lock` fails on `triton` or `amd-torchvision-device-gfx1151` | Missing `override-dependencies` for Linux-only `triton` or `prerelease = "allow"` | Ensure `[tool.uv]` in [`pyproject.toml`](./pyproject.toml) includes both settings (see [`doc/getting-amd-torch-to-work.md`](./doc/getting-amd-torch-to-work.md)). |
| Output repeats `"The\n\nThe..."` endlessly | Model loaded with `AutoModelForCausalLM` or `pixel_values` stripped | Fixed in [`lightocr/engine.py`](./lightocr/engine.py) by using `Qwen3_5ForConditionalGeneration` in `bfloat16` with `apply_chat_template`. |

---

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`).
2. Ensure the environment is synced (`uv sync`) and GPU verification passes (`uv run python rocm-check.py`).
3. Run tests with `uv run pytest`.
4. Commit your changes with clear messages and open a Pull Request.

---

## License

This project is provided under the [MIT License](https://opensource.org/licenses/MIT). Model weights (`lightonai/LightOnOCR-3-4B`) are subject to their upstream license on [Hugging Face](https://huggingface.co/lightonai/LightOnOCR-3-4B).
