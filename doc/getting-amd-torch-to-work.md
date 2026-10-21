# Getting AMD ROCm PyTorch (`gfx1151`) to Work with `uv` on Windows

This document explains how to configure `uv` and `pyproject.toml` to install and lock AMD ROCm 10.1.0 PyTorch wheels for `gfx1151` (e.g., Ryzen AI Max / Radeon 8060S / Strix Halo) on Windows, and why manual `uv pip install` commands fail or get overwritten.

---

## Why `uv pip install` Alone Doesn't Work

If you run the standard AMD installation commands inside a `uv`-managed project:

```powershell
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1151]==10.1.0"
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.14.0+rocm10.1.0" "torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0" "torchaudio==2.11.0.3+rocm10.1.0"
```

you will hit three separate issues:

### 1. Pre-release transitive dependency rejection
`torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0` depends on `amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0`. Because `0.29.0a0` is an alpha pre-release, `uv` rejects the resolution unless pre-releases are explicitly allowed (`--prerelease=allow` on the CLI, or `prerelease = "allow"` in `pyproject.toml`).

### 2. `uv pip install` does not update `pyproject.toml`
`uv pip install` modifies `.venv` imperative-style without recording the custom index or version pins in `pyproject.toml` or `uv.lock`. The next time you run `uv sync` or `uv run`, `uv` reconciles `.venv` against `pyproject.toml` (`torch>=2.0` from PyPI) and **overwrites** the ROCm build with the CPU-only build (`torch==2.14.1+cpu`).

### 3. Unconditional `triton` dependency during `uv lock`
When `uv lock` resolves `torch==2.14.0+rocm10.1.0` from `https://stable.repo.amd.com/rocm/whl-next/`, the package metadata declares a dependency on `triton==3.8.0+git669b31ac.rocm10.1.0` without an OS marker. Because `triton` only publishes Linux wheels on the AMD index, `uv lock` fails on Windows (`sys_platform == 'win32'`) unless `triton` is restricted to Linux via `override-dependencies`.

---

## Working `pyproject.toml` Configuration

To make `uv lock`, `uv sync`, and `uv run` preserve the ROCm `gfx1151` stack on Windows (Python 3.13), configure `pyproject.toml` as follows:

```toml
[project]
name = "lighton-ocr"
version = "0.1.0"
description = "OCR model for transcribing printed and handwritten text."
requires-python = ">=3.13,<3.14"
dependencies = [
    "rocm[libraries,device-gfx1151]==10.1.0",
    "torch[device-gfx1151]==2.14.0+rocm10.1.0",
    "torchvision[device-gfx1151]==0.29.0a0+rocm10.1.0",
    "torchaudio==2.11.0.3+rocm10.1.0",
    "transformers>=4.30",
    "accelerate>=0.20",
    "Pillow>=9.0",
]

[dependency-groups]
dev = [
    "pytest",
]

[tool.uv]
# Required for amd-torchvision-device-gfx1151==0.29.0a0+rocm10.1.0
prerelease = "allow"
# Restrict lockfile resolution to Windows x64 where these wheels are used
environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]
# Prevent Linux-only triton requirement in torch metadata from breaking Windows resolution
override-dependencies = [
    "triton; sys_platform == 'linux'",
]

[[tool.uv.index]]
name = "amd-rocm"
url = "https://stable.repo.amd.com/rocm/whl-next/"
```

---

## Syncing and Verifying the Installation

1. **Lock and sync the virtual environment:**

   ```powershell
   uv lock
   uv sync
   ```

2. **Verify ROCm / HIP and GPU detection:**

   Always run Python via `uv run` (or `.venv\Scripts\python.exe`) so you use the project's virtual environment rather than the global system Python:

   ```powershell
   uv run python rocm-check.py
   ```

   Expected output:

   ```text
   PyTorch: 2.14.0+rocm10.1.0
   ROCm/HIP: 7.16.26385
   GPU Available: True
   Device: AMD Radeon(TM) 8060S Graphics
   ```

   > **Note:** Running bare `python rocm-check.py` in PowerShell without activating `.venv` or prefixing with `uv run` will invoke the system Python installation (which may still have `2.14.1+cpu`).
