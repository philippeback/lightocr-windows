# LightOnOCR-3-4B

LightOnOCR-3-4B is a 3‑4 B parameter OCR model developed by LightOnAI. It is designed to transcribe printed and handwritten text from images with high accuracy across multiple languages. The model builds on the LightOn transformer architecture, optimized for efficient inference on GPUs with limited VRAM.

## Architecture

- Transformer‑based encoder‑decoder with 24 layers, 16 attention heads, and a hidden size of 2048.
- Uses LightOn’s lightweight attention mechanism to reduce memory footprint.
- Trained on a large multilingual OCR corpus (~10 M images) covering Latin, Cyrillic, Arabic, and many Asian scripts.

## Use Cases

- Document digitization (PDF, scanned forms).
- Real‑time text extraction in mobile or embedded devices.
- Multilingual OCR for international applications.
