# Notes & Gotchas

- The model is trained on a mix of printed and handwritten text; performance may drop on extremely low‑contrast images.
- Maximum image size is 1024 × 1024 pixels; larger images should be resized or tiled.
- The model supports Latin, Cyrillic, Arabic, and several Asian scripts, but does not cover all Unicode blocks.
- When using `device_map="auto"`, ensure that the `accelerate` config is set to use FP16 for optimal memory usage.
- The tokenizer may produce special tokens like `<pad>`; use `skip_special_tokens=True` to clean the output.
