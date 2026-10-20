# Inference Guide

The following example demonstrates how to load the model and perform OCR on an image.

```python
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Load processor and model
processor = AutoProcessor.from_pretrained("lightonai/LightOnOCR-3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "lightonai/LightOnOCR-3-4B",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load and preprocess image
image = Image.open("sample.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

# Decode
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
```

**Preprocessing**  
- Images should be RGB and resized to the model’s expected resolution (e.g., 224 × 224) if necessary.  
- No additional normalization is required; the processor handles it internally.

**Post‑processing**  
- The output token IDs are decoded using the processor’s tokenizer.  
- `skip_special_tokens=True` removes any special tokens added during training.
