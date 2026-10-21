import os
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .utils import validate_image_path, logger

class LightOnOCR:
    """
    Wrapper around the LightOnOCR-3-4B model for easy OCR inference.
    """

    def __init__(self, model_id: str = "lightonai/LightOnOCR-3-4B",
                 device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.float16):
        """
        Load the processor and model.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    def _preprocess(self, image_input):
        """
        Load and preprocess an image.

        Parameters
        ----------
        image_input : str or PIL.Image.Image
            Path to an image file or a PIL Image instance.

        Returns
        -------
        PIL.Image.Image
            The preprocessed RGB image, resized if necessary.
        """
        if isinstance(image_input, str):
            if not validate_image_path(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input)
        else:
            image = image_input

        image = image.convert("RGB")

        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.ANTIALIAS)

        return image

    def transcribe(self, image_input):
        """
        Perform OCR on the given image.

        Parameters
        ----------
        image_input : str or PIL.Image.Image
            Path to an image file or a PIL Image instance.

        Returns
        -------
        str
            The decoded text.
        """
        image = self._preprocess(image_input)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return text
