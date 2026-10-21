import os
import re
import logging

# Silence MIOpen missing .fdb.txt warning on Windows ROCm before importing torch
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .utils import validate_image_path, logger

# Silence transformers hub_kernels fallback warnings on platforms without triton/causal_conv1d
logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)


class _RepeatTokenStoppingCriteria(StoppingCriteria):
    """
    Stop generation early if the model enters a degenerate single-token or 2-token
    repetition loop (e.g., repeating '█' or whitespace endlessly on progress bars).
    """

    def __init__(self, prompt_length: int, max_single_repeats: int = 24, max_pair_repeats: int = 16):
        super().__init__()
        self.prompt_length = prompt_length
        self.max_single_repeats = max_single_repeats
        self.max_pair_repeats = max_pair_repeats

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = input_ids.shape[1] - self.prompt_length
        if gen_len >= self.max_single_repeats:
            tail = input_ids[0, -self.max_single_repeats :]
            if torch.all(tail == tail[-1]):
                return True
        if gen_len >= self.max_pair_repeats * 2:
            tail_pairs = input_ids[0, -(self.max_pair_repeats * 2) :].view(self.max_pair_repeats, 2)
            if torch.all(tail_pairs == tail_pairs[-1]):
                return True
        return False


class LightOnOCR:
    """
    Wrapper around the LightOnOCR-3-4B model for easy OCR inference.
    """

    def __init__(
        self,
        model_id: str = "lightonai/LightOnOCR-3-4B",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        torch_dtype: torch.dtype | None = None,
        local_files_only: bool | None = None,
    ):
        """
        Load the processor and model.
        If local_files_only is None (default), tries loading from local Hugging Face
        cache first without any network calls, falling back to online download if not cached.
        """
        resolved_dtype = torch_dtype if torch_dtype is not None else dtype

        if local_files_only is None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, local_files_only=True
                )
                self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=resolved_dtype,
                    device_map=device_map,
                    local_files_only=True,
                )
                return
            except OSError:
                logger.info("Model not fully cached locally; fetching from Hugging Face Hub...")
                local_files_only = False

        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=resolved_dtype,
            device_map=device_map,
            local_files_only=local_files_only,
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
            resample = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((max_size, max_size), resample)

        return image

    def transcribe(self, image_input, prompt: str = "", max_new_tokens: int = 2048) -> str:
        """
        Perform OCR on the given image.

        Parameters
        ----------
        image_input : str or PIL.Image.Image
            Path to an image file or a PIL Image instance.
        prompt : str, optional
            Prompt mode: "" for plain markdown transcription, or "grounding" for layout boxes.
        max_new_tokens : int, optional
            Maximum number of tokens to generate.

        Returns
        -------
        str
            The decoded text.
        """
        image = self._preprocess(image_input)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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

        prompt_len = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList(
            [_RepeatTokenStoppingCriteria(prompt_length=prompt_len)]
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )

        generated_ids = output_ids[0, prompt_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        # Collapse any runaway repeated block characters (e.g. progress bars)
        text = re.sub(r"(█)\1{39,}", lambda m: m.group(1) * 40, text)
        return text.rstrip()
