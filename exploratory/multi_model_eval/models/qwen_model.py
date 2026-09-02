"""
Qwen Vision-Language Model Wrapper
===================================

Supports Qwen2-VL and Qwen3-VL models for hallucination evaluation.
"""

import torch
from typing import Any, Optional, List, Dict
from PIL import Image
import time

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)

from .base_model import BaseVLMWrapper, ModelConfig, ModelOutput


class QwenVLMWrapper(BaseVLMWrapper):
    """
    Wrapper for Qwen Vision-Language Models.

    Supports:
    - Qwen2-VL (2B, 7B variants)
    - Qwen3-VL (2B, 8B variants)
    """

    SUPPORTED_MODELS = {
        "Qwen/Qwen2-VL-2B-Instruct": {"size": "2B", "version": "2"},
        "Qwen/Qwen2-VL-7B-Instruct": {"size": "7B", "version": "2"},
        "Qwen/Qwen2.5-VL-7B-Instruct": {"size": "7B", "version": "2.5"},
        "Qwen/Qwen3-VL-2B-Instruct": {"size": "2B", "version": "3"},
        "Qwen/Qwen3-VL-8B-Instruct": {"size": "8B", "version": "3"}
    }

    def __init__(self, config: ModelConfig):
        """Initialize Qwen VLM wrapper"""
        # Validate model name
        if config.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {config.model_name}")

        self.model_info = self.SUPPORTED_MODELS[config.model_name]
        super().__init__(config)

    def _load_model(self):
        """Load Qwen model and processor"""
        print(f"Loading {self.config.model_name}...")

        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )

            # Determine model class based on version
            if self.model_info["version"] in ["2", "2.5"]:
                model_class = Qwen2VLForConditionalGeneration
            else:
                # Qwen3 uses AutoModelForCausalLM
                model_class = AutoModelForCausalLM

            # Load model
            self.model = model_class.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                device_map=self.config.device if self.config.device != "cpu" else None,
                cache_dir=self.config.cache_dir
            )

            if self.config.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            print(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image, resolution: int) -> Dict:
        """
        Preprocess image for Qwen models.

        Args:
            image: PIL Image
            resolution: Target resolution

        Returns:
            Preprocessed inputs dictionary
        """
        # Resize image to target resolution
        image_resized = image.resize((resolution, resolution), Image.Resampling.LANCZOS)

        return {"image": image_resized, "resolution": resolution}

    def _format_prompt(self, text_prompt: str, version: str) -> str:
        """
        Format prompt based on Qwen version.

        Args:
            text_prompt: User's question/prompt
            version: Qwen version (2, 2.5, or 3)

        Returns:
            Formatted prompt string
        """
        if version == "3":
            # Qwen3 format
            return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Qwen2/2.5 format
            return f"<|vision|>\n{text_prompt}"

    def _generate_response(self,
                          image_input: Dict,
                          text_prompt: str,
                          return_attention: bool = False) -> ModelOutput:
        """
        Generate response using Qwen model.

        Args:
            image_input: Preprocessed image dictionary
            text_prompt: Text prompt
            return_attention: Whether to return attention maps

        Returns:
            ModelOutput with generated text and metrics
        """
        image = image_input["image"]
        resolution = image_input["resolution"]

        # Format the prompt based on model version
        formatted_prompt = self._format_prompt(text_prompt, self.model_info["version"])

        # Prepare messages for Qwen3 or text for Qwen2
        if self.model_info["version"] == "3":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.device)

        else:
            # Qwen2/2.5 processing
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

        # Count input tokens
        input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0

        # Generate with attention if requested
        with torch.no_grad():
            if return_attention:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                    temperature=self.config.temperature,
                    do_sample=False if self.config.temperature == 0 else True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )
                attention_maps = outputs.attentions if hasattr(outputs, 'attentions') else None
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                    temperature=self.config.temperature,
                    do_sample=False if self.config.temperature == 0 else True
                )
                attention_maps = None

        # Decode the generated text
        if return_attention and hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences

        # Remove input tokens from generated sequence
        generated_ids = generated_ids[:, input_token_count:]

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Calculate visual tokens (approximate based on resolution)
        visual_tokens = self.measure_token_usage(image, resolution)

        return ModelOutput(
            text=generated_text.strip(),
            tokens_used=visual_tokens + input_token_count,
            attention_maps=attention_maps
        )

    def measure_token_usage(self, image: Image.Image, resolution: int) -> int:
        """
        Calculate the number of visual tokens for Qwen models.

        Qwen uses different tokenization strategies:
        - Qwen2: Dynamic patching with 14x14 base patches
        - Qwen3: Improved tokenization with variable patch sizes
        """
        if self.model_info["version"] == "3":
            # Qwen3 uses more efficient tokenization
            # Approximate formula based on documentation
            if resolution <= 224:
                return 256
            elif resolution <= 336:
                return 576
            elif resolution <= 448:
                return 1024
            elif resolution <= 672:
                return 2304
            else:
                return 4096
        else:
            # Qwen2 standard patching
            patch_size = 14
            num_patches = (resolution // patch_size) ** 2
            return num_patches + 1  # +1 for CLS token

    def run_attention_analysis(self,
                             image: Image.Image,
                             prompt: str,
                             resolutions: List[int]) -> Dict[int, float]:
        """
        Analyze attention entropy across different resolutions.

        Args:
            image: Input image
            prompt: Text prompt
            resolutions: List of resolutions to test

        Returns:
            Dictionary mapping resolution to attention entropy
        """
        entropy_results = {}

        for res in resolutions:
            output = self.process(
                image,
                prompt,
                resolution=res,
                return_attention=True
            )

            if output.attention_maps is not None:
                entropy = self.get_attention_entropy(output.attention_maps)
                entropy_results[res] = entropy
            else:
                entropy_results[res] = 0.0

        return entropy_results