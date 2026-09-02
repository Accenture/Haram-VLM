"""
LLaVA Vision-Language Model Wrapper
====================================

Wrapper for LLaVA (Large Language and Vision Assistant) models.
Supports LLaVA-1.5 and LLaVA-1.6 variants.
"""

import torch
from typing import Any, Optional, List, Dict
from PIL import Image
import warnings

from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

from .base_model import BaseVLMWrapper, ModelConfig, ModelOutput


class LLaVAWrapper(BaseVLMWrapper):
    """
    Wrapper for LLaVA Vision-Language Models.

    Supports:
    - LLaVA-1.5 (7B, 13B variants)
    - LLaVA-1.6 (LLaVA-NeXT) (7B, 13B, 34B variants)
    """

    SUPPORTED_MODELS = {
        # LLaVA 1.5 models
        "llava-hf/llava-1.5-7b-hf": {
            "version": "1.5",
            "size": "7B",
            "class": LlavaForConditionalGeneration
        },
        "llava-hf/llava-1.5-13b-hf": {
            "version": "1.5",
            "size": "13B",
            "class": LlavaForConditionalGeneration
        },
        # LLaVA 1.6 (NeXT) models
        "llava-hf/llava-v1.6-mistral-7b-hf": {
            "version": "1.6",
            "size": "7B",
            "class": LlavaNextForConditionalGeneration
        },
        "llava-hf/llava-v1.6-vicuna-7b-hf": {
            "version": "1.6",
            "size": "7B",
            "class": LlavaNextForConditionalGeneration
        },
        "llava-hf/llava-v1.6-vicuna-13b-hf": {
            "version": "1.6",
            "size": "13B",
            "class": LlavaNextForConditionalGeneration
        },
        "llava-hf/llava-v1.6-34b-hf": {
            "version": "1.6",
            "size": "34B",
            "class": LlavaNextForConditionalGeneration,
            "requires_quantization": True
        }
    }

    def __init__(self, config: ModelConfig):
        """Initialize LLaVA wrapper"""
        if config.model_name not in self.SUPPORTED_MODELS:
            # Check if it's a shorthand
            shorthand_map = {
                "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
                "llava-1.6-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "llava-1.6-mistral": "llava-hf/llava-v1.6-mistral-7b-hf"
            }
            if config.model_name in shorthand_map:
                config.model_name = shorthand_map[config.model_name]
            else:
                raise ValueError(f"Unsupported model: {config.model_name}")

        self.model_info = self.SUPPORTED_MODELS[config.model_name]
        super().__init__(config)

    def _load_model(self):
        """Load LLaVA model and processor"""
        print(f"Loading {self.config.model_name}...")

        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )

            # Setup quantization if needed (for large models)
            quantization_config = None
            if self.model_info.get("requires_quantization", False) and self.config.device != "cpu":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.dtype,
                    bnb_4bit_use_double_quant=True
                )
                print("Using 4-bit quantization for large model")

            # Get the appropriate model class
            model_class = self.model_info["class"]

            # Load model
            if quantization_config:
                self.model = model_class.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=self.config.cache_dir
                )
            else:
                self.model = model_class.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self.config.dtype,
                    device_map=self.config.device if self.config.device != "cpu" else None,
                    cache_dir=self.config.cache_dir
                )

                if self.config.device == "cpu":
                    self.model = self.model.to(self.device)

            self.model.eval()

            # Set generation config
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.temperature = self.config.temperature
                self.model.generation_config.max_new_tokens = self.config.max_length
                self.model.generation_config.do_sample = False if self.config.temperature == 0 else True

            print(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image, resolution: int) -> Image.Image:
        """
        Preprocess image for LLaVA models.

        Args:
            image: PIL Image
            resolution: Target resolution

        Returns:
            Resized PIL Image
        """
        # LLaVA typically uses 336x336 for v1.5 and supports multiple resolutions for v1.6
        if self.model_info["version"] == "1.5":
            # LLaVA 1.5 uses fixed 336x336
            target_res = 336
        else:
            # LLaVA 1.6 supports multiple resolutions
            # Map requested resolution to supported ones
            supported_resolutions = [336, 672, 1008, 1344]
            target_res = min(supported_resolutions, key=lambda x: abs(x - resolution))

        return image.resize((target_res, target_res), Image.Resampling.LANCZOS)

    def _format_prompt(self, text_prompt: str) -> str:
        """
        Format prompt for LLaVA conversation format.

        Args:
            text_prompt: User's question/prompt

        Returns:
            Formatted conversation prompt
        """
        # Check if this is a yes/no question (POPE-style)
        is_yes_no = any(phrase in text_prompt.lower() for phrase in [
            "is there", "are there", "can you see", "does the image", "do you see"
        ])

        # LLaVA uses a specific conversation format
        if self.model_info["version"] == "1.5":
            # LLaVA 1.5 format
            if is_yes_no:
                return f"USER: <image>\n{text_prompt} Please answer with only 'yes' or 'no'.\nASSISTANT:"
            else:
                return f"USER: <image>\n{text_prompt}\nASSISTANT:"
        else:
            # LLaVA 1.6 format (supports multiple conversation templates)
            if is_yes_no:
                return f"<|im_start|>user\n<image>\n{text_prompt} Please answer with only 'yes' or 'no'.<|im_end|>\n<|im_start|>assistant\n"
            else:
                return f"<|im_start|>user\n<image>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _clean_response(self, response: str, prompt: str) -> str:
        """
        Clean up model response, especially for yes/no questions.

        Args:
            response: Raw model response
            prompt: Original prompt to determine question type

        Returns:
            Cleaned response text
        """
        # Check if this was a yes/no question
        is_yes_no = any(phrase in prompt.lower() for phrase in [
            "is there", "are there", "can you see", "does the image", "do you see"
        ])

        if is_yes_no:
            # Extract yes/no from response
            response_lower = response.lower()

            # Remove common prefixes that LLaVA might use
            response_lower = response_lower.replace("answer:", "").strip()
            response_lower = response_lower.replace("the answer is", "").strip()
            response_lower = response_lower.replace("based on the image,", "").strip()

            # Look for clear yes/no at the beginning
            if response_lower.startswith(('yes', 'no')):
                # Return just the first word
                first_word = response_lower.split()[0] if response_lower.split() else response_lower
                if first_word in ['yes', 'no']:
                    return first_word

            # Look for yes/no anywhere in the response
            if 'yes' in response_lower and 'no' not in response_lower:
                return 'yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'no'

            # Default to no if unclear
            return 'no'

        return response

    def _generate_response(self,
                          image_input: Image.Image,
                          text_prompt: str,
                          return_attention: bool = False) -> ModelOutput:
        """
        Generate response using LLaVA model.

        Args:
            image_input: Preprocessed image
            text_prompt: Text prompt
            return_attention: Whether to return attention maps

        Returns:
            ModelOutput with generated text and metrics
        """
        # Format the conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image"},
                ],
            },
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=image_input,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Count input tokens
        input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0

        # Generate response
        with torch.no_grad():
            if return_attention:
                # Generate with attention outputs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                    temperature=self.config.temperature,
                    do_sample=False if self.config.temperature == 0 else True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )

                generated_ids = outputs.sequences
                attention_maps = outputs.attentions if hasattr(outputs, 'attentions') else None
            else:
                # Standard generation
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                    temperature=self.config.temperature,
                    do_sample=False if self.config.temperature == 0 else True
                )
                attention_maps = None

        # Decode generated text
        # Remove input tokens from the generated sequence
        generated_ids = generated_ids[:, input_token_count:]

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Clean response for yes/no questions
        cleaned_text = self._clean_response(generated_text.strip(), text_prompt)

        # Calculate visual tokens
        visual_tokens = self.measure_token_usage(image_input, image_input.size[0])

        return ModelOutput(
            text=cleaned_text,
            tokens_used=visual_tokens + input_token_count,
            attention_maps=attention_maps
        )

    def measure_token_usage(self, image: Image.Image, resolution: int) -> int:
        """
        Calculate the number of visual tokens for LLaVA models.

        LLaVA tokenization:
        - v1.5: Fixed 24x24 = 576 patches (336x336 with 14x14 patches)
        - v1.6: Dynamic, supports multiple resolutions
        """
        if self.model_info["version"] == "1.5":
            # Fixed 576 visual tokens for LLaVA 1.5
            return 576
        else:
            # LLaVA 1.6 with dynamic resolution
            # Uses CLIP ViT-L/14 with 14x14 patch size
            patch_size = 14

            # Map to actual resolution used
            if resolution <= 336:
                actual_res = 336
            elif resolution <= 672:
                actual_res = 672
            elif resolution <= 1008:
                actual_res = 1008
            else:
                actual_res = 1344

            num_patches = (actual_res // patch_size) ** 2
            return num_patches + 1  # +1 for CLS token

    def get_supported_resolutions(self) -> List[int]:
        """Get list of resolutions supported by this model"""
        if self.model_info["version"] == "1.5":
            return [336]
        else:
            return [336, 672, 1008, 1344]