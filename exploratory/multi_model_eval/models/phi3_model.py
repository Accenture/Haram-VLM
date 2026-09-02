"""
Phi-3 Vision Model Wrapper
==========================

Wrapper for Microsoft's Phi-3 Vision model.
Phi-3-Vision is a lightweight 4.2B parameter model optimized for edge deployment.
"""

import torch
from typing import Any, Optional, List, Dict
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)

from .base_model import BaseVLMWrapper, ModelConfig, ModelOutput


class Phi3VisionWrapper(BaseVLMWrapper):
    """
    Wrapper for Microsoft Phi-3 Vision model.

    Phi-3-Vision features:
    - 4.2B parameters (3.8B LLM + 0.4B vision)
    - 128K context length
    - Optimized for edge deployment
    - Strong performance despite small size
    """

    SUPPORTED_MODELS = {
        "microsoft/Phi-3-vision-128k-instruct": {
            "size": "4.2B",
            "context_length": 128000,
            "vision_encoder": "CLIP ViT-L/14"
        },
        "microsoft/Phi-3.5-vision-instruct": {
            "size": "4.2B",
            "context_length": 128000,
            "vision_encoder": "CLIP ViT-L/14",
            "version": "3.5"
        }
    }

    def __init__(self, config: ModelConfig):
        """Initialize Phi-3 Vision wrapper"""
        # Handle shorthand
        if config.model_name == "phi-3-vision":
            config.model_name = "microsoft/Phi-3-vision-128k-instruct"
        elif config.model_name == "phi-3.5-vision":
            config.model_name = "microsoft/Phi-3.5-vision-instruct"

        if config.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {config.model_name}")

        self.model_info = self.SUPPORTED_MODELS[config.model_name]
        super().__init__(config)

    def _load_model(self):
        """Load Phi-3 Vision model and processor"""
        print(f"Loading {self.config.model_name}...")

        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True  # Phi-3 may require custom code
            )

            # Phi-3 Vision is small enough to run without quantization
            # But we can add 8-bit quantization for even smaller memory footprint
            quantization_config = None
            if hasattr(self.config, 'use_8bit') and self.config.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.config.dtype
                )
                print("Using 8-bit quantization")

            # Load model
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=self.config.dtype,
                    _attn_implementation="eager"  # Don't use flash attention
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self.config.dtype,
                    device_map=self.config.device if self.config.device != "cpu" else None,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True,
                    _attn_implementation="eager"  # Don't use flash attention
                )

                if self.config.device == "cpu":
                    self.model = self.model.to(self.device)

            self.model.eval()

            print(f"Model loaded successfully on {self.config.device}")
            print(f"Context length: {self.model_info['context_length']:,} tokens")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _preprocess_image(self, image: Image.Image, resolution: int) -> Image.Image:
        """
        Preprocess image for Phi-3 Vision.

        Phi-3 Vision supports multiple resolutions but has an optimal range.

        Args:
            image: PIL Image
            resolution: Target resolution

        Returns:
            Resized PIL Image
        """
        # Phi-3 Vision works best with these resolutions
        supported_resolutions = [224, 336, 448, 672, 896]

        # Find closest supported resolution
        if resolution in supported_resolutions:
            target_res = resolution
        else:
            target_res = min(supported_resolutions, key=lambda x: abs(x - resolution))

        # Resize image
        return image.resize((target_res, target_res), Image.Resampling.LANCZOS)

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
            "is there", "are there", "can you see", "does the image"
        ])

        if is_yes_no:
            # Extract yes/no from response
            response_lower = response.lower()

            # Look for clear yes/no at the beginning
            if response_lower.startswith(('yes', 'no')):
                # Return just the first word
                return response.split()[0] if response.split() else response

            # Look for yes/no anywhere in the response
            if 'yes' in response_lower and 'no' not in response_lower:
                return 'yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'no'

            # If unclear, return the full response (will be handled by POPE evaluation)
            return response

        return response

    def _format_prompt(self, text_prompt: str, use_system: bool = True) -> str:
        """
        Format prompt for Phi-3 conversation format.

        Args:
            text_prompt: User's question/prompt
            use_system: Whether to include system prompt

        Returns:
            Formatted conversation prompt
        """
        # Check if this is a yes/no question (POPE-style)
        is_yes_no = any(phrase in text_prompt.lower() for phrase in [
            "is there", "are there", "can you see", "does the image"
        ])

        # Phi-3 uses a specific template
        if use_system:
            if is_yes_no:
                # More specific system prompt for yes/no questions
                system_prompt = "You are a helpful AI assistant. Answer the following question about the image with 'yes' or 'no'. Be accurate and do not hallucinate objects that are not present."
            else:
                system_prompt = "You are a helpful AI assistant that accurately describes images."
            formatted = f"<|system|>\n{system_prompt}<|end|>\n"
        else:
            formatted = ""

        # For yes/no questions, add explicit instruction
        if is_yes_no:
            formatted += f"<|user|>\n<|image|>\n{text_prompt} Please answer with 'yes' or 'no'.<|end|>\n<|assistant|>\n"
        else:
            formatted += f"<|user|>\n<|image|>\n{text_prompt}<|end|>\n<|assistant|>\n"

        return formatted

    def _generate_response(self,
                          image_input: Image.Image,
                          text_prompt: str,
                          return_attention: bool = False) -> ModelOutput:
        """
        Generate response using Phi-3 Vision model.

        Args:
            image_input: Preprocessed image
            text_prompt: Text prompt
            return_attention: Whether to return attention maps

        Returns:
            ModelOutput with generated text and metrics
        """
        # Create messages for chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ],
            }
        ]

        # Apply chat template - use fallback if not available
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (AttributeError, Exception):
            # Fallback to manual template formatting
            prompt = self._format_prompt(text_prompt, use_system=False)

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image_input,
            return_tensors="pt"
        ).to(self.device)

        # Count input tokens
        input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0

        # Generation parameters
        generation_args = {
            "max_new_tokens": self.config.max_length,
            "temperature": self.config.temperature,
            "do_sample": False if self.config.temperature == 0 else True,
        }

        # Add specific Phi-3 parameters if needed
        if self.config.temperature > 0:
            generation_args["top_p"] = 0.95

        # Generate response
        with torch.no_grad():
            if return_attention:
                generation_args.update({
                    "return_dict_in_generate": True,
                    "output_attentions": True
                })
                outputs = self.model.generate(**inputs, **generation_args)
                generated_ids = outputs.sequences
                attention_maps = outputs.attentions if hasattr(outputs, 'attentions') else None
            else:
                generated_ids = self.model.generate(**inputs, **generation_args)
                attention_maps = None

        # Decode generated text
        # Remove input tokens from the generated sequence
        generated_ids = generated_ids[:, input_token_count:]

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Calculate visual tokens
        visual_tokens = self.measure_token_usage(image_input, image_input.size[0])

        # Clean up response for yes/no questions
        cleaned_text = self._clean_response(generated_text.strip(), text_prompt)

        return ModelOutput(
            text=cleaned_text,
            tokens_used=visual_tokens + input_token_count,
            attention_maps=attention_maps
        )

    def measure_token_usage(self, image: Image.Image, resolution: int) -> int:
        """
        Calculate the number of visual tokens for Phi-3 Vision.

        Phi-3 Vision uses CLIP ViT-L/14 encoder:
        - Patch size: 14x14
        - Supports multiple resolutions
        """
        patch_size = 14

        # Calculate patches based on resolution
        num_patches = (resolution // patch_size) ** 2

        # Phi-3 may use compression, approximate token count
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

    def run_edge_optimization_test(self,
                                  image: Image.Image,
                                  prompt: str) -> Dict[str, Any]:
        """
        Test Phi-3's optimization for edge deployment.

        Measures performance at different configurations.

        Args:
            image: Test image
            prompt: Test prompt

        Returns:
            Dictionary with performance metrics
        """
        import time
        results = {}

        # Test different resolutions
        resolutions = [224, 336, 448]

        for res in resolutions:
            start_time = time.time()

            output = self.process(
                image,
                prompt,
                resolution=res,
                return_metrics=True
            )

            results[f"res_{res}"] = {
                "inference_time": time.time() - start_time,
                "tokens_used": output.tokens_used,
                "response_length": len(output.text)
            }

        return results