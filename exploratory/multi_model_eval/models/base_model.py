"""
Base Model Wrapper for Vision-Language Models
==============================================

Provides a unified interface for different VLM architectures.
All model implementations should inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from PIL import Image
import time
from pathlib import Path


@dataclass
class ModelOutput:
    """Structured output from VLM models"""
    text: str
    confidence: Optional[float] = None
    attention_maps: Optional[torch.Tensor] = None
    tokens_used: Optional[int] = None
    inference_time: Optional[float] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_length: int = 512
    temperature: float = 0.0  # Use greedy decoding for consistency
    use_flash_attention: bool = False
    cache_dir: Optional[str] = None


class BaseVLMWrapper(ABC):
    """
    Abstract base class for Vision-Language Model wrappers.

    This class provides a consistent interface for:
    - Loading and initializing models
    - Processing images at different resolutions
    - Generating responses with detailed metrics
    - Extracting attention maps and internal states
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load the model, processor, and tokenizer"""
        pass

    @abstractmethod
    def _preprocess_image(self, image: Image.Image, resolution: int) -> Any:
        """
        Preprocess image to the specified resolution.

        Args:
            image: PIL Image
            resolution: Target resolution (assumes square for simplicity)

        Returns:
            Preprocessed image ready for model input
        """
        pass

    @abstractmethod
    def _generate_response(self,
                          image_input: Any,
                          text_prompt: str,
                          return_attention: bool = False) -> ModelOutput:
        """
        Generate response from the model.

        Args:
            image_input: Preprocessed image
            text_prompt: Text prompt/question
            return_attention: Whether to return attention maps

        Returns:
            ModelOutput with generated text and optional metadata
        """
        pass

    def process(self,
                image: Image.Image,
                prompt: str,
                resolution: int = 336,
                return_attention: bool = False,
                return_metrics: bool = True) -> ModelOutput:
        """
        Main processing method for VLM inference.

        Args:
            image: Input PIL Image
            prompt: Text prompt/question
            resolution: Image resolution to use
            return_attention: Whether to extract attention maps
            return_metrics: Whether to compute inference metrics

        Returns:
            ModelOutput with results and metrics
        """
        # Start timing
        start_time = time.time()

        # Preprocess image
        image_input = self._preprocess_image(image, resolution)

        # Generate response
        output = self._generate_response(
            image_input,
            prompt,
            return_attention=return_attention
        )

        # Add timing if requested
        if return_metrics:
            output.inference_time = time.time() - start_time

        return output

    def batch_process(self,
                     images: List[Image.Image],
                     prompts: List[str],
                     resolution: int = 336,
                     batch_size: int = 8) -> List[ModelOutput]:
        """
        Process multiple images in batches.

        Args:
            images: List of PIL Images
            prompts: List of prompts (same length as images)
            resolution: Image resolution to use
            batch_size: Batch size for processing

        Returns:
            List of ModelOutput objects
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            for img, prompt in zip(batch_images, batch_prompts):
                output = self.process(img, prompt, resolution)
                results.append(output)

        return results

    def get_attention_entropy(self, attention_maps: torch.Tensor) -> float:
        """
        Calculate attention entropy as a measure of attention diffusion.

        Higher entropy indicates more diffused attention (potentially worse).

        Args:
            attention_maps: Attention weight tensor

        Returns:
            Average entropy across all attention heads
        """
        if attention_maps is None:
            return 0.0

        # Ensure attention maps are probabilities
        if attention_maps.dim() > 2:
            # Average over heads and layers if necessary
            attention_maps = attention_maps.mean(dim=0)

        # Calculate entropy: -sum(p * log(p))
        eps = 1e-10
        attention_probs = torch.softmax(attention_maps, dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + eps)).sum(dim=-1)

        return entropy.mean().item()

    def extract_object_attention(self,
                                attention_maps: torch.Tensor,
                                text_tokens: List[str]) -> Dict[str, float]:
        """
        Extract attention weights for specific object tokens.

        Args:
            attention_maps: Attention weight tensor
            text_tokens: List of decoded text tokens

        Returns:
            Dictionary mapping object words to their average attention
        """
        # This is a simplified version - real implementation would need
        # proper token-to-word mapping
        object_attention = {}

        # Common object words to track
        object_words = ['cat', 'dog', 'car', 'person', 'chair', 'table',
                       'bird', 'tree', 'building', 'sky', 'grass']

        for i, token in enumerate(text_tokens):
            for obj in object_words:
                if obj.lower() in token.lower():
                    if attention_maps is not None and i < attention_maps.shape[-1]:
                        object_attention[obj] = attention_maps[:, i].mean().item()

        return object_attention

    def measure_token_usage(self,
                          image: Image.Image,
                          resolution: int) -> int:
        """
        Measure the number of visual tokens used for an image at given resolution.

        Args:
            image: Input PIL Image
            resolution: Target resolution

        Returns:
            Number of visual tokens
        """
        # This is model-specific and should be overridden
        # Default assumes patch size of 14 (common for CLIP-based models)
        patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches + 1  # +1 for [CLS] token

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.config.model_name}, device={self.config.device})"

    def cleanup(self):
        """Free up GPU memory"""
        if self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()