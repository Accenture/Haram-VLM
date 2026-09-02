"""
HARAM-VLM Core Modules
Hallucination-Aware Resolution-Adaptive Vision-Language Model Components
"""

from .resolution_router import (
    ResolutionRouter,
    AdaptiveResolutionRouter
)

from .hallucination_predictor import (
    HallucinationPredictor,
    CalibratedHallucinationPredictor,
    HallucinationFeatures,
    create_predictor
)

from .adaptive_token_manager import (
    AdaptiveTokenManager,
    HierarchicalTokenManager,
    CompressionStrategy
)

__all__ = [
    # Resolution Router
    'ResolutionRouter',
    'AdaptiveResolutionRouter',

    # Hallucination Predictor
    'HallucinationPredictor',
    'CalibratedHallucinationPredictor',
    'HallucinationFeatures',
    'create_predictor',

    # Token Manager
    'AdaptiveTokenManager',
    'HierarchicalTokenManager',
    'CompressionStrategy'
]

# Version info
__version__ = '0.1.0'