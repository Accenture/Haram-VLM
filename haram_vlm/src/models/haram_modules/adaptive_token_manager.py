"""
Adaptive Token Manager for HARAM-VLM
Efficiently manages visual tokens based on hallucination risk and computational budget
Implements Progressive Visual Compression inspired by LLaVA-UHD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class CompressionStrategy(Enum):
    """Token compression strategies"""
    NONE = "none"                    # No compression
    PROGRESSIVE = "progressive"      # Gradual compression (LLaVA-UHD style)
    AGGRESSIVE = "aggressive"       # Maximum compression
    ADAPTIVE = "adaptive"           # Risk-based adaptive compression


class AdaptiveTokenManager(nn.Module):
    """
    Manages visual tokens efficiently based on:
    - Selected resolution
    - Hallucination risk
    - Computation budget
    - Query requirements

    Token scaling formula: tokens = (resolution / patch_size)^2
    For patch_size=14:
    - 224px → 256 tokens
    - 448px → 1024 tokens
    - 672px → 2304 tokens
    - 896px → 4096 tokens
    """

    def __init__(
        self,
        patch_size: int = 14,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        compression_ratio_min: float = 0.25,  # Minimum 25% tokens retained
        compression_ratio_max: float = 1.0,   # Maximum 100% tokens retained
        use_learned_selection: bool = True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.compression_ratio_min = compression_ratio_min
        self.compression_ratio_max = compression_ratio_max
        self.use_learned_selection = use_learned_selection

        # Token importance scoring network
        if use_learned_selection:
            self.importance_scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        # Progressive compression modules (inspired by LLaVA-UHD PVC)
        self.compression_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(2)  # 2 compression stages
        ])

        # Token merging layer
        self.token_merger = nn.Linear(hidden_dim * 2, hidden_dim)

        # Statistics tracking
        self.register_buffer('total_tokens_processed', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_tokens_retained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('compression_history', torch.zeros(100))  # Last 100 compressions
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))

    def calculate_token_budget(
        self,
        resolution: int,
        risk_score: float,
        max_budget: Optional[int] = None
    ) -> int:
        """
        Calculate token budget based on risk and constraints

        High risk → More tokens
        Low risk → Fewer tokens
        """
        # Base token count
        base_tokens = (resolution // self.patch_size) ** 2

        if max_budget is None:
            max_budget = base_tokens

        # Risk-based adjustment
        # High risk (>0.5) → Use more tokens
        # Low risk (<0.2) → Use fewer tokens
        if risk_score > 0.5:
            retention_ratio = 0.9 + 0.1 * (risk_score - 0.5) * 2  # 90-100%
        elif risk_score > 0.2:
            retention_ratio = 0.5 + 0.4 * (risk_score - 0.2) / 0.3  # 50-90%
        else:
            retention_ratio = 0.25 + 0.25 * risk_score / 0.2  # 25-50%

        retention_ratio = np.clip(retention_ratio, self.compression_ratio_min, self.compression_ratio_max)
        target_tokens = int(base_tokens * retention_ratio)

        return min(target_tokens, max_budget)

    def select_important_tokens(
        self,
        tokens: torch.Tensor,
        num_keep: int,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select most important tokens

        Args:
            tokens: [batch, num_tokens, hidden_dim]
            num_keep: Number of tokens to keep
            importance_scores: Optional pre-computed importance scores

        Returns:
            selected_tokens: [batch, num_keep, hidden_dim]
            indices: [batch, num_keep] indices of selected tokens
        """
        batch_size, num_tokens, hidden_dim = tokens.shape

        if num_keep >= num_tokens:
            # Keep all tokens
            indices = torch.arange(num_tokens).unsqueeze(0).expand(batch_size, -1)
            return tokens, indices.to(tokens.device)

        # Compute importance scores if not provided
        if importance_scores is None:
            if self.use_learned_selection:
                # Learned importance scoring
                importance_scores = self.importance_scorer(tokens).squeeze(-1)  # [batch, num_tokens]
            else:
                # Simple L2 norm as importance
                importance_scores = torch.norm(tokens, dim=-1)  # [batch, num_tokens]

        # Select top-k tokens
        _, indices = torch.topk(importance_scores, num_keep, dim=1)  # [batch, num_keep]

        # Gather selected tokens
        selected_tokens = torch.gather(
            tokens,
            1,
            indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )

        return selected_tokens, indices

    def progressive_compression(
        self,
        tokens: torch.Tensor,
        target_tokens: int,
        preserve_spatial: bool = True
    ) -> torch.Tensor:
        """
        Progressive Visual Compression (inspired by LLaVA-UHD)

        Gradually compresses tokens through multiple stages
        """
        batch_size, num_tokens, hidden_dim = tokens.shape

        if target_tokens >= num_tokens:
            return tokens

        # Stage 1: Initial compression to 50% if needed
        if target_tokens < num_tokens * 0.5:
            tokens = self.compression_layers[0](tokens)
            tokens, _ = self.select_important_tokens(tokens, num_tokens // 2)
            num_tokens = tokens.shape[1]

        # Stage 2: Final compression to target
        if target_tokens < num_tokens:
            tokens = self.compression_layers[1](tokens)
            tokens, indices = self.select_important_tokens(tokens, target_tokens)

            # Preserve spatial structure if requested
            if preserve_spatial:
                # Sort indices to maintain relative positions
                indices, _ = torch.sort(indices, dim=1)
                tokens = torch.gather(
                    tokens,
                    1,
                    indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
                )

        return tokens

    def aggressive_compression(
        self,
        tokens: torch.Tensor,
        compression_ratio: float = 0.25
    ) -> torch.Tensor:
        """
        Aggressive compression for low-risk scenarios
        """
        batch_size, num_tokens, hidden_dim = tokens.shape
        target_tokens = max(1, int(num_tokens * compression_ratio))

        # Direct selection of most important tokens
        compressed, _ = self.select_important_tokens(tokens, target_tokens)
        return compressed

    def adaptive_compression(
        self,
        tokens: torch.Tensor,
        risk_score: float,
        query_features: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Adaptive compression based on multiple factors
        """
        batch_size, num_tokens, hidden_dim = tokens.shape

        # Determine compression strategy
        if risk_score > 0.7:
            # High risk: minimal compression
            return self.progressive_compression(tokens, int(num_tokens * 0.9))
        elif risk_score > 0.3:
            # Medium risk: progressive compression
            target = int(num_tokens * (0.5 + 0.4 * (risk_score - 0.3) / 0.4))
            return self.progressive_compression(tokens, target)
        else:
            # Low risk: aggressive compression
            return self.aggressive_compression(tokens, 0.25 + risk_score)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        resolution: int,
        risk_score: float,
        strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
        max_budget: Optional[int] = None,
        query_features: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Manage visual tokens based on risk and budget

        Args:
            visual_tokens: [batch, num_tokens, hidden_dim]
            resolution: Current image resolution
            risk_score: Hallucination risk score [0, 1]
            strategy: Compression strategy to use
            max_budget: Maximum token budget
            query_features: Optional query-specific features

        Returns:
            Dictionary with:
                - tokens: Compressed tokens
                - compression_ratio: Ratio of tokens retained
                - num_tokens_original: Original token count
                - num_tokens_compressed: Compressed token count
                - strategy_used: Compression strategy applied
        """
        batch_size, num_tokens_original, hidden_dim = visual_tokens.shape

        # Calculate token budget
        token_budget = self.calculate_token_budget(resolution, risk_score, max_budget)

        # Apply compression strategy
        if strategy == CompressionStrategy.NONE:
            compressed_tokens = visual_tokens
        elif strategy == CompressionStrategy.PROGRESSIVE:
            compressed_tokens = self.progressive_compression(visual_tokens, token_budget)
        elif strategy == CompressionStrategy.AGGRESSIVE:
            compressed_tokens = self.aggressive_compression(visual_tokens)
        elif strategy == CompressionStrategy.ADAPTIVE:
            compressed_tokens = self.adaptive_compression(visual_tokens, risk_score, query_features)
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        num_tokens_compressed = compressed_tokens.shape[1]
        compression_ratio = num_tokens_compressed / num_tokens_original

        # Update statistics
        self.total_tokens_processed += num_tokens_original
        self.total_tokens_retained += num_tokens_compressed
        self.compression_history[self.history_idx] = compression_ratio
        self.history_idx = (self.history_idx + 1) % 100

        return {
            'tokens': compressed_tokens,
            'compression_ratio': compression_ratio,
            'num_tokens_original': num_tokens_original,
            'num_tokens_compressed': num_tokens_compressed,
            'strategy_used': strategy.value,
            'token_budget': token_budget
        }

    def get_efficiency_stats(self) -> Dict[str, float]:
        """
        Get efficiency statistics
        """
        if self.total_tokens_processed == 0:
            return {}

        avg_compression = self.compression_history[:min(self.history_idx, 100)].mean().item()
        total_savings = 1 - (self.total_tokens_retained.float() / self.total_tokens_processed.float())

        return {
            'total_tokens_processed': self.total_tokens_processed.item(),
            'total_tokens_retained': self.total_tokens_retained.item(),
            'average_compression_ratio': avg_compression,
            'total_token_savings': total_savings.item(),
            'tokens_saved': (self.total_tokens_processed - self.total_tokens_retained).item()
        }


class HierarchicalTokenManager(AdaptiveTokenManager):
    """
    Extended manager with hierarchical token organization
    Inspired by pyramid/hierarchical vision approaches
    """

    def __init__(self, *args, num_levels: int = 3, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_levels = num_levels

        # Hierarchical pooling layers
        self.hierarchy_poolers = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2, stride=2)
            for _ in range(num_levels - 1)
        ])

        # Level-wise importance weights
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

    def create_token_hierarchy(
        self,
        tokens: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Create hierarchical representation of tokens
        """
        batch_size, num_tokens, hidden_dim = tokens.shape
        hierarchy = [tokens]

        # Create pyramid levels
        current_tokens = tokens.transpose(1, 2)  # [B, hidden_dim, num_tokens]

        for pooler in self.hierarchy_poolers:
            current_tokens = pooler(current_tokens)
            hierarchy.append(current_tokens.transpose(1, 2))

        return hierarchy

    def select_from_hierarchy(
        self,
        hierarchy: List[torch.Tensor],
        target_tokens: int,
        risk_score: float
    ) -> torch.Tensor:
        """
        Select tokens from different hierarchy levels based on risk
        """
        # Higher risk → Use more fine-grained tokens
        # Lower risk → Use more coarse-grained tokens

        if risk_score > 0.7:
            # Mostly fine-grained
            weights = [0.8, 0.15, 0.05]
        elif risk_score > 0.3:
            # Balanced
            weights = [0.5, 0.35, 0.15]
        else:
            # Mostly coarse-grained
            weights = [0.2, 0.4, 0.4]

        selected_tokens = []
        remaining_budget = target_tokens

        for level, weight in enumerate(weights[:len(hierarchy)]):
            level_tokens = hierarchy[level]
            num_select = min(int(target_tokens * weight), remaining_budget)

            if num_select > 0:
                selected, _ = self.select_important_tokens(level_tokens, num_select)
                selected_tokens.append(selected)
                remaining_budget -= num_select

        # Concatenate tokens from different levels
        if selected_tokens:
            return torch.cat(selected_tokens, dim=1)
        else:
            return hierarchy[0][:, :target_tokens]  # Fallback


if __name__ == "__main__":
    # Test the token manager
    manager = AdaptiveTokenManager()

    # Create dummy visual tokens
    batch_size = 1
    num_tokens = 2304  # 672px resolution
    hidden_dim = 1024
    visual_tokens = torch.randn(batch_size, num_tokens, hidden_dim)

    # Test different risk scenarios
    test_scenarios = [
        (0.8, "High risk - minimal compression"),
        (0.5, "Medium risk - moderate compression"),
        (0.2, "Low risk - aggressive compression"),
    ]

    print("Adaptive Token Management Test:")
    print("-" * 60)

    for risk_score, description in test_scenarios:
        result = manager(
            visual_tokens,
            resolution=672,
            risk_score=risk_score,
            strategy=CompressionStrategy.ADAPTIVE
        )

        print(f"\n{description} (risk={risk_score}):")
        print(f"  Original tokens: {result['num_tokens_original']}")
        print(f"  Compressed tokens: {result['num_tokens_compressed']}")
        print(f"  Compression ratio: {result['compression_ratio']:.1%}")
        print(f"  Tokens saved: {result['num_tokens_original'] - result['num_tokens_compressed']}")

    # Print efficiency stats
    stats = manager.get_efficiency_stats()
    if stats:
        print(f"\nOverall Efficiency Stats:")
        print(f"  Total tokens processed: {stats['total_tokens_processed']}")
        print(f"  Total tokens retained: {stats['total_tokens_retained']}")
        print(f"  Average compression: {stats['average_compression_ratio']:.1%}")
        print(f"  Total savings: {stats['total_token_savings']:.1%}")