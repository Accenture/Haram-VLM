"""
HARAM training integration.

The standalone HARAM modules (ResolutionRouter, HallucinationPredictor,
AdaptiveTokenManager) were written for single-sample, string-input inference and
cannot be plugged into a batched HuggingFace Trainer loop directly. This module
adapts them into a multi-task training wrapper around the base Phi-3-Vision model:

    total_loss = lm_loss
               + w_halluc * MSE(predicted_risk, expected_hallucination_risk)
               + w_router * CE(resolution_logits, resolution_bucket)
               + w_token  * risk_modulated_importance_regularizer

All three modules' real trainable sub-networks participate, supervised by the
per-sample `metadata` already present in the training JSON (resolution,
expected_hallucination_risk, query_type).

Scope (v1): the token manager is wired as a NON-DESTRUCTIVE regularizer over the
visual hidden states — it does not drop tokens, because doing so would break
Phi-3-Vision's fixed image-placeholder alignment within a single supervised
forward pass. Destructive compression / live resolution re-encoding require a
multi-pass or RL training loop and are intentionally left for a future version.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.haram_modules import (
    HallucinationPredictor,
    ResolutionRouter,
    AdaptiveTokenManager,
)

# Resolution buckets the router classifies over (must match ResolutionRouter.SUPPORTED_RESOLUTIONS)
RES_BUCKETS = [224, 336, 448, 672, 896]


def query_type_to_id(query_type: str) -> int:
    """Map a metadata query_type string to a 4-way class id used by the predictor features."""
    s = (query_type or "").lower()
    if "yes_no" in s or "yes/no" in s:
        return 0
    if "count" in s:
        return 1
    if "descri" in s:
        return 2
    return 3


def _resolution_bucket(resolution: torch.Tensor) -> torch.Tensor:
    """[B] int resolutions -> [B] nearest-bucket indices into RES_BUCKETS."""
    buckets = torch.tensor(RES_BUCKETS, device=resolution.device, dtype=torch.float32)
    diff = (resolution.float().unsqueeze(1) - buckets.unsqueeze(0)).abs()
    return diff.argmin(dim=1)


class HaramHead(nn.Module):
    """Holds the three HARAM modules + a projection, and exposes batched differentiable calls."""

    def __init__(self, hidden_size: int, num_features: int = 16, router_dim: int = 768):
        super().__init__()
        self.num_features = num_features

        # Real HARAM modules (their sub-networks are what actually train here).
        self.predictor = HallucinationPredictor(
            hidden_dim=256, num_features=num_features, use_learned_correction=True
        )
        self.router = ResolutionRouter(
            hidden_dim=router_dim, use_image_features=False, use_text_encoder=False
        )
        self.token_manager = AdaptiveTokenManager(hidden_dim=hidden_size)

        # Projects the host VLM's pooled hidden state down to the router's space.
        self.text_proj = nn.Linear(hidden_size, router_dim)

        # v1 uses only the token manager's importance scorer (non-destructive
        # regularizer). Freeze its heavy, currently-unused transformer compression
        # stack so it stays out of the optimizer / optimizer-state. These are
        # reserved for a future destructive-compression (multi-pass) version.
        for p in self.token_manager.compression_layers.parameters():
            p.requires_grad = False
        for p in self.token_manager.token_merger.parameters():
            p.requires_grad = False

        # efficiency_weight is an inference-time resolution penalty knob; route_logits()
        # (training path) doesn't use it, so freeze it to keep the optimizer clean.
        self.router.efficiency_weight.requires_grad = False

    def build_features(self, resolution: torch.Tensor, query_type_id: torch.Tensor) -> torch.Tensor:
        """Batched, differentiable reconstruction of HallucinationPredictor.extract_features.

        Feature layout matches the original module (16 dims). Only the network is
        learned; the features themselves are deterministic functions of metadata.
        """
        B = resolution.shape[0]
        device = resolution.device
        res = resolution.float()

        feats = torch.zeros(B, self.num_features, device=device)
        feats[:, 0] = res / 896.0                                   # normalized resolution
        feats[:, 1] = 1.2 * torch.exp(-0.0035 * res)                # validated base-risk curve
        feats[:, 2] = 1.0                                           # token density (assumed nominal)
        oh = F.one_hot(query_type_id.clamp(0, 3), num_classes=4).float()
        feats[:, 3:7] = oh                                          # query-type one-hot
        feats[:, 7] = 0.1                                           # normalized query length (default)
        # feats[:, 8:12] specific-risk indicator flags -> 0 (unknown at train time)
        feats[:, 12] = 0.5                                          # image clarity (default)
        feats[:, 13] = 0.5                                          # image complexity (default)
        feats[:, 14] = 0.25                                         # normalized object count (default)
        return feats

    def predict_risk(self, resolution: torch.Tensor, query_type_id: torch.Tensor):
        """Full predictor path -> (final_risk, confidence), both [B], differentiable.

        Mirrors HallucinationPredictor.forward: blends the empirical base-risk curve
        (modulated by the learned correction network) with the learned risk head, and
        also returns the confidence head's output. This exercises feature_encoder,
        risk_head, correction_network AND confidence_head.
        """
        # Match the predictor's parameter dtype (fp32 single-GPU, bf16 under DeepSpeed).
        pdtype = next(self.predictor.parameters()).dtype
        feats = self.build_features(resolution, query_type_id).to(pdtype)
        encoded = self.predictor.feature_encoder(feats)

        # Upcast the small head outputs to fp32 for stable, dtype-safe loss math.
        learned_risk = self.predictor.risk_head(encoded).squeeze(-1).float()       # [B] in [0,1]
        confidence = self.predictor.confidence_head(encoded).squeeze(-1).float()   # [B] in [0,1]
        correction = self.predictor.correction_network(feats).squeeze(-1).float()  # [B] in [-1,1]

        base_risk = (1.2 * torch.exp(-0.0035 * resolution.float())).clamp(0.01, 0.5)
        adjusted_risk = base_risk * (1.0 + 0.5 * correction)
        final_risk = (0.7 * adjusted_risk + 0.3 * learned_risk).clamp(0.0, 1.0)
        return final_risk, confidence

    def router_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        rdtype = next(self.text_proj.parameters()).dtype
        return self.router.route_logits(self.text_proj(pooled.to(rdtype)))     # [B, 5]

    def token_importance(self, hidden: torch.Tensor) -> torch.Tensor:
        idtype = next(self.token_manager.importance_scorer.parameters()).dtype
        return self.token_manager.importance_scorer(hidden.to(idtype)).squeeze(-1)  # [B, T] in [0,1]


class HaramForTraining(nn.Module):
    """Wraps a (possibly LoRA/PEFT) Phi-3-Vision model with HARAM auxiliary heads."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        w_halluc: float = 0.3,
        w_router: float = 0.3,
        w_token: float = 0.05,
    ):
        super().__init__()
        self.base = base_model
        self.head = HaramHead(hidden_size)
        self.config = base_model.config
        self.w_halluc = w_halluc
        self.w_router = w_router
        self.w_token = w_token

    # --- delegations the HF Trainer may call on the top-level model ---
    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.base.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.base.gradient_checkpointing_disable(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_sizes=None,
        resolution=None,
        halluc_risk=None,
        query_type_id=None,
        **kwargs,
    ):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            use_cache=False,
        )
        lm_loss = outputs.loss
        total = lm_loss

        aux_log = {}
        if resolution is not None and halluc_risk is not None and query_type_id is not None:
            # Pool the final multimodal hidden state over attended positions. Kept in the
            # model's compute dtype; the head methods cast to their own param dtype.
            hidden = outputs.hidden_states[-1]                     # [B, T, H]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

            # 1) Hallucination predictor: regress the validated expected risk via the
            #    full empirical+learned+correction path, and self-supervise the
            #    confidence head against the predictor's own accuracy.
            risk_pred, conf_pred = self.head.predict_risk(resolution, query_type_id)
            risk_target = halluc_risk.float()
            risk_loss = F.mse_loss(risk_pred, risk_target)
            conf_target = (1.0 - (risk_pred.detach() - risk_target).abs()).clamp(0.0, 1.0)
            conf_loss = F.mse_loss(conf_pred, conf_target)
            halluc_loss = risk_loss + 0.1 * conf_loss

            # 2) Resolution router: classify the sample's resolution bucket.
            r_logits = self.head.router_logits(pooled).float()
            r_target = _resolution_bucket(resolution)
            router_loss = F.cross_entropy(r_logits, r_target)

            # 3) Adaptive token manager (non-destructive): encourage lower visual-token
            #    importance mass when hallucination risk is low (i.e. more compressible).
            importance = self.head.token_importance(hidden).float()  # [B, T]
            tmask = attention_mask.float()
            mean_importance = (importance * tmask).sum(1) / tmask.sum(1).clamp(min=1.0)
            token_loss = (mean_importance * (1.0 - halluc_risk.float())).mean()

            total = (
                lm_loss
                + self.w_halluc * halluc_loss
                + self.w_router * router_loss
                + self.w_token * token_loss
            )
            aux_log = {
                "lm_loss": lm_loss.detach(),
                "halluc_loss": halluc_loss.detach(),
                "router_loss": router_loss.detach(),
                "token_loss": token_loss.detach(),
            }

        return CausalLMOutputWithPast(
            loss=total,
            logits=outputs.logits,
        )


class HaramTrainer(Trainer):
    """Minimal Trainer for the HARAM wrapper: combined-loss compute + LoRA/head save."""

    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)

        # Save the base model / LoRA adapter.
        base = unwrapped.base
        if hasattr(base, "save_pretrained"):
            base.save_pretrained(output_dir)
        else:  # pragma: no cover - non-PEFT fallback
            torch.save(base.state_dict(), os.path.join(output_dir, "base_state_dict.bin"))

        # Save the HARAM head, excluding the frozen/unused token-manager compression
        # stack (reserved for a future destructive-compression version) to keep
        # checkpoints small. Reload with strict=False.
        head_sd = {
            k: v for k, v in unwrapped.head.state_dict().items()
            if "token_manager.compression_layers" not in k
            and "token_manager.token_merger" not in k
        }
        torch.save(head_sd, os.path.join(output_dir, "haram_head.bin"))

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
