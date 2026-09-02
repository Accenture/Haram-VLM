# HARAM-VLM: the fine-tuned variant

Paper: **Sec. 4.3** (the learned risk alternative), **App. A** (architecture and
multi-task objective), **Sec. 6.7** (ablations). Produces Tables 4–6.

This is the *lightly fine-tuned* variant used to study how training affects the
controller. **The training-free controller in [`../controller/`](../controller/) does not
require any of this.** The variant exists to answer one question: does better calibration
buy more savings? It does — light task fine-tuning raises Phi-3-Vision's mean token
saving from ~31% to ~54% (~1.7×).

## What is trained

A frozen Phi-3-Vision base (CLIP ViT-L/14 vision encoder, ~4B-parameter LLM) is adapted
with rank-64 LoRA on the language model. The vision tower and the LLM weights stay
frozen; the image projector is tuned. In a single supervised forward pass over an
(image, question, answer) triple, the final multimodal hidden state is mean-pooled and
drives three lightweight auxiliary heads, each supervised by per-sample metadata already
present in the training data:

| Head | Module | Loss | Supervision |
|---|---|---|---|
| Resolution router | `src/models/haram_modules/resolution_router.py` | `L_route` (cross-entropy over 5 buckets) | metadata resolution bucket |
| Hallucination predictor | `src/models/haram_modules/hallucination_predictor.py` | `L_risk` (MSE on blended risk) | expected-risk target |
| Adaptive token manager | `src/models/haram_modules/adaptive_token_manager.py` | `L_token` (importance regulariser) | risk target |

The objective is `L = L_LM + 0.3 L_risk + 0.3 L_route + 0.05 L_token`. Only the LoRA
adapters and the (small) heads receive gradients; the heads are kept in fp32 for stable
auxiliary losses while the base runs in bf16.

`src/training/haram_model.py` is the batched, differentiable training-time wiring of the
three modules — the standalone modules were written for single-sample, string-input
inference, so the training path reconstructs their feature computation in a batched form.
`src/models/haram_vlm.py` is that standalone inference prototype; `demo_modules.py`
exercises it. Neither is used to produce any number in the paper.

## Important scope limit

**Resolution routing in this variant is training-time only.** The router and token
manager act as auxiliary training signals; the released checkpoint runs at a **fixed
16-crop resolution** at inference. The token manager is deliberately non-destructive — no
tokens are dropped within the supervised pass, since that would break Phi-3-Vision's
image-placeholder alignment. Inference-time adaptive resolution is the training-free
controller, which is a separate wrapper.

## Ablations, both modest or negative

- **Auxiliary heads:** plain LoRA with no HARAM heads reaches mean held-out F1 0.919 vs
  0.932 with the heads — a consistent but small +0.012 to +0.014 per split. The bulk of
  performance comes from the LoRA fine-tuning itself. The heads are a helpful
  training-time ingredient, not the load-bearing component; the controller is.
- **Vision tower:** extending LoRA into the vision tower does not help (mean F1
  0.932 → 0.929, mean hallucination 6.9% → 7.2%, every split slightly worse). The default
  keeps it frozen.

## Layout

```
src/models/haram_modules/   the three auxiliary modules
src/models/haram_vlm.py     standalone inference prototype (not used for paper results)
src/training/               training scaffold + batched HARAM wiring (see ../NOTICE)
src/model/Phi3{,_5}_vision/ vendored Phi-3-V modeling code (see ../NOTICE)
src/serve/                  CLI + demo server inherited from the upstream scaffold
scripts/                    training launchers, DeepSpeed configs, LoRA merge, smoke test
demo_modules.py             exercises the standalone modules
```

## Training

Multi-GPU training uses `torchrun` (DistributedDataParallel) with FlashAttention. The
main recipe is 4 GPUs, per-device batch 4, gradient accumulation 8 (effective 128),
3 epochs, AdamW at lr 1e-4 (projector 1e-5), cosine schedule with 0.03 warmup.

```bash
# end-to-end sanity check: env, tiny COCO subset, 5-step LoRA run with the heads wired in
bash haram_vlm/scripts/setup_and_smoke_test.sh

# the main recipe (expects the clean training JSON; see ../protocol/ and ../DATA.md)
bash haram_vlm/scripts/run_full_training_4gpu.sh
```

The smoke test needs a training-format JSON from
[`../protocol/generate_coco_pope.py`](../protocol/generate_coco_pope.py); point it
elsewhere with `SOURCE_JSON=...`.
