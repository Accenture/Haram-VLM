# The learned risk signal

Paper: **Sec. 6.4** (learned risk head on hard negatives), **App. D** (richer-feature
ablation). Produces Tables 2 and 7.

Confidence is weakest exactly where it matters most: on the adversarial split the scout
asserts a co-occurring but absent object *with high confidence*, so low-confidence gating
never flags it. The fix tested here is a small head trained to predict **whether the
scout is wrong**, used to rank escalation instead of raw confidence.

The head trains on probes over *training* images (disjoint from the test set),
held-out-validated and seed-averaged. It sees only what is available at inference before
deciding to escalate: the scout's forward pass.

## Two stages

**1. Dump scout features** (GPU, `qwen3` env). One low-res scout pass and one high-res
pass per probe:

| Script | Records |
|---|---|
| `dump_features_qwen3.py` | last-token last-layer hidden state, scout confidence, both predictions, both token costs |
| `dump_features_qwen3_rich.py` | the above **plus** multi-layer hidden states, logit-distribution scalars, and visual-attention features (App. D) |
| `dump_features_vstar.py` | the same for V\*Bench (InternVL3-8B, 1 → 24 tiles) |

`dump_features_qwen3_rich.py` writes to a separate directory and re-dumps over the same
probes, so it never clobbers the plain dump.

**2. Train and evaluate the head** (CPU-light):

| Script | Head | Reports |
|---|---|---|
| `train_eval_linear.py` | logistic probe | first-pass baseline: learned vs confidence vs oracle |
| `train_eval_mlp.py` | PCA + calibrated MLP | **Table 2** — the head used in the paper |
| `train_eval_feature_ablation.py` | same recipe, increasing feature sets | **Table 7** — the App. D ablation |
| `train_eval_vstar.py` | PCA + logistic, k-fold CV | the "Adaptive (learned)" row of Table 3 |

V\*Bench has only ~170 items, so its probe must be scored out-of-fold — hence the
separate cross-validated script.

## Results, including the negatives

The learned head sharpens error-ranking on exactly the splits where confidence collapses
(adversarial AUROC 0.82 → 0.89, popular 0.85 → 0.89), which converts into token savings
of adversarial 50 → 68% and popular 71 → 79%. On the easy random split confidence is
already near the oracle and the head does not help; rank fusion recovers best-of-both.

Two negative results, both reproducible here:

- **Richer scout features do not close the oracle gap** (Table 7). Logit-distribution
  scalars add only +0.8–0.9pt AUROC on the hard splits. Multi-layer states and the union
  of all features *hurt* — with ~2.4k training probes against a 4096-d state, they overfit
  (D >> N). Visual-attention features carry real signal alone (adversarial AUROC 0.881 vs
  confidence 0.815) but do not beat the last-token embedding, because the hidden state
  already encodes what the scout attended to. The paper therefore keeps the head simple.
- **On V\*Bench the learned head does not help at all** (AUROC 0.69, matching confidence;
  −1% saving). When the scout never resolves the target, its hidden state carries no
  signal that it is wrong. No internal risk estimate can flag what the model never
  perceived.

The residual oracle gap is thus not a feature-extraction problem. It would need a
different lever — e.g. escalation-aware training — rather than richer scout features.

```bash
python risk_head/dump_features_qwen3.py --help
python risk_head/train_eval_mlp.py --help
```
