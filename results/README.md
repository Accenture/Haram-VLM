# Results

Evaluation JSONs and the captured stdout of every run behind the paper's tables. The
`.log` files are deliberately committed (`.gitignore` un-ignores `results/**/*.log`)
because they are the primary record of what was actually run.

Each JSON carries its own provenance in its top-level scalar fields — `pope` (the probe
file), and `ckpt` or `model` (what was evaluated). Those `ckpt` strings record the
*original* run paths and are left untouched, so they still name the pre-release directory
layout.

## POPE evaluations — `protocol/eval_pope.py`

Each directory holds `pope_{random,popular,adversarial}.json`, with `overall`, `seen`,
`unseen`, and per-item `predictions`. The `seen`/`unseen` tags are what make the
contamination cost measurable rather than assumed.

| Directory | Checkpoint | Mean F1 | Mean H | Role |
|---|---|---|---|---|
| `pope_eval_clean/` | `haram_full_4gpu_20260610_072155` | **0.932** | 6.9% | **Main clean model — Table 4** |
| `pope_eval_oldbaseline/` | `haram_full_4gpu_20260610_003136` | 0.915 | 7.0% | Contaminated-training baseline, on the *clean* held-out test — Table 5 |
| `pope_eval/` | `haram_full_4gpu_20260610_003136` | 0.916 | 6.9% | The same contaminated model on the *upstream* POPE files (`coco_pope_*.json`). Kept to show what the leaky setup reports; not cited. |
| `pope_eval_xl/` | `haram_full_4gpu_20260611_044833` | 0.938 | 6.5% | Data-scale study, 6k → 36k training images (Sec. 6.6) |
| `pope_eval_plainlora/` | `haram_plainlora_6k_4gpu_20260615_030648` | 0.919 | 6.6% | Ablation: plain LoRA, no HARAM heads (Sec. 6.7) |
| `pope_eval_visionlora/` | `haram_visionlora_4gpu_20260610_204531` | 0.929 | 7.2% | Ablation: adds vision-tower LoRA. **Negative** (Sec. 6.7) |

H = hallucination rate = FP / (TN + FP), i.e. the rate of asserting an absent object.

These reproduce the paper's ablation claims directly: the heads add +0.013 mean F1
(0.919 → 0.932), vision-LoRA costs 0.003 (0.932 → 0.929) and raises H (6.9% → 7.2%), and
scaling the disjoint training set sixfold adds +0.006 (0.932 → 0.938).

## Adaptive-resolution sweeps — `controller/adaptive_infer_*.py`

Each directory holds `adaptive_{random,popular,adversarial}.json` with `pareto`,
`always_low`, `always_high`, and per-item `records`. Because both the scout and the
high-resolution pass are recorded for every probe, the entire escalation sweep is
computed offline from these files — no GPU needed to reproduce Table 1 or Fig. 4.

| Directory | Model | Budget | Fig. 4 panel |
|---|---|---|---|
| `adaptive_pareto/` | Phi-3-Vision, fine-tuned (`haram_full_4gpu_20260610_072155`) | 4 → 16 crops | (a) |
| `adaptive_pareto_base/` | Phi-3-Vision, raw (`ckpt: base`) | 4 → 16 crops | (b) |
| `adaptive_pareto_qwen3/` | `Qwen/Qwen3-VL-8B-Instruct` | ~128 → 1024 image tokens | (c) |
| `adaptive_pareto_internvl/` | `OpenGVLab/InternVL3-8B-hf` | 1 → 12 tiles | (d) |

`adaptive_pareto_qwen3/` additionally supplies the qualitative examples in Fig. 6.

## Learned risk head — `risk_head/`

| Directory | Contents |
|---|---|
| `risk_v2/` | PCA + calibrated MLP on the last-token hidden state — **Table 2** |
| `risk_v3/` | The richer-feature ablation, per feature set — **Table 7** |

Both are keyed by `splits`. `risk_v3/` also carries a `_2k` variant at the reduced
training size used to show the D >> N overfitting.

The `.npz` scout-feature dumps these are computed from (~700 MB) are **not** committed —
regenerate with `risk_head/dump_features_qwen3.py`.

## Not included

The V\*Bench runs behind Table 3 are not committed as JSON; regenerate with
`controller/adaptive_infer_vstar.py` and `risk_head/train_eval_vstar.py`.
