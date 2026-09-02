# HARAM-VLM

Code, evaluation harness, and results for
**"Look Closer Only When Needed: Training-Free Adaptive-Resolution Perception for
Vision-Language Models."**

Vision-language models process every query at the same resolution and the same
visual-token budget. Most queries are answerable from a coarse view, so that budget is
mostly wasted. This work replaces the fixed choice with a per-query one: a cheap
low-resolution **scout** pass answers the query and estimates its own reliability, and
only uncertain queries are **escalated** to a high-resolution pass. The controller is a
single threshold on the scout's confidence — training-free, no added parameters, wraps
any frozen VLM.

> **Manuscript source is not included.** The paper is under double-blind review; the
> `.tex` source will be added once the review concludes. The code, the evaluation
> harness, and the result files behind every number below are all here.

## Headline results

**Adaptive escalation recovers full-resolution accuracy at a fraction of the tokens**,
across three architectures. Token saving is measured against always-high at iso-accuracy
(within 0.5pt), under rank-based escalation:

| Model | Saving (confidence) | Oracle ceiling |
|---|---|---|
| Phi-3-Vision, fine-tuned | 52–58% | 59–60% |
| Phi-3-Vision, raw | 28–34% | 53–56% |
| Qwen3-VL-8B, raw | 50–80% | 78–86% |
| InternVL3-8B, raw | 76–86% | 83–88% |

The oracle column escalates exactly the scout-wrong queries; the gap to it is the
headroom a better risk signal could capture. Full per-split numbers are Table 1 of the
paper, reproducible from [`results/adaptive_pareto*/`](results/) via
[`controller/rank_based_compare.py`](controller/rank_based_compare.py).

**Calibration governs the savings.** Light task fine-tuning of Phi-3-Vision raises the
mean saving from ~31% to ~54% (~1.7×) — a better-calibrated scout escalates a smaller,
better-chosen set.

**A learned risk head helps where confidence is blind.** On Qwen3-VL, a calibrated MLP on
the scout's last-token hidden state, trained to predict scout errors, lifts savings on the
hard splits: adversarial 50 → 68%, popular 71 → 79%.

**Held-out accuracy of the fine-tuned model** (clean, image-disjoint protocol; reproduces
from [`results/pope_eval_clean/`](results/pope_eval_clean/)):

| Split | Accuracy | F1 | Hallucination |
|---|---|---|---|
| Random | 0.939 | 0.939 | 5.7% |
| Popular | 0.949 | 0.948 | 3.4% |
| Adversarial | 0.908 | 0.910 | 11.5% |

**A clean evaluation protocol.** Fine-tuning on POPE-derived data and evaluating on POPE
leaks ~93% of evaluation (image, question) pairs into training. The image-disjoint
COCO→POPE generator in [`protocol/`](protocol/) reduces that to 0% and scales arbitrarily.

### Scope and negative results

Please read these before building on the results:

- **V\*Bench is the boundary case, not a win.** On high-resolution visual search,
  resolution helps enormously (+22pt, 0.582 → 0.800) but the training-free signal saves
  almost nothing (6%): the scout cannot resolve a tiny target, so it answers confidently
  and wrongly. A learned risk probe does not help either (AUROC 0.69 vs 0.69, −1%
  saving). An oracle saves 60%, so the headroom is real — but the lever there is
  perception, not a better risk detector.
- **Richer scout features do not help.** Multi-layer hidden states, visual-attention
  features, and logit-distribution scalars fail to improve meaningfully over the plain
  last-token embedding; multi-layer and the union of all features overfit. Ablation in
  [`risk_head/train_eval_feature_ablation.py`](risk_head/train_eval_feature_ablation.py).
- **The auxiliary heads are a modest ingredient, not the mechanism.** Plain LoRA without
  the HARAM heads reaches mean held-out F1 0.919 vs 0.932 with them. The controller, not
  the heads, is load-bearing.
- **Vision-tower LoRA is a negative result** (mean F1 0.932 → 0.929). The default keeps
  the vision tower frozen.
- **Resolution routing in the *fine-tuned model* is training-time only.** Its router and
  token manager are auxiliary training signals; that checkpoint runs at a fixed 16-crop
  resolution. The adaptive controller is a separate, training-free inference wrapper —
  see [`controller/`](controller/).
- **The Fig. 1 motivation curve is illustrative.** Its five points are hardcoded and only
  three were measured; see [figures/README.md](figures/README.md).

## Layout

Directories are named after the contribution they implement, and cross-referenced to the
paper:

```
controller/     Training-free predict-then-allocate (Sec. 4, 6.2, 6.3, 6.5)
                  scout -> confidence -> escalate, per architecture; Tables 1-3, Fig. 4
risk_head/      Learned risk signal (Sec. 6.4, App. D)
                  scout-feature dumps + head training; Tables 2, 7
protocol/       Clean image-disjoint COCO->POPE protocol (Sec. 5, App. B)
                  probe generation + POPE evaluation; Tables 4, 5, Fig. 3
haram_vlm/      The fine-tuned HARAM-VLM variant (Sec. 4.3, App. A)
                  LoRA + three auxiliary heads, training scaffold, vendored Phi-3-V
figures/        Figure generation from the saved result JSONs
results/        Evaluation JSONs and captured logs backing every table
exploratory/    NOT in the paper — follow-up and preliminary work; see its README
tools/          Maintainer utilities (Hugging Face weight upload, model card)
```

## Setup

Two environments, because the Qwen3-VL / InternVL path needs a much newer
`transformers` than the vendored Phi-3-V modeling code tolerates:

```bash
# Phi-3-Vision path: haram_vlm/, protocol/eval_pope.py, controller/adaptive_infer_phi3.py
conda create -n haram python=3.10 -y && conda activate haram
pip install -r requirements.txt

# Qwen3-VL / InternVL path: the rest of controller/ and all of risk_head/
conda create -n qwen3 python=3.10 -y && conda activate qwen3
pip install -r requirements_qwen3.txt
```

`flash-attn` is commented out in both files; install it separately if you want it, since
it needs CUDA 11.6+ and a build toolchain:

```bash
pip install flash-attn --no-build-isolation
```

## Data and weights

Datasets and trained adapters are **not** in this repository. See
[DATA.md](DATA.md) for how to obtain COCO, POPE, and V\*Bench, and how to download and
load the LoRA adapters from
[`samaonline/haram-vlm-phi3v-lora`](https://huggingface.co/samaonline/haram-vlm-phi3v-lora).

Scripts resolve paths from environment variables, all with sensible fallbacks:

| Variable | Meaning | Default |
|---|---|---|
| `HARAM_ROOT` | repo root — holds `coco_build/`, `results/` | inferred from script location |
| `HARAM_WORK` | scratch dir for outputs | `$PWD/work` |
| `HARAM_RESULTS` | result JSONs read by `figures/` | `$HARAM_ROOT/results` |
| `HF_HOME` | Hugging Face cache | `~/.cache/huggingface` |
| `PYTHON` | interpreter used by the shell launchers | `python` |

Every path is also overridable per-run via CLI flags — every script under
`controller/`, `risk_head/`, and `protocol/` takes `--help`. The `figures/` scripts
take no arguments and are driven entirely by `HARAM_RESULTS`/`HF_HOME`.

## Reproducing

```bash
# 1. Build the clean, image-disjoint benchmark from COCO annotations
python protocol/generate_coco_pope.py --help

# 2. The training-free controller (records both passes; sweep the threshold offline)
python controller/adaptive_infer_qwen3.py --help      # Qwen3-VL      (Table 1)
python controller/adaptive_infer_internvl.py --help   # InternVL3-8B  (Table 1)
python controller/adaptive_infer_phi3.py --help       # Phi-3-Vision  (Table 1)
python controller/adaptive_infer_vstar.py --help      # V*Bench       (Table 3)

# 3. Rank-based comparison: confidence vs learned vs oracle (Tables 1, 2).
#    The confidence and oracle columns for all four architectures in Table 1
#    reproduce from the committed results/ with NO GPU and no downloads:
python controller/rank_based_compare.py

# 4. The learned risk head (Table 2) and the richer-feature ablation (Table 7)
python risk_head/dump_features_qwen3.py --help
python risk_head/train_eval_mlp.py --help
python risk_head/train_eval_feature_ablation.py --help

# 5. Evaluate a fine-tuned checkpoint on held-out POPE (Tables 4, 5)
python protocol/eval_pope.py --help

# 6. Figures from the saved eval JSONs (no GPU needed)
bash figures/build.sh
```

## Citation

```bibtex
@misc{haram-vlm,
  title  = {Look Closer Only When Needed: Training-Free Adaptive-Resolution
            Perception for Vision-Language Models},
  author = {Amirul Islam and Gyuhak Kim and Jiayun Wang},
  note   = {Center for Advanced AI, Accenture},
  year   = {2026},
  url    = {https://github.com/Accenture/Haram-VLM}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable form. The venue and DOI will be
added once the review concludes.

## License

Apache-2.0 — see [LICENSE](LICENSE). This repository redistributes third-party code
(Phi3-Vision-Finetune, Microsoft/HuggingFace Phi-3-V) under the same license; see
[NOTICE](NOTICE) for full attribution. Benchmark datasets (COCO, POPE, V\*Bench) retain
their own licenses and are **not** redistributed here.
