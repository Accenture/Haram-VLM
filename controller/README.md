# The training-free controller

Paper: **Sec. 4** (method), **Sec. 6.2–6.3** (results, calibration), **Sec. 6.5**
(V\*Bench boundary case). Produces Tables 1–3 and Fig. 4.

A cheap low-resolution *scout* pass answers the query and reports its confidence; only
queries whose risk exceeds a threshold pay for a high-resolution pass. Expected cost is
`T_low + P(risk > tau) * T_high`. Nothing is trained and no parameters are added — the
controller is a single threshold on the frozen model's own confidence.

Each script runs **both** passes for every probe and records each pass's prediction, the
scout's confidence, and each pass's visual-token cost. The escalation threshold is
therefore swept *offline* from one dump, so the whole accuracy-vs-tokens Pareto comes
from a single GPU run.

| Script | Architecture | How resolution is controlled | Env |
|---|---|---|---|
| `adaptive_infer_phi3.py` | Phi-3-Vision (fine-tuned or raw) | image crops, 4 → 16 | `haram` |
| `adaptive_infer_qwen3.py` | Qwen3-VL-8B | native pixel budget, ~128 → 1024 tokens | `qwen3` |
| `adaptive_infer_internvl.py` | InternVL3-8B | dynamic tiling, 1 → 12 tiles | `qwen3` |
| `adaptive_infer_vstar.py` | InternVL3-8B on V\*Bench | dynamic tiling, 1 → 24 tiles | `qwen3` |

`rank_based_compare.py` is the table generator — **CPU-only**, no VLM loaded. It reads the
dumps above and reports, per
split, the token saving needed to reach within 0.5pt of always-high accuracy under four
escalation rankings: **confidence** (training-free), **learned** (the head in
[`../risk_head/`](../risk_head/)), **combined** (rank fusion), and **oracle** (escalate
exactly the scout-wrong queries) as the ceiling.

## Why rank-based

Escalating "the top k% most uncertain queries, sweeping k" is robust to a model's
absolute confidence scale — some models saturate their confidences, which makes a fixed
threshold `tau` incomparable across architectures. Rank-based escalation is the
consistent metric throughout the paper.

## Scope

- On V\*Bench the controller **saves almost nothing** (6%, escalating 89%). The scout
  cannot resolve a tiny target, so it is confidently wrong and low confidence never fires.
  An oracle saves 60%: the headroom is real, but the lever is perception, not risk
  estimation. This is reported as the method's boundary, not as a result.
- Escalated queries pay for both passes. The cost accounting in these scripts includes
  both; a shared low-level encoder could amortise the scout, which is future work.

## Reproducing Table 1 without a GPU

Because both passes are already recorded in [`../results/`](../results/), the confidence
and oracle columns for all four architectures reproduce with no GPU and no downloads:

```bash
python controller/rank_based_compare.py
```

That prints all twelve rows of Table 1. The *learned* and *combined* columns additionally
need the Qwen3 scout-feature dumps from [`../risk_head/`](../risk_head/); they are skipped
with a note if absent.

To re-run the underlying inference (GPU required):

```bash
python controller/adaptive_infer_qwen3.py --help
```
