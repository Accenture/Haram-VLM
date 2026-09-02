# Figures

Figure generation for the paper. **No GPU required** for the quantitative figures — they
are built from the saved evaluation JSONs in [`../results/`](../results/).

```bash
bash build.sh          # regenerates everything, skipping what needs image data
```

Or individually:

```bash
python generate_figures.py     # Fig. 1 motivation, Fig. 3 contamination, results
python generate_pareto.py      # Fig. 4 accuracy-vs-tokens Pareto (4 panels)
python generate_vstar_qual.py  # Fig. 7 V*Bench qualitative examples
python generate_pope_qual.py   # Fig. 6 POPE qualitative examples
```

By default these read `../results/`; override with `HARAM_RESULTS`. The two qualitative
scripts additionally need the COCO images and V\*Bench — see [../DATA.md](../DATA.md) —
and `build.sh` skips them with a warning if the data is absent.

| File | Content | Source data |
|---|---|---|
| `fig_motivation.pdf` | Fig. 1 — resolution vs hallucination | hardcoded; see the warning below |
| `fig_results.pdf` | held-out POPE accuracy / F1 / hallucination, clean model | `results/pope_eval_clean/` |
| `fig_contamination.pdf` | Fig. 3 — (a) 93% vs 0% train/test overlap; (b) controlled before/after | `results/pope_eval_{clean,oldbaseline}/` |
| `fig_pareto.pdf` | Fig. 4 — accuracy-vs-tokens across 4 model configurations | `results/adaptive_pareto*/` |
| `fig_pope_qual.pdf` | Fig. 6 — the three controller decision regimes | `results/adaptive_pareto_qwen3/` + COCO images |
| `fig_vstar_qual.pdf` | Fig. 7 — V\*Bench confident mistakes | V\*Bench |

## Warning: Fig. 1 is illustrative, not reproducible from this repo

The five points in `fig_motivation` are **hardcoded literals** in
[`generate_figures.py`](generate_figures.py), not recomputed from anything in
`../results/`. They come from the project's preliminary resolution sweep on Phi-3-Vision
(the harness now in [`../exploratory/multi_model_eval/`](../exploratory/multi_model_eval/)),
whose raw run is not part of this release.

Of the five points, only **224 / 448 / 672 px (26.7 / 13.3 / 3.3%) were measured**;
**336 px (20.0%) is interpolated and 896 px (2.0%) extrapolated**. The annotated
Pearson *r* = −0.997 is therefore a fit to a partly interpolated curve, not to five
independent measurements — and the nearest fully-measured Phi-3-Vision sweep in the
preliminary harness gave *r* = −0.787.

The paper presents this as a motivating observation only (Sec. 3: *"the point for us is
not the headline number but its shape"*), and **no claim in the paper depends on it**.
Every quantitative result is in Tables 1–7 and reproduces from `../results/`. If you need
the resolution/hallucination curve itself, re-measure it rather than citing this figure.
