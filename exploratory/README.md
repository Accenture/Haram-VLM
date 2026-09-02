# Exploratory work — not in the paper

**Nothing in this directory backs any claim in the paper.** It is kept because it records
useful negative results and the research line that led to, and follows on from, the
published controller. Treat every number here as provisional: these runs were not held to
the paper's evaluation standard, several are single-seed, and some use small or
non-disjoint samples.

If you only want to reproduce the paper, ignore this directory entirely.

## `foveation/` — looking closer *where*, not just *when*

The paper's boundary case (Sec. 6.5) is V\*Bench: when the scout cannot resolve a tiny
target, no risk signal can flag the error, and the lever has to be perception. Sec. 6.8(3)
names the fix as "spatially concentrated" attention and leaves it to future work. This is
that work, in progress.

| Script | Question | Status |
|---|---|---|
| `phase0_foveation.py` | Does concentrating high-res tokens on the query-relevant region beat uniform allocation, using **ground-truth** boxes? | **Feasible.** Oracle foveation 0.971 vs uniform-high 0.694 on 170 V\*Bench items, at 0.57× the tokens. An *oracle* upper bound only. |
| `phase1a_selfground.py` | Can the VLM find the region itself, training-free, in one shot? | **Negative.** 43% center-hit; self-grounding fails. |
| `phase1b_coarse2fine.py` | Does iterative coarse-to-fine zoom localise better without training? | Partial: 0.68 center-hit, 0.80 accuracy — but at 3 encodes. |
| `generate_localization_data.py` | COCO → fine-grained (image, query, target-box) triples | data generator |
| `dump_survey_features.py` | frozen survey-grid features for the localizer | feature dump |
| `localizer.py`, `train_localizer.py` | a learned query-conditioned localizer head | in progress |
| `phase0b_halluc_foveation.py` | Does foveation cut *absent-object* hallucination on POPE-adversarial negatives? | oracle probe |
| `phase0c_selfconsistency.py` | Does resolution self-consistency detect hallucination, tested fairly on both labels? | probe |

The Phase 0 oracle result is a feasibility go/no-go, **not** a learned method. Do not cite
it as a result.

## `decoding/` — perception vs decoding-side mitigation

A separate line asking whether hallucination has two distinguishable failure modes —
*prior-driven* (the language prior overrides the image) and *perception-bound* (the
evidence was never resolved) — and whether decoding-side methods and resolution fix
different ones.

- `phase_perception_vs_vcd.py`, `phase_decoding_compare.py` — resolution vs VCD and DoLa
  on clean POPE-adversarial
- `arch_2x2.py`, `arch_2x2_phi3.py` — the same 2×2 across Qwen3-VL, InternVL, Phi-3-V
- `move2_decode_and_look.py` — a combined policy: DoLa always, escalate resolution only
  when a cheap signal fires
- `move3_layer_trajectory.py` — the layer-wise mechanism behind the 2×2
- `vstar_dola.py` — does DoLa help on perception-bound V\*Bench? (it should not)
- `opera.py` — OPERA over-trust penalty baseline
- `chair.py`, `captioning_chair.py` — CHAIR scorer and captioning conditions. Absolute
  CHAIR values depend on the synonym list and parser; only the *relative* comparison is
  meaningful, and both are held fixed across methods.
- `contamination_twist.py` — does benchmark contamination inflate decoding-side gains?
  Uses the deliberately contaminated checkpoint, split seen/unseen.
- `render_*.py`, `dump_examples.py` — figure rendering for the above (CPU only)

## `counting/` — beyond yes/no

The same predict-then-allocate policy on an integer-count task (InternVL), reading the
scout's first-token distribution over digit tokens. The paper's "beyond yes/no" evidence
is V\*Bench (Sec. 6.5); counting was not carried into it.

## `preliminary/` — early resolution-hypothesis probes

The first resolution sweeps, predating the controller and the clean protocol. `haram`
env, small samples, Qwen2.5-VL / Qwen3-VL setup checks.

## `multi_model_eval/` — standalone multi-model harness

A self-contained harness for evaluating several VLMs (Phi-3-Vision, LLaVA, Qwen) at a
range of resolutions on POPE. This is the lineage of the Fig. 1 motivation curve; see
[`../figures/README.md`](../figures/README.md) for why that curve is illustrative only.
Its own reported correlations (Phi-3-Vision *r* = −0.787, LLaVA −0.934, Qwen2-VL −0.705)
are from small-sample runs and are **not** the paper's numbers.

> Scripts that generated **simulated** rather than measured results have been removed
> from this release, as have the plotting scripts that mixed simulated and real data.
