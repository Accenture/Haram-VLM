# The clean COCO→POPE protocol

Paper: **Sec. 5** (the contamination diagnosis and the fix), **App. B** (generation
details, scales, contamination check). Produces Tables 4 and 5 and Fig. 3.

POPE draws its random / popular / adversarial probes from a single 500-image pool.
Because those probes double as a convenient fine-tuning signal, it is common practice to
train on POPE-derived data and then evaluate on POPE. **That leaks ~93% of exact
(image, question) evaluation pairs into training**, so the reported numbers largely
measure memorisation. This directory regenerates the benchmark so that train and test
images are disjoint, which reduces the overlap to 0%.

## Generation

`generate_coco_pope.py` builds the benchmark from COCO val2014 instance annotations:

1. Keep images with ≥ 3 object categories (~19k eligible in val2014).
2. Sample a **disjoint** train/test image split.
3. Per image, emit balanced yes/no probes — *k* positives (present categories) and *k*
   negatives (*k* = 3), with negatives drawn under the three POPE regimes:
   - **random** — a uniformly sampled absent category
   - **popular** — the globally most frequent absent category
   - **adversarial** — the absent category with the highest co-occurrence with the
     image's present objects, from a corpus-wide co-occurrence matrix
4. Training mixes all three negative types per image, so the model sees hard negatives
   during training. The test set is emitted as three separate POPE-style files.

Because the image split is disjoint, every test probe is genuinely held out. The
generator is deterministic (fixed seed) and downloads only the images it selects.

`generate_coco_pope_scale.py` scales the training set by additionally drawing from
train2014, while **holding the existing 1k-image test set fixed** so earlier results stay
comparable. It loads the test image list and excludes those images from the new training
pool, so the larger training set is still disjoint from the held-out test.

Each training probe also carries metadata — a resolution bucket sampled from
{224, 336, 448, 672, 896}, the corresponding empirical risk target, and the query type.
This is consumed *only* by the auxiliary heads in [`../haram_vlm/`](../haram_vlm/) and is
irrelevant to the training-free controller.

## Scales

| Set | Images | Balanced probes | Used for |
|---|---|---|---|
| Main train | 6k | 36k | Tables 4, 5 |
| Scaled train | 36k | 216k | the data-scale study (Sec. 6.6) |
| Test (fixed) | 1k | 18k (3 × 6k) | everything |

The controller experiments evaluate on 2,000 balanced probes per split.

## Evaluation

`eval_pope.py` loads a base Phi-3-Vision plus a trained LoRA adapter, runs greedy yes/no
generation on a POPE split using the **same prompt format as training**, and reports POPE
metrics. It additionally tags each question as `seen` or `unseen` — whether its
(image, question) pair appears in the training JSON — and reports metrics overall *and*
split by that tag. That tagging is what makes the contamination cost in Table 5
measurable rather than assumed.

```bash
# CPU only -- needs COCO annotations, downloads just the images it selects
python protocol/generate_coco_pope.py --help

# GPU + the 'haram' env (loads Phi-3-Vision plus a LoRA adapter)
python protocol/eval_pope.py --help
```

> The upstream POPE files are **not** used anywhere in the paper, and are not
> redistributed here — the exploratory harness downloads them on first run. Upstream
> POPE (MIT) is at https://github.com/RUCAIBox/POPE.
