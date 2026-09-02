# Data and weights

Nothing in this document is redistributed by this repository. Datasets keep their own
licenses (see [NOTICE](NOTICE)); trained adapters are published separately on the
Hugging Face Hub at
[`samaonline/haram-vlm-phi3v-lora`](https://huggingface.co/samaonline/haram-vlm-phi3v-lora).

## Quick start

```bash
export HARAM_ROOT=$(pwd)          # repo root; coco_build/ is created below
export HARAM_WORK=$PWD/work       # scratch for eval outputs
export HF_HOME=~/.cache/huggingface
```

## 1. COCO 2014

The POPE probes and the exploratory localization triples are both built from COCO 2014 —
`val2014` for evaluation, `train2014` only for the scaled training set and the
exploratory localizer data.

```bash
mkdir -p "$HARAM_ROOT/coco_build/images" "$HARAM_ROOT/coco_build/annotations"
cd "$HARAM_ROOT/coco_build"

# Images (~13 GB val + ~19 GB train; val alone is enough to reproduce the POPE tables)
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/train2014.zip     # only for the scaled/localizer sets
unzip -q -j val2014.zip -d images/
unzip -q -j train2014.zip -d images/

# Instance annotations (~250 MB zip)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -q annotations_trainval2014.zip
# -> annotations/instances_{train,val}2014.json
```

The scripts expect a **flat** `coco_build/images/` holding `COCO_val2014_*.jpg` and
`COCO_train2014_*.jpg` side by side, which is why `unzip -j` is used above.

## 2. POPE probes — regenerate, do not download

Evaluation uses **regenerated, image-disjoint** POPE-style probes, never the upstream
POPE files. This is the point of Sec. 5: training on POPE-derived data and evaluating on
POPE leaks ~93% of exact (image, question) evaluation pairs into training. Regenerate
from COCO instead:

```bash
cd "$HARAM_ROOT"
python protocol/generate_coco_pope.py \
    --ann       coco_build/annotations/instances_val2014.json \
    --image-dir coco_build/images \
    --out-dir   coco_build/data
# -> coco_build/data/pope_test_{random,popular,adversarial}.json   (POPE-style, JSONL)
#    coco_build/data/haram_train_cocoLarge.json                    (training format,
#                                                                   with head metadata)

# to scale the training set while holding the same test images out:
python protocol/generate_coco_pope_scale.py \
    --ann          coco_build/annotations/instances_{train,val}2014.json \
    --image-dir    coco_build/images \
    --exclude-test coco_build/data/pope_test_random.json \
    --out          coco_build/data/haram_train_36k.json
```

The generator is deterministic (fixed seed) and downloads only the images it selects, so
you do not need the full COCO zips to build a small benchmark.

Upstream POPE (MIT) is at https://github.com/RUCAIBox/POPE. It is **not** redistributed
here; the exploratory harness downloads the split files on first run. No number in the
paper uses them.

## 3. V\*Bench

Used for the high-resolution visual-search evaluation (Sec. 6.5, Table 3) and by the
exploratory foveation work. Pulled from the Hub:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('craigwu/vstar_bench', repo_type='dataset')
"
```

The scripts locate it under `$HF_HOME/hub/datasets--craigwu--vstar_bench/snapshots/`.
V\*Bench provides 115 direct-attribute and 56 relative-position multiple-choice questions
over 2000–3300px images; the evaluation uses 170 of the 171 (one is dropped for a missing
image/options field).

## 4. Base models

Downloaded automatically from the Hub on first run:

| Model | Used by |
|---|---|
| `microsoft/Phi-3-vision-128k-instruct` | `haram_vlm/`, `protocol/eval_pope.py`, `controller/adaptive_infer_phi3.py` |
| `microsoft/Phi-3.5-vision-instruct` | vendored modeling code (alternative base) |
| `Qwen/Qwen3-VL-8B-Instruct` | `controller/adaptive_infer_qwen3.py`, all of `risk_head/` |
| `OpenGVLab/InternVL3-8B-hf` | `controller/adaptive_infer_internvl.py`, `adaptive_infer_vstar.py` |

The Phi-3-V path needs the `haram` env (pinned `transformers`); the Qwen3-VL / InternVL
path needs the `qwen3` env (`transformers >= 4.57`). See the README.

## 5. Trained LoRA adapters

All adapters are rank-64 LoRA (`lora_alpha=64`, `dropout=0.05`) on
`microsoft/Phi-3-vision-128k-instruct`. The default recipe targets **LLM layers only** —
`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` across all 32 blocks — leaving the
vision tower frozen.

| Run | Size | Role | Eval dir |
|---|---|---|---|
| `haram_full_4gpu_20260610_072155` | 385M | **Main clean model.** Image-disjoint training data; the numbers in Table 4. | `results/pope_eval_clean/` |
| `haram_full_4gpu_20260610_003136` | 385M | Contaminated first run, kept as the controlled before/after baseline (Table 5). | `results/pope_eval_oldbaseline/` |
| `haram_visionlora_4gpu_20260610_204531` | 496M | Ablation: adds vision-tower LoRA. Negative result — does not help. | `results/pope_eval_visionlora/` |
| `haram_full_4gpu_20260611_044833` | 385M | Larger-data run, 36k training images (the data-scale study, Sec. 6.6). | `results/pope_eval_xl/` |
| `haram_plainlora_6k_4gpu_20260615_030648` | 385M | Plain LoRA without the HARAM heads (they add ~+0.013 mean F1). | `results/pope_eval_plainlora/` |

Every mapping above is verified from the `ckpt` field recorded inside each result JSON;
see [results/README.md](results/README.md) for the per-directory breakdown.

The two smoke-test runs (`haram_smoke_*`) are 5-step sanity checks and are not published.

### Downloading

Published to **https://huggingface.co/samaonline/haram-vlm-phi3v-lora**.

> **Note:** the Hub repo is currently **private**. Until it is made public, downloading
> requires a token with read access to it (`huggingface-cli login`, or `HF_TOKEN`).

Grab everything:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('samaonline/haram-vlm-phi3v-lora', local_dir='haram_vlm/output')
"
```

Or just the main clean model, which is what Table 4 reports:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('samaonline/haram-vlm-phi3v-lora',
                  allow_patterns='haram_full_4gpu_20260610_072155/*',
                  local_dir='haram_vlm/output')
"
```

Load it as a PEFT adapter on the Phi-3-Vision base:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-vision-128k-instruct",
    trust_remote_code=True, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(
    base, "samaonline/haram-vlm-phi3v-lora",
    subfolder="haram_full_4gpu_20260610_072155")
```

### Publishing (maintainers)

`tools/upload_weights_to_hf.py` pushes the adapters to the Hub. It reads the token from
the `HF_TOKEN` environment variable and never writes it to disk:

```bash
export HF_TOKEN=hf_...     # a write-scoped token
python tools/upload_weights_to_hf.py \
    --source /path/to/haram_vlm/output \
    --repo-id samaonline/haram-vlm-phi3v-lora \
    --private \
    --dry-run          # inspect the file list first, then drop --dry-run
```

Only `adapter_model.safetensors`, `adapter_config.json`, and the tokenizer/processor
configs are uploaded. Optimizer state (`optimizer.pt`, ~1 GB per checkpoint) is
deliberately excluded — it is only useful for resuming training — and smoke-test runs are
skipped entirely. The Hub README is published from
[`tools/hf_model_card.md`](tools/hf_model_card.md); edit that file to update the model
card rather than editing it in the Hub web UI, or the next upload will overwrite it.

## Cached feature dumps

`risk_head/` writes `.npz` scout-feature dumps (~700 MB) which are **not** in this
repository — `.npz` is gitignored. Regenerate with `risk_head/dump_features_qwen3.py`.

## What was intentionally left out of this repo

| Excluded | Size | Why |
|---|---|---|
| `haram_vlm/output/**` | 20 GB | Checkpoints + optimizer state. Adapters published separately, per the table above. |
| `coco_build/` | 6.8 GB | Redistributable only from the COCO source; see §1. |
| `results/risk_features*/` | 700 MB | Cached `.npz` feature dumps. Regenerate with `risk_head/dump_features_qwen3.py`. |
| `benchmarks/counting/` | 123 MB | Generated benchmark images. Regenerate with `exploratory/counting/generate_counting.py`. |
| `exploratory/multi_model_eval/model_cache/` | — | Local Hugging Face cache. |
| Packed conda envs (`haram_env.tar.gz`) | 2.9 GB ×2 | Rebuild from `requirements*.txt`. |
