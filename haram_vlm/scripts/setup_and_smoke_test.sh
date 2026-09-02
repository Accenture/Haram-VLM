#!/bin/bash
# =============================================================================
# HARAM-VLM: one-shot environment setup + HARAM-integrated smoke test
#
# Does, end to end:
#   Step 1. Create a Python 3.10 conda env and install all training deps.
#   Step 2. Download only the COCO val2014 images the smoke subset references.
#   Step 3. Build a tiny smoke dataset with image paths rewritten to basenames.
#   Step 4. Run a 5-step LoRA fine-tune with --haram_enable (all 3 HARAM modules
#           wired in via auxiliary losses) and confirm a checkpoint is written.
#
# Re-run friendly: set SKIP_DEPS=1 to skip env creation/installs.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-haram}"
COCO_DIR="$PROJECT_ROOT/data/coco_smoke"
SMOKE_JSON="$PROJECT_ROOT/data/haram_smoke.json"
# Training-format JSON produced by the clean protocol:
#   python protocol/generate_coco_pope.py --help
SOURCE_JSON="${SOURCE_JSON:-$(dirname "$PROJECT_ROOT")/coco_build/data/haram_train_cocoLarge.json}"
N_SAMPLES="${N_SAMPLES:-8}"
MODEL_ID="${MODEL_ID:-microsoft/Phi-3-vision-128k-instruct}"

echo "============================================================"
echo " HARAM-VLM setup + smoke test"
echo " Project: $PROJECT_ROOT"
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: Python env + dependencies
# -----------------------------------------------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ "${SKIP_DEPS:-0}" != "1" ]; then
    echo ""
    echo ">>> Step 1: creating env '$ENV_NAME' (python 3.10) and installing deps..."
    if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
        conda create -y -n "$ENV_NAME" python=3.10
    fi
    conda activate "$ENV_NAME"

    # CUDA 12.1 wheels (H100-compatible). Pinned transformers for the bundled
    # custom Phi-3-Vision modeling files.
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install \
        "transformers==4.43.0" \
        "accelerate>=0.30,<0.34" \
        "peft>=0.11,<0.13" \
        "deepspeed>=0.14,<0.16" \
        pillow numpy ujson sentencepiece einops protobuf tensorboard
else
    echo ""
    echo ">>> Step 1: SKIP_DEPS=1 -> activating existing env '$ENV_NAME'..."
    conda activate "$ENV_NAME"
fi

python - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__} | cuda avail={torch.cuda.is_available()} | "
      f"transformers {transformers.__version__}")
PY

# -----------------------------------------------------------------------------
# Steps 2 + 3: build smoke subset (basename image paths) and fetch its images
# -----------------------------------------------------------------------------
echo ""
echo ">>> Steps 2-3: building smoke subset ($N_SAMPLES samples) and downloading COCO images..."
mkdir -p "$COCO_DIR"

if [ ! -f "$SOURCE_JSON" ]; then
    echo "ERROR: no training JSON at $SOURCE_JSON" >&2
    echo "Generate it first (see DATA.md):" >&2
    echo "  python protocol/generate_coco_pope.py --help" >&2
    echo "Or point SOURCE_JSON at an existing training-format JSON." >&2
    exit 1
fi

python - "$SOURCE_JSON" "$SMOKE_JSON" "$COCO_DIR" "$N_SAMPLES" <<'PY'
import json, os, sys, urllib.request

src, dst, coco_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
data = json.load(open(src))[:n]

COCO_URL = "http://images.cocodataset.org/val2014/{}"
out = []
for rec in data:
    base = os.path.basename(rec["image"])
    local = os.path.join(coco_dir, base)
    if not os.path.exists(local):
        url = COCO_URL.format(base)
        print(f"  downloading {base} ...")
        urllib.request.urlretrieve(url, local)
    else:
        print(f"  cached     {base}")
    rec = dict(rec)
    rec["image"] = base  # rewrite absolute Mac path -> basename (uses --image_folder)
    out.append(rec)

json.dump(out, open(dst, "w"), indent=1)
print(f"  wrote {len(out)} samples -> {dst}")
PY

# -----------------------------------------------------------------------------
# Step 4: HARAM-integrated smoke training
# -----------------------------------------------------------------------------
echo ""
echo ">>> Step 4: running HARAM smoke training (5 steps, LoRA, frozen LLM+vision)..."
echo "    NOTE: first run downloads the Phi-3-Vision weights (~8GB) from HuggingFace."
echo ""

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$PROJECT_ROOT/output/haram_smoke_$TS"

python "$PROJECT_ROOT/src/training/train.py" \
    --model_id "$MODEL_ID" \
    --data_path "$SMOKE_JSON" \
    --image_folder "$COCO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --haram_enable True \
    --haram_halluc_weight 0.3 \
    --haram_router_weight 0.3 \
    --haram_token_weight 0.05 \
    --lora_enable True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_img_projector True \
    --bf16 True \
    --disable_flash_attn2 True \
    --num_crops 4 \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --gradient_checkpointing False \
    --remove_unused_columns False \
    --report_to none \
    --dataloader_num_workers 0

echo ""
echo "============================================================"
if [ -f "$OUTPUT_DIR/haram_head.bin" ]; then
    echo " SMOKE TEST PASSED"
    echo "   LoRA adapter + HARAM head saved to:"
    echo "   $OUTPUT_DIR"
    ls -1 "$OUTPUT_DIR"
else
    echo " SMOKE TEST FAILED: no haram_head.bin in $OUTPUT_DIR"
    exit 1
fi
echo "============================================================"
