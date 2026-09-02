#!/bin/bash
# =============================================================================
# HARAM-VLM full training on the first 4 GPUs.
#
# Launches with torchrun => one process per GPU (DistributedDataParallel).
# This deliberately AVOIDS the plain `python train.py` path, which on a
# multi-GPU box makes HuggingFace Trainer wrap the model in nn.DataParallel.
# Under DataParallel the HARAM head's `next(module.parameters())` dtype lookups
# raise StopIteration on the replicas (see haram_model.py:127,142,146) and
# training dies on step 0. DDP gives each process its own un-replicated model,
# so those lookups work.
#
# Data: 7,200-sample full HARAM set, image paths are COCO basenames resolved
# against --image_folder. All referenced images are present locally.
#
# Env: expects the 'haram' conda env (run scripts/setup_and_smoke_test.sh once,
# or restore it via env_pack/restore_env.sh). flash-attn is NOT installed, so we
# force eager attention with --disable_flash_attn2 True.
#
# Override knobs via env vars, e.g.:
#   GPU_IDS=0,1,2,3 PER_DEVICE_BS=4 NUM_CROPS=16 bash run_full_training_4gpu.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- config (env-overridable) ------------------------------------------------
ENV_NAME="${ENV_NAME:-haram}"
MODEL_ID="${MODEL_ID:-microsoft/Phi-3-vision-128k-instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"          # first 4 GPUs
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29533}"

# Default to the SCALED clean disjoint COCO dataset (36k train imgs / 216k questions,
# val2014+train2014). Disjoint from the fixed 1k-image held-out test set.
# Smaller 6k-image set is at haram_train_cocoLarge.json (override TRAIN_DATA to use it).
# Old POPE-derived set (haram_train_full.json) is ~93% contaminated — never use it.
TRAIN_DATA="${TRAIN_DATA:-${HARAM_WORK:-$PWD/work}/coco_build/data/haram_train_cocoXL.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${HARAM_WORK:-$PWD/work}/coco_build/images}"

NUM_CROPS="${NUM_CROPS:-16}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"          # effective batch = 4 gpu * 2 * 8 = 64
EPOCHS="${EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-1e-4}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/haram_full_4gpu_$TS}"

# --- activate env ------------------------------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo " HARAM-VLM full training (4 GPU / DDP)"
echo "   GPUs        : $GPU_IDS"
echo "   model       : $MODEL_ID"
echo "   train data  : $TRAIN_DATA"
echo "   images      : $IMAGE_FOLDER"
echo "   num_crops   : $NUM_CROPS   per-device bs: $PER_DEVICE_BS   grad-accum: $GRAD_ACCUM"
echo "   output      : $OUTPUT_DIR"
echo "============================================================"
python - <<'PY'
import torch
print(f"  torch {torch.__version__} | cuda avail={torch.cuda.is_available()} | "
      f"visible GPUs={torch.cuda.device_count()}")
PY

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$PROJECT_ROOT/src/training/train.py" \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --haram_enable True \
    --haram_halluc_weight 0.3 \
    --haram_router_weight 0.3 \
    --haram_token_weight 0.05 \
    --lora_enable True \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_img_projector True \
    --bf16 True \
    --num_crops "$NUM_CROPS" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --projector_lr 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_strategy steps \
    --gradient_checkpointing False \
    --ddp_find_unused_parameters True \
    --remove_unused_columns False \
    --report_to tensorboard \
    --dataloader_num_workers 4 \
    --run_name "haram_full_4gpu_$TS"

echo ""
echo "============================================================"
if [ -f "$OUTPUT_DIR/haram_head.bin" ]; then
    echo " TRAINING COMPLETE"
    echo "   LoRA adapter + HARAM head saved to: $OUTPUT_DIR"
    ls -1 "$OUTPUT_DIR"
else
    echo " WARNING: no haram_head.bin in $OUTPUT_DIR (check the log above)"
fi
echo "============================================================"
