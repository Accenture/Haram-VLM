#!/bin/bash
# =============================================================================
# HARAM-VLM full training on the first 4 GPUs — WITH vision-tower LoRA.
#
# Self-contained (does NOT call run_full_training_4gpu.sh). Same clean COCO data
# and same recipe as the LLM-only run, but adds LoRA adapters to the frozen vision
# tower (--vision_lora True) so the visual encoder adapts to the task.
#
# Memory note: making the vision encoder trainable means all num_crops crops'
# activations through the 24-layer ViT must be kept for backprop (frozen vision
# discarded them). That OOMs at per-device batch 4, so this script defaults to
# batch 2 / grad-accum 16 (same effective batch = 4 gpu * 2 * 16 = 128).
# If it still OOMs: PER_DEVICE_BS=1 GRAD_ACCUM=32 (or drop NUM_CROPS to 8).
#
# Launched via torchrun (DDP) to avoid the nn.DataParallel StopIteration bug in
# the HARAM head. flash-attn must be installed (it is): we do NOT pass
# --disable_flash_attn2, so train.py defaults to flash_attention_2.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- config (env-overridable) ------------------------------------------------
ENV_NAME="${ENV_NAME:-haram}"
MODEL_ID="${MODEL_ID:-microsoft/Phi-3-vision-128k-instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29534}"

# Clean, disjoint-split COCO dataset (override via env if needed).
TRAIN_DATA="${TRAIN_DATA:-${HARAM_WORK:-$PWD/work}/coco_build/data/haram_train_cocoLarge.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${HARAM_WORK:-$PWD/work}/coco_build/images}"

NUM_CROPS="${NUM_CROPS:-16}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"     # halved vs LLM-only run (vision activations)
GRAD_ACCUM="${GRAD_ACCUM:-16}"          # effective batch = 4 gpu * 2 * 16 = 128
EPOCHS="${EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-1e-4}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/haram_visionlora_4gpu_$TS}"

# --- activate env ------------------------------------------------------------
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo " HARAM-VLM full training (4 GPU / DDP) + VISION LoRA"
echo "   GPUs        : $GPU_IDS"
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
    --vision_lora True \
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
    --run_name "haram_visionlora_4gpu_$TS"

echo ""
echo "============================================================"
if [ -f "$OUTPUT_DIR/haram_head.bin" ]; then
    echo " TRAINING COMPLETE -> $OUTPUT_DIR"
    ls -1 "$OUTPUT_DIR"
else
    echo " WARNING: no haram_head.bin in $OUTPUT_DIR (check log)"
fi
echo "============================================================"
