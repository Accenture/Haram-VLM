#!/bin/bash
# =============================================================================
# Ablation: plain LoRA fine-tuning (NO HARAM heads) on the first 4 GPUs.
# Identical recipe to run_full_training_4gpu.sh but --haram_enable False, i.e.
# no resolution router, no hallucination predictor, no token manager. Trained on
# the SAME 6k clean data as the HARAM model (output/haram_full_4gpu_20260610_072155)
# so the two are directly comparable -> isolates whether the auxiliary heads help.
# torchrun (DDP). Model cache on the share (HF_HOME).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-haram}"
MODEL_ID="${MODEL_ID:-microsoft/Phi-3-vision-128k-instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; NUM_GPUS="${NUM_GPUS:-4}"; MASTER_PORT="${MASTER_PORT:-29535}"
TRAIN_DATA="${TRAIN_DATA:-${HARAM_WORK:-$PWD/work}/coco_build/data/haram_train_cocoLarge.json}"  # 6k, matches HARAM model
IMAGE_FOLDER="${IMAGE_FOLDER:-${HARAM_WORK:-$PWD/work}/coco_build/images}"
NUM_CROPS="${NUM_CROPS:-16}"; PER_DEVICE_BS="${PER_DEVICE_BS:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"; LORA_RANK="${LORA_RANK:-64}"; LORA_ALPHA="${LORA_ALPHA:-64}"; LR="${LR:-1e-4}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/haram_plainlora_6k_4gpu_$TS}"

CONDA_BASE="$(conda info --base)"; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"; export TOKENIZERS_PARALLELISM=false
# NOTE: training base (Phi-3) uses the LOCAL HF cache (already downloaded). The cloudfiles
# HF cache works for single-process inference downloads but its file-locks fail under
# multi-process DDP on CIFS, so we do NOT point HF_HOME at the share for training.

echo "=== PLAIN-LoRA ablation (no HARAM heads) -> $OUTPUT_DIR ==="
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$PROJECT_ROOT/src/training/train.py" \
    --model_id "$MODEL_ID" --data_path "$TRAIN_DATA" --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --haram_enable False \
    --lora_enable True --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_vision_tower True --freeze_llm True --tune_img_projector True \
    --bf16 True --num_crops "$NUM_CROPS" --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" --projector_lr 1e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --max_grad_norm 1.0 --logging_steps 5 --save_strategy no \
    `# NOTE: Phi3VTrainer._save_checkpoint() has an outdated signature (no 'metrics' kwarg) that` \
    `# crashes on intermediate saves in transformers 4.43; the final LoRA adapter is saved by` \
    `# train.py's end-of-training lora branch, so save_strategy=no still produces the model.` \
    --gradient_checkpointing False --ddp_find_unused_parameters True --remove_unused_columns False \
    --report_to tensorboard --dataloader_num_workers 4 --run_name "plainlora_6k_$TS"
echo "=== done -> $OUTPUT_DIR ==="
ls -1 "$OUTPUT_DIR" 2>/dev/null | head
