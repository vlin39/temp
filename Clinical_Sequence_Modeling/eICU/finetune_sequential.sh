#!/bin/bash

# ======================= SLURM JOB ARRAY DIRECTIVES =======================
# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 20-00:00:00

# Name and Logs
#SBATCH -J eicu_but_sweep_%A_%a
#SBATCH -o logs/eicu_sweep_%A_%a.out
#SBATCH -e logs/eicu_sweep_%A_%a.err

# ARRAY SIZE CALCULATION:
# 6 Models * 3 LRs * 3 WDs = 54 Jobs
# Run this script with: sbatch --array=0-53 launch_eicu_finetune_sweep.sh
# ==========================================================================

module load python/3.9.16s-x3wdtvt
# source /users/awang463/venvs/myenv/bin/activate

# --- 1. HYPERPARAMETER GRID ---

LLMS=(
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-v0.1"
    "deepseek-ai/deepseek-llm-7b-base"
    "microsoft/Phi-3-mini-4k-instruct"
    "BioMistral/BioMistral-7B"
    "epfl-llm/meditron-7b"
)

LRS=(1e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01 0.05 0.1)

# --- 2. ARRAY INDEX LOGIC (The 3D Grid) ---

num_llms=${#LLMS[@]}
num_lrs=${#LRS[@]}
num_wds=${#WEIGHT_DECAYS[@]}

# Logic:
# Index changes fastest for Weight Decay, then LR, then LLM
llm_idx=$(( SLURM_ARRAY_TASK_ID / (num_lrs * num_wds) ))
remaining=$(( SLURM_ARRAY_TASK_ID % (num_lrs * num_wds) ))
lr_idx=$(( remaining / num_wds ))
wd_idx=$(( remaining % num_wds ))

# Extract specific params for this job
LLM_MODEL_NAME=${LLMS[$llm_idx]}
LEARNING_RATE=${LRS[$lr_idx]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$wd_idx]}

# --- 3. STATIC PARAMETERS ---
TASK="mortality_task"  # Change to "los_task" if needed
HEAD_TYPE="traditional" # or "evidential"
WANDB_PROJECT="eicu_but_${TASK}_sweep"
DATA_DIR="/users/awang463/data/awang463/missingness/eICU/data/finetune_seq"
SAVE_DIR="/users/awang463/scratch"

# Ensure the scratch directory exists so torch.save doesn't crash
mkdir -p $SAVE_DIR

# Clean name for WandB
LLM_NAME_CLEAN=$(echo "$LLM_MODEL_NAME" | sed 's/\//_/')
WANDB_NAME="BUT-${LLM_NAME_CLEAN}-lr${LEARNING_RATE}-wd${WEIGHT_DECAY}"

# UNIQUE SAVE PREFIX: Includes the scratch path, clean model name, LR, and WD
SAVE_PREFIX="${SAVE_DIR}/eicu_BUT_${LLM_NAME_CLEAN}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"

echo "===================================================="
echo "JOB ID: $SLURM_ARRAY_TASK_ID"
echo "  - LLM: $LLM_MODEL_NAME"
echo "  - LR: $LEARNING_RATE"
echo "  - WD: $WEIGHT_DECAY"
echo "  - Task: $TASK"
echo "  - Save Prefix: $SAVE_PREFIX"
echo "===================================================="

# --- 4. COMMAND CONSTRUCTION ---

COMMAND="python3 finetune_sequential.py \
    --task $TASK \
    --data_dir $DATA_DIR \
    --llm_model_name $LLM_MODEL_NAME \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --epochs 40 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --modality_dropout 0 \
    --batch_size 1 \
    --accumulation_steps 8 \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME \
    --save_prefix $SAVE_PREFIX"

# Add Evidential Head if selected
if [ "$HEAD_TYPE" = "evidential" ]; then
    COMMAND+=" --use_evidential_head"
fi

echo "Running: $COMMAND"
$COMMAND
