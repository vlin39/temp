#!/bin/bash

# ======================= SLURM JOB ARRAY DIRECTIVES =======================
# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 4

# Memory
#SBATCH --mem=150G
#SBATCH -t 20:00:00

## Provide a job name and logging based on array ID
#SBATCH -J but_sweep_%A_%a
#SBATCH -o logs/finetune_sequential_sweep_%A_%a.out
#SBATCH -e logs/finetune_sequential_sweep_%A_%a.err
# ==========================================================================

module load python/3.9.16s-x3wdtvt
# <source your environment>

LLMS=(
    "allenai/Olmo-3-7B-Instruct-SFT"
    "allenai/Olmo-Hybrid-Instruct-SFT-7B"
)

HEAD_TYPE="traditional" # Set between evidential and traditional
LRS=(1e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01 0.05 0.1) 
LORA_R=8          
LORA_ALPHA=8
LORA_DROPOUT=0.3    
ANNEALING=15


# Set this to --compute_embeddings_on_fly if you want to use the raw path
#ON_FLY_FLAG="--compute_embeddings_on_fly" 
ON_FLY_FLAG="" # Comment out above and use this to use precomputed

num_llms=${#LLMS[@]}
num_lrs=${#LRS[@]}
num_wds=${#WEIGHT_DECAYS[@]}  # NEW: Count weight decay options

# Calculate array indices based on LLM, LR, and Weight Decay
total_runs=$((num_llms * num_lrs * num_wds))

# NEW: Calculate indices for 3D parameter space
llm_idx=$(( SLURM_ARRAY_TASK_ID / (num_lrs * num_wds) ))
remaining=$(( SLURM_ARRAY_TASK_ID % (num_lrs * num_wds) ))
lr_idx=$(( remaining / num_wds ))
wd_idx=$(( remaining % num_wds ))

LLM_MODEL_NAME=${LLMS[$llm_idx]}
LEARNING_RATE=${LRS[$lr_idx]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$wd_idx]}  # NEW: Get weight decay value


# --- Static Parameters ---
SEED_NUMBER=42
BATCH_SIZE=1
ACCUMULATION_STEPS=8
EPOCHS=40
OBJECTIVE="auroc"
METRICS=("auroc" "auprc" "f1")
WANDB_PROJECT="hybrid_mimic_mortality_but_finetuning"
STATE_DICT_PATH="/users/awang463/data/awang463/missing_modalities/models/contrastive/0_masked_global_1e-4-wd_0.005-gamma_2.0_dropped_filler/masked_global_filler_prob_0.0_lr_0.0001-wd_0.005-temp_0.07_20251209-213342_final.pth"
#WANDB_PROJECT="og_pheno_but_finetuning"
TASK="mortality"
SAVE_PREFIX="mortality_mimic_hybrid"
FUSION_METHOD="concatenation"
MODALITIES=("img" "ts" "demo" "text_rad" "text_ds")

# --- Dynamic WandB Run Name (Updated) ---
LLM_NAME_CLEAN=$(echo "$LLM_MODEL_NAME" | sed 's/\//_/')
# UPDATED: Include weight decay in run name
WANDB_NAME="${SAVE_PREFIX}-${LLM_NAME_CLEAN}-lr${LEARNING_RATE}-wd${WEIGHT_DECAY}"

# --- Construct the command ---
echo "===================================================="
echo "SLURM JOB ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "RUNNING SEQUENTIAL (BUT) JOB WITH PARAMS:"
echo "  - LLM: $LLM_MODEL_NAME"
echo "  - Head Type: $HEAD_TYPE"
echo "  - LR: $LEARNING_RATE"
echo "  - Weight Decay: $WEIGHT_DECAY"  # NEW: Log weight decay
echo "  - LoRA R: $LORA_R"
echo "  - LoRA Alpha: $LORA_ALPHA"
echo "  - WandB Name: $WANDB_NAME"
echo "===================================================="

# --- UPDATED COMMAND CONSTRUCTION ---
COMMAND="python3 finetune_sequential.py \
    $ON_FLY_FLAG \
    --sequential_model \
    --use_lora \
    --llm_model_name $LLM_MODEL_NAME \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --annealing_steps $ANNEALING \
    --model_name "baseline" \
    --modalities ${MODALITIES[@]} \
    --fusion_method $FUSION_METHOD \
    --seed_category single \
    --fusion_method $FUSION_METHOD \
    --seed_number $SEED_NUMBER \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --epochs $EPOCHS \
    --objective $OBJECTIVE \
    --metrics ${METRICS[@]} \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME \
    --task $TASK \
    --state_dict $STATE_DICT_PATH \
    --save_prefix $SAVE_PREFIX \
    --weight_decay $WEIGHT_DECAY --task $TASK"

# Conditionally add the evidential head flag
if [ "$HEAD_TYPE" = "evidential" ]; then
    COMMAND+=" --use_evidential_head"
fi

# Run the command
echo "Running command: $COMMAND"
$COMMAND

