#!/bin/bash
#SBATCH -p rsingh47-gcondo --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 20-00:00:00
#SBATCH -J mort_e2e_sweep_%A_%a
#SBATCH -o logs/eicu_e2e_sweep_%A_%a.out
#SBATCH --array=18-35  

# --- Updated OLMo & Qwen Suite ---
LLMS=(
    "allenai/Olmo-3-7B-Instruct-SFT"
    "allenai/Olmo-Hybrid-Instruct-SFT-7B"
    "Qwen/Qwen3-8B"  
    "Qwen/Qwen3.5-9B"  
)

# --- hyperparameters ---
LRS=(1e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01 0.05 0.1)

num_llms=${#LLMS[@]}
num_lrs=${#LRS[@]}
num_wds=${#WEIGHT_DECAYS[@]}
    
# 3D Grid Index Routing
# Total jobs required = 4 (models) * 3 (lrs) * 3 (wds) = 36 jobs (0 to 35)
llm_idx=$(( SLURM_ARRAY_TASK_ID / (num_lrs * num_wds) ))
remaining=$(( SLURM_ARRAY_TASK_ID % (num_lrs * num_wds) ))
lr_idx=$(( remaining / num_wds ))
wd_idx=$(( remaining % num_wds ))

LLM_MODEL_NAME=${LLMS[$llm_idx]}
LEARNING_RATE=${LRS[$lr_idx]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$wd_idx]}

TASK="mortality_task" 
DATA_DIR="/users/awang463/data/awang463/missingness/eICU/data/finetune_seq/${TASK}/precomputed_vectors"
SAVE_DIR="/users/awang463/scratch"
mkdir -p $SAVE_DIR

WANDB_PROJECT="eicu_hybrid_${TASK}"

LLM_NAME_CLEAN=$(echo "$LLM_MODEL_NAME" | sed 's/\//_/')
SAVE_PREFIX="${SAVE_DIR}/eicu_hybrid_${LLM_NAME_CLEAN}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"

COMMAND="python3 finetune_sequential_baseline.py \
    --task $TASK \
    --data_dir $DATA_DIR \
    --llm_model_name $LLM_MODEL_NAME \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --epochs 40 \
    --use_lora \
    --batch_size 2 \
    --accumulation_steps 8 \
    --wandb_project $WANDB_PROJECT \
    --save_prefix $SAVE_PREFIX"

echo "Running: $COMMAND"
$COMMAND