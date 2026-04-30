import wandb
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default='cortex_cell_tacit')
    parser.add_argument('--project', type=str, default='hybrid_mimic_mortality_but_finetuning')
    parser.add_argument('--output_csv', type=str, default='best_olmo_models.csv')
    args = parser.parse_args()

    api = wandb.Api()
    
    print(f"Fetching runs from {args.entity}/{args.project}...")
    try:
        runs = api.runs(f"{args.entity}/{args.project}")
    except Exception as e:
        print(f"Error connecting to W&B: {e}")
        return

    data = []
    target_models = [
        "allenai/Olmo-3-7B-Instruct-SFT",
        "allenai/Olmo-Hybrid-Instruct-SFT-7B"
    ]

    for run in runs:
        if run.state != "finished":
            continue
            
        config = run.config
        summary = run.summary
        
        llm = config.get("llm_model_name", "unknown_llm")
        if llm not in target_models:
            continue

        task = config.get("task", "mortality")
        
        # --- THE FIX: Unpack lists from argparse ---
        lr_raw = config.get("learning_rate")
        lr = lr_raw[0] if isinstance(lr_raw, list) else lr_raw
        
        wd_raw = config.get("weight_decay")
        wd = wd_raw[0] if isinstance(wd_raw, list) else wd_raw
        # -------------------------------------------
        
        save_prefix = config.get("save_prefix", "mortality_mimic_hybrid")
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 8)
        seed = config.get("seed_number", 42)
        
        # Construct the exact file path based on finetune_sequential.py
        llm_name_clean = llm.replace('/', '_')
        path_str = f"{save_prefix}_BUT_{task}_{llm_name_clean}_lr{lr}_r{lora_r}_a{lora_alpha}_seed{seed}"
        file_path = f"/users/awang463/scratch/{path_str}/{path_str}_best.pth"
        
        val_auroc = summary.get("val_auroc", 0.0)
        
        data.append({
            "Task": task,
            "LLM": llm,
            "Learning_Rate": lr,
            "Weight_Decay": wd,
            "Val_AUROC": val_auroc,
            "File_Path": file_path,
            "Run_URL": run.url
        })

    df = pd.DataFrame(data)

    if df.empty:
        print("No finished runs found for the target Olmo models.")
        return

    print(f"Successfully fetched {len(df)} Olmo runs. Filtering for the best models...")
    
    # Get the index of the max Val_AUROC for each LLM
    idx = df.groupby('LLM')['Val_AUROC'].idxmax()
    best_df = df.loc[idx].reset_index(drop=True)

    print("\n====================== BEST MODELS REPORT ======================")
    display_cols = ['LLM', 'Learning_Rate', 'Weight_Decay', 'Val_AUROC']
    print(best_df[display_cols].to_string(index=False))
    print("================================================================")
    
    best_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved full table including file paths to {args.output_csv}")

if __name__ == "__main__":
    main()