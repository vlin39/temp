import wandb
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default='cortex_cell_tacit')
    # UPDATED: Changed default output name to match the new hybrid project
    parser.add_argument('--output_csv', type=str, default='hybrid_best_sweep_models.csv')
    args = parser.parse_args()

    # UPDATED: Matches the WANDB_PROJECT parameter from the launch script
    projects = ["eicu_hybrid_mortality_task"]
    api = wandb.Api()
    data = []

    for proj in projects:
        print(f"Fetching runs from {args.entity}/{proj}...")
        try:
            runs = api.runs(f"{args.entity}/{proj}")
        except Exception as e:
            print(f"Error connecting to W&B for project {proj}: {e}")
            continue

        for run in runs:
            config = run.config
            summary = run.summary
            
            task = config.get("task", "unknown_task")
            llm = config.get("llm_model_name", "unknown_llm")
            lr = config.get("learning_rate")
            wd = config.get("weight_decay")
            save_prefix = config.get("save_prefix")
            
            file_path = f"{save_prefix}_{task}_best.pth" if save_prefix else "Unknown"
            
            val_auroc = summary.get("val_auroc", 0.0)
            #print(f"val_auroc: {val_auroc}")
            val_mse = summary.get("val_mse", float('inf'))
            
            data.append({
                "Task": task,
                "LLM": llm,
                "Learning_Rate": lr,
                "Weight_Decay": wd,
                "Val_AUROC": val_auroc,
                "Val_MSE": val_mse,
                "File_Path": file_path,
                "Run_URL": run.url
            })

    df = pd.DataFrame(data)

    if df.empty:
        print("No finished runs found.")
        return

    print(f"Successfully fetched {len(df)} runs. Filtering for the best models...")
    
    best_models = []
    tasks = df['Task'].unique()
    
    for t in tasks:
        task_df = df[df['Task'] == t]
        
        if t == 'los_task':
            idx = task_df.groupby('LLM')['Val_MSE'].idxmin()
        else:
            idx = task_df.groupby('LLM')['Val_AUROC'].idxmax()
            
        best_task_df = task_df.loc[idx]
        best_models.append(best_task_df)

    final_df = pd.concat(best_models).reset_index(drop=True)
    
    print("\n====================== BEST HYBRID MODELS REPORT ======================")
    print(final_df[['Task', 'LLM', 'Learning_Rate', 'Weight_Decay', 'Val_AUROC', 'Val_MSE']].to_string(index=False))
    print("=======================================================================")
    
    final_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved full table including file paths to {args.output_csv}")

if __name__ == "__main__":
    main()