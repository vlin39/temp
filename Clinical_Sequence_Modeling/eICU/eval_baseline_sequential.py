import os
import json
import argparse
import pandas as pd
import torch
import webdataset
from braceexpand import braceexpand

from finetune_sequential_baseline import EndToEndBUT, E2ECollateFn, map_precomputed_sample, validate

class SpoofedArgs:
    def __init__(self, llm_model_name, task):
        self.llm_model_name = llm_model_name
        self.task = task
        self.use_evidential_head = False
        self.input_embedding_dim = 256
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.batch_size = 2 
        self.num_val_samples = 7922 
        # UPDATED: Explicitly set attention for the new EndToEndBUT initialization
        self.attn_implementation = "flash_attention_2"

def get_test_dataloader(data_dir, task, max_dim, batch_size):
    task_dir = os.path.join(data_dir, task, "precomputed_vectors")
    test_pattern = os.path.join(task_dir, "test_{000000..000003}.tar") 
    
    collate_fn = E2ECollateFn(max_dim)
    test_ds = webdataset.WebDataset(list(braceexpand(test_pattern))).decode().map(map_precomputed_sample)
    
    return torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True
    )

def main():
    parser = argparse.ArgumentParser()
    # UPDATED: Match the new output from the extraction script
    parser.add_argument('--input_csv', type=str, default='hybrid_best_sweep_models.csv')
    parser.add_argument('--output_csv', type=str, default='hybrid_final_test_metrics.csv')
    parser.add_argument('--base_data_dir', type=str, default='/users/awang463/data/awang463/missingness/eICU/data/finetune_seq')
    script_args = parser.parse_args()

    if not os.path.exists(script_args.input_csv):
        print(f"Input CSV {script_args.input_csv} not found!")
        return

    df = pd.read_csv(script_args.input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for _, row in df.iterrows():
        print(f"\n=======================================================")
        print(f"Evaluating {row['LLM']} for {row['Task']}...")
        print(f"=======================================================")
        
        args = SpoofedArgs(row['LLM'], row['Task'])
        
        task_dir = os.path.join(script_args.base_data_dir, row['Task'], "precomputed_vectors")
        dims_path = os.path.join(task_dir, 'input_dims.json')
        with open(dims_path, 'r') as f:
            input_dims = json.load(f)
        max_dim = max(input_dims.values())
        
        test_loader = get_test_dataloader(script_args.base_data_dir, row['Task'], max_dim, args.batch_size)
        model = EndToEndBUT(args, input_dims, num_classes=1).to(device)
        
        try:
            model.load_state_dict(torch.load(row['File_Path'], map_location=device))
            print(f"Weights loaded successfully from {row['File_Path']}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            continue

        metrics = validate(model, test_loader, args, device)
        
        row_data = row.to_dict()
        for k, v in metrics.items():
            row_data[k.replace('val_', 'test_')] = v
            
        results.append(row_data)
        print(row_data)
        
        del model
        torch.cuda.empty_cache()

    out_df = pd.DataFrame(results)
    out_df.to_csv(script_args.output_csv, index=False)
    print(f"\nSaved full test metrics to {script_args.output_csv}")

if __name__ == "__main__":
    main()