import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import webdataset
from braceexpand import braceexpand
from torch.utils.data import DataLoader

# UPDATED: Removed the Phi-3 Hotfix entirely to prevent interference with Qwen/OLMo.

from finetune_sequential_baseline import EndToEndBUT, map_precomputed_sample

_MODALITY_ID_MAP = {
    '[PREDICT]': 0, 'demographics': 1, 'diagnosis': 2, 
    'treatment': 3, 'medication': 4, 'lab': 5, 'aps': 6
}

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
        # UPDATED: Simplified to always use flash_attention_2 for Qwen/OLMo
        self.attn_implementation = "flash_attention_2"

class E2ECollateWithEventCounts:
    """
    Custom collator for raw tabular vectors that tracks 'event_counts' 
    for stepwise evaluation.
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    def __call__(self, batch):
        sequences_raw_vectors, sequences_labels, sequences_predict_masks, sequences_mod_ids = [], [], [], []
        sequences_event_counts = []

        for item in batch:
            events = item['events']
            final_label = item['labels']
            if final_label.ndim == 0: final_label = final_label.unsqueeze(0)
            masked_label = torch.full((final_label.shape[0],), -100, dtype=final_label.dtype)

            curr_raw, curr_labels, curr_masks, curr_mod_ids, curr_counts = [], [], [], [], []
            data_event_counter = 0

            for event in events:
                e_type = event['type']
                if e_type == '[PREDICT]':
                    curr_raw.append(torch.zeros(self.max_dim))
                    curr_labels.append(final_label)
                    curr_masks.append(True)
                    curr_mod_ids.append(torch.tensor(0, dtype=torch.long))
                    curr_counts.append(data_event_counter)
                else:
                    vec = torch.from_numpy(event['raw_vector']).squeeze(0)
                    padded_vec = torch.zeros(self.max_dim)
                    padded_vec[:vec.shape[0]] = vec  
                    
                    curr_raw.append(padded_vec)
                    curr_labels.append(masked_label)
                    curr_masks.append(False)
                    curr_mod_ids.append(torch.tensor(_MODALITY_ID_MAP.get(e_type, 0), dtype=torch.long))
                    curr_counts.append(-1)
                    
                    data_event_counter += 1

            if not curr_raw: continue
            sequences_raw_vectors.append(torch.stack(curr_raw))
            sequences_labels.append(torch.stack(curr_labels))
            sequences_predict_masks.append(torch.tensor(curr_masks))
            sequences_mod_ids.append(torch.stack(curr_mod_ids))
            sequences_event_counts.append(torch.tensor(curr_counts, dtype=torch.long))

        if not sequences_raw_vectors: return {}

        padded_raw = torch.nn.utils.rnn.pad_sequence(sequences_raw_vectors, batch_first=True, padding_value=0.0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(sequences_labels, batch_first=True, padding_value=-100)
        padded_predict_mask = torch.nn.utils.rnn.pad_sequence(sequences_predict_masks, batch_first=True, padding_value=False)
        padded_mod_ids = torch.nn.utils.rnn.pad_sequence(sequences_mod_ids, batch_first=True, padding_value=0)
        padded_event_counts = torch.nn.utils.rnn.pad_sequence(sequences_event_counts, batch_first=True, padding_value=-1)

        lengths = [len(s) for s in sequences_raw_vectors]
        max_len = padded_raw.shape[1]
        att_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
        for i, l in enumerate(lengths): att_mask[i, :l] = 1

        return {
            'padded_raw_vectors': padded_raw, 'padded_labels': padded_labels,
            'predict_mask': padded_predict_mask, 'attention_mask': att_mask,
            'modality_ids': padded_mod_ids, 'padded_event_counts': padded_event_counts
        }

def get_test_dataloader(data_dir, task, max_dim, batch_size):
    task_dir = os.path.join(data_dir, task, "precomputed_vectors")
    test_pattern = os.path.join(task_dir, "test_{000000..000003}.tar") 
    
    collate_fn = E2ECollateWithEventCounts(max_dim)
    test_ds = webdataset.WebDataset(list(braceexpand(test_pattern))).decode().map(map_precomputed_sample)
    
    return DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def plot_metrics(stats_df, model_name, task, output_dir):
    sns.set_style("whitegrid")
    
    df = stats_df[stats_df['count'] > 5].sort_values('num_events')
    
    if len(df) == 0:
        print("Not enough data to plot.")
        return

    metric_name = 'Spearman Correlation' if task == 'los_task' else 'AUROC'
    metric_col = 'spearman' if task == 'los_task' else 'auroc'

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    color_metric = '#1f77b4' 
    ax_top.plot(df['num_events'], df[metric_col], color=color_metric, marker='o', label=metric_name, linewidth=2)
    ax_top.set_ylabel(metric_name, color=color_metric, fontsize=12, fontweight='bold')
    ax_top.tick_params(axis='y', labelcolor=color_metric)
    
    if task != 'los_task':
        ax_top.set_ylim(0.5, 1.0) 
    else:
        y_min, y_max = df[metric_col].min(), df[metric_col].max()
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax_top.set_ylim(min(0.0, y_min - padding), min(1.0, y_max + padding))
        
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(loc='lower right')
    ax_top.set_title(f"Performance vs Sequence Length\nBase Model: {model_name} | Task: {task}", fontsize=14, pad=10)

    color_cnt = '#7f7f7f' 
    ax_bottom.bar(df['num_events'], df['count'], color=color_cnt, alpha=0.6, width=0.8)
    ax_bottom.set_ylabel('Sample Count', color='black', fontsize=12, fontweight='bold')
    ax_bottom.set_xlabel('Number of Clinical Events Observed', fontsize=12, fontweight='bold')
    ax_bottom.set_yscale('log') 
    
    plt.tight_layout()
    
    clean_name = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"{clean_name}_{task}_base_trajectory.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    # UPDATED: Default input matches our new extraction script output
    parser.add_argument('--input_csv', type=str, default='hybrid_best_sweep_models.csv')
    parser.add_argument('--base_data_dir', type=str, default='/users/awang463/data/awang463/missingness/eICU/data/finetune_seq')
    parser.add_argument('--output_dir', type=str, default='base_stepwise_plots')
    script_args = parser.parse_args()

    if not os.path.exists(script_args.input_csv):
        print(f"Input CSV {script_args.input_csv} not found!")
        return

    df = pd.read_csv(script_args.input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(script_args.output_dir, exist_ok=True)

    for _, row in df.iterrows():
        print(f"\n=======================================================")
        print(f"Stepwise Eval: {row['LLM']} for {row['Task']}...")
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

        model.eval()
        step_data = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Aggregating Predictions"):
                if not batch: continue
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        batch['padded_raw_vectors'].to(device),
                        batch['modality_ids'].to(device),
                        batch['attention_mask'].to(device),
                        batch['predict_mask'].to(device)
                    )
                
                labels = batch['padded_labels'].to(device)
                event_counts = batch['padded_event_counts'].to(device)
                
                flat_out = outputs.view(-1, outputs.shape[-1])
                flat_lbl = labels.view(-1)
                flat_cnt = event_counts.view(-1)
                
                valid_mask = (flat_lbl != -100)
                
                active_out = flat_out[valid_mask].squeeze(-1).float().cpu().numpy()
                active_lbl = flat_lbl[valid_mask].float().cpu().numpy()
                active_cnt = flat_cnt[valid_mask].cpu().numpy()
                
                if len(active_out) == 0: continue
                
                if row['Task'] == 'los_task':
                    clamped_out = np.clip(active_out, a_min=-5.0, a_max=10.0)
                    preds_to_store = np.expm1(clamped_out)
                else:
                    preds_to_store = torch.sigmoid(torch.tensor(active_out)).numpy()
                
                for i in range(len(active_cnt)):
                    cnt = int(active_cnt[i])
                    if cnt not in step_data:
                        step_data[cnt] = {'preds': [], 'labels': []}
                        
                    step_data[cnt]['preds'].append(preds_to_store[i])
                    step_data[cnt]['labels'].append(active_lbl[i])

        results = []
        print("Computing metrics per step...")
        
        sorted_steps = sorted(step_data.keys())
        for step in sorted_steps:
            data_dict = step_data[step]
            y_true = np.array(data_dict['labels'])
            y_score = np.array(data_dict['preds'])
            count = len(y_true)
            
            auroc, spearman_corr = np.nan, np.nan
            
            if row['Task'] == 'los_task':
                if len(np.unique(y_true)) > 1 and len(np.unique(y_score)) > 1:
                    try:
                        spearman_corr, _ = spearmanr(y_true, y_score)
                    except Exception:
                        spearman_corr = np.nan
            else:
                if len(np.unique(y_true)) > 1:
                    try:
                        auroc = roc_auc_score(y_true, y_score)
                    except Exception:
                        auroc = np.nan
                
            results.append({
                'num_events': step,
                'auroc': auroc,
                'spearman': spearman_corr,
                'count': count
            })
            
        metrics_df = pd.DataFrame(results)
        clean_name = row['LLM'].replace("/", "_")
        metrics_df.to_csv(os.path.join(script_args.output_dir, f"{clean_name}_{row['Task']}_metrics.csv"), index=False)
        
        plot_metrics(metrics_df, row['LLM'], row['Task'], script_args.output_dir)
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()