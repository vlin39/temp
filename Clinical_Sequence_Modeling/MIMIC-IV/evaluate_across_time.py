import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import webdataset
from braceexpand import braceexpand
from torch.utils.data import DataLoader

# Import models
from picme_src.models import BeliefUpdateTransformer

def calculate_entropy(logits):
    """Calculate predictive entropy for a batch of logits."""
    if logits.shape[-1] == 1:
        p_pos = torch.sigmoid(logits)
        probs = torch.cat([1 - p_pos, p_pos], dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
    
    probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

def map_precomputed_sample(sample):
    """Decodes .pkl/.npy from precomputed shards"""
    return {
        '__key__': sample['__key__'],
        'events': sample['events.pkl'],
        'labels': torch.tensor(sample['labels.npy'])
    }

def collation_with_event_counts(batch):
    """
    Custom collate that tracks 'data_event_counter' for every token.
    Returns padded tensors including 'event_counts'.
    """
    sequences_of_embeddings = []
    sequences_of_labels = []
    sequences_of_predict_masks = []
    sequences_of_event_counts = [] 
    
    EMBEDDING_DIM = None
    DUMMY_PREDICT_TOKEN = None

    for item in batch:
        patient_events = item['events']
        final_labels = torch.tensor(item['labels'])
        if final_labels.ndim == 0: final_labels = final_labels.unsqueeze(0)
        
        patient_label_vector = final_labels
        masked_label_vector = torch.full_like(patient_label_vector, -100)

        current_embeddings = []
        current_labels = []
        current_mask = []
        current_counts = []
        
        data_event_counter = 0 
        
        if not patient_events: continue
            
        # Discover dim
        if EMBEDDING_DIM is None:
            for event in patient_events:
                if event['type'] != '[PREDICT]' and 'embedding' in event:
                    EMBEDDING_DIM = event['embedding'].shape[-1]
                    DUMMY_PREDICT_TOKEN = torch.zeros(EMBEDDING_DIM)
                    break
        if EMBEDDING_DIM is None: continue

        for event in patient_events:
            etype = event['type']
            
            if etype == '[PREDICT]':
                current_embeddings.append(DUMMY_PREDICT_TOKEN)
                current_labels.append(patient_label_vector)
                current_mask.append(True)
                current_counts.append(data_event_counter) 
            else:
                embedding = torch.from_numpy(event['embedding']).squeeze(0)
                if embedding.shape[0] != EMBEDDING_DIM: continue
                
                current_embeddings.append(embedding)
                current_labels.append(masked_label_vector)
                current_mask.append(False)
                current_counts.append(-1)
                data_event_counter += 1

        if not current_embeddings: continue

        sequences_of_embeddings.append(torch.stack(current_embeddings))
        sequences_of_labels.append(torch.stack(current_labels))
        sequences_of_predict_masks.append(torch.tensor(current_mask))
        sequences_of_event_counts.append(torch.tensor(current_counts, dtype=torch.long))

    if not sequences_of_embeddings: return {}

    padded_embeddings = torch.nn.utils.rnn.pad_sequence(sequences_of_embeddings, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(sequences_of_labels, batch_first=True, padding_value=-100)
    predict_mask = torch.nn.utils.rnn.pad_sequence(sequences_of_predict_masks, batch_first=True, padding_value=False)
    padded_event_counts = torch.nn.utils.rnn.pad_sequence(sequences_of_event_counts, batch_first=True, padding_value=-1)

    lengths = [len(seq) for seq in sequences_of_embeddings]
    max_len = padded_embeddings.shape[1]
    attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    return {
        'padded_embeddings': padded_embeddings,
        'padded_labels': padded_labels,
        'predict_mask': predict_mask,
        'attention_mask': attention_mask,
        'padded_event_counts': padded_event_counts
    }

def get_loader(args):
    data_dir = os.path.join(args.data_dir, "picme_sequential_mortality_precomputed")
    val_shards = "{" + f"000000..{18:06d}" + "}"
    urls = list(braceexpand(os.path.join(data_dir, f"val_{val_shards}.tar")))
    
    dataset = (
        webdataset.WebDataset(urls, nodesplitter=webdataset.split_by_worker)
        .decode("l")
        .map(map_precomputed_sample)
        .with_length(190)
    )
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collation_with_event_counts, num_workers=4)

def plot_metrics(stats_df, model_name, output_dir):
    sns.set_style("whitegrid")
    
    df = stats_df[stats_df['count'] > 5].sort_values('num_events')
    
    if len(df) == 0:
        print("Not enough data to plot.")
        return

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- TOP PLOT: METRICS ---
    color_auc = '#1f77b4' 
    ax_top.plot(df['num_events'], df['auroc'], color=color_auc, marker='o', label='AUROC', linewidth=2)
    ax_top.set_ylabel('AUROC', color=color_auc, fontsize=12, fontweight='bold')
    ax_top.tick_params(axis='y', labelcolor=color_auc)
    ax_top.set_ylim(0.5, 1.0) 
    ax_top.grid(True, linestyle='--', alpha=0.5)

    color_ent = '#d62728' 
    ax_top_ent = ax_top.twinx()
    ax_top_ent.plot(df['num_events'], df['mean_entropy'], color=color_ent, marker='x', linestyle='--', label='Uncertainty', linewidth=2)
    ax_top_ent.set_ylabel('Mean Entropy (Uncertainty)', color=color_ent, fontsize=12, fontweight='bold')
    ax_top_ent.tick_params(axis='y', labelcolor=color_ent)
    
    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_top_ent.get_legend_handles_labels()
    ax_top.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    ax_top.set_title(f"Performance vs Uncertainty\nModel: {model_name}", fontsize=14, pad=10)

    # --- BOTTOM PLOT: COUNTS ---
    color_cnt = '#7f7f7f' 
    ax_bottom.bar(df['num_events'], df['count'], color=color_cnt, alpha=0.6, width=0.8)
    ax_bottom.set_ylabel('Sample Count', color='black', fontsize=12, fontweight='bold')
    ax_bottom.set_xlabel('Number of Clinical Events Observed', fontsize=12, fontweight='bold')
    ax_bottom.set_yscale('log') 
    
    plt.tight_layout()
    
    clean_name = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"{clean_name}_performance_trajectory.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.llm_model_name} on Mortality...")
    
    # Initialize the model using the cleaned args
    model = BeliefUpdateTransformer(args, num_classes=2)
    sd = torch.load(args.state_dict_path, map_location=device)
    
    # Using strict=False to bypass intentional mismatched missing tokens if they arise
    model.load_state_dict(sd, strict=False) 
    model.to(device)
    model.eval()
    
    loader = get_loader(args)
    step_data = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Aggregating Predictions"):
            if not batch: continue
            
            emb = batch['padded_embeddings']
            labels = batch['padded_labels'].to(device)
            p_mask = batch['predict_mask'].to(device)
            att_mask = batch['attention_mask'].to(device)
            event_counts = batch['padded_event_counts'].to(device)
            
            if emb.shape[0] == 0: continue

            p_token = model.predict_token.squeeze(0).to(device)
            input_emb = torch.where(
                p_mask.unsqueeze(-1),
                p_token.view(1, 1, -1),
                emb.to(device)
            )
            
            outputs = model(input_emb, att_mask)
            
            valid_mask = (labels[:, :, 0] != -100) 
            
            flat_outputs = outputs[valid_mask]
            flat_labels = labels[valid_mask][:, 0] 
            flat_counts = event_counts[valid_mask]
            
            if flat_outputs.numel() == 0: continue
            
            entropies = calculate_entropy(flat_outputs)
            
            f_outputs = flat_outputs.cpu().numpy()
            f_labels = flat_labels.cpu().numpy()
            f_counts = flat_counts.cpu().numpy()
            f_entropies = entropies.cpu().numpy()
            
            for i in range(len(f_counts)):
                cnt = int(f_counts[i])
                if cnt not in step_data:
                    step_data[cnt] = {'preds': [], 'labels': [], 'entropies': []}
                
                if flat_outputs.shape[-1] == 1:
                    p_death = torch.sigmoid(torch.tensor(f_outputs[i])).item()
                else:
                    p_death = torch.softmax(torch.tensor(f_outputs[i]), dim=0)[1].item()
                    
                step_data[cnt]['preds'].append(p_death)
                step_data[cnt]['labels'].append(f_labels[i])
                step_data[cnt]['entropies'].append(f_entropies[i])

    results = []
    print("Computing metrics per step...")
    
    sorted_steps = sorted(step_data.keys())
    for step in sorted_steps:
        data_dict = step_data[step]
        y_true = np.array(data_dict['labels'])
        y_score = np.array(data_dict['preds'])
        ents = np.array(data_dict['entropies'])
        
        count = len(y_true)
        mean_ent = np.mean(ents)
        
        if len(np.unique(y_true)) > 1:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except Exception:
                auroc = np.nan
        else:
            auroc = np.nan
            
        results.append({
            'num_events': step,
            'auroc': auroc,
            'mean_entropy': mean_ent,
            'count': count
        })
        
    df = pd.DataFrame(results)
    
    os.makedirs(args.output_dir, exist_ok=True)
    clean_name = args.llm_model_name.replace("/", "_")
    df.to_csv(os.path.join(args.output_dir, f"{clean_name}_metrics.csv"), index=False)
    
    plot_metrics(df, args.llm_model_name, args.output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stepwise Evaluation for Belief Update Transformer")
    
    # --- Paths ---
    parser.add_argument("--state_dict_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--data_dir", type=str, default="/users/awang463/data/awang463/missing_modalities/everything")
    parser.add_argument("--output_dir", type=str, default="olmo_stepwise_plots")
    
    # --- Model Architecture ---
    parser.add_argument("--llm_model_name", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--input_embedding_dim", type=int, default=256)
    parser.add_argument("--sequential_model", action="store_true", help="Flag to indicate sequential BUT model")
    parser.add_argument("--use_evidential_head", action="store_true", help="Flag if model was trained with Evidential Deep Learning")
    
    # --- LoRA (PEFT) Parameters ---
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA for model loading")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.3)
    
    # --- Dataloader ---
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    # Run the main evaluation function
    main(args)