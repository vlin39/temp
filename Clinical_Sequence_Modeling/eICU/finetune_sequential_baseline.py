import argparse
import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes as bnb
from tqdm import tqdm
import webdataset
import wandb
from braceexpand import braceexpand
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# eICU Modality Mapping
_MODALITY_ID_MAP = {
    '[PREDICT]': 0, 'demographics': 1, 'diagnosis': 2, 
    'treatment': 3, 'medication': 4, 'lab': 5, 'aps': 6
}
_ID_TO_MODALITY = {v: k for k, v in _MODALITY_ID_MAP.items()}

class TabularEncoder(nn.Module):
    """Simple MLP Encoder ported for E2E learning."""
    def __init__(self, input_dim, projection_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class EndToEndBUT(nn.Module):
    def __init__(self, args, input_dims, num_classes):
        super().__init__()
        self.input_embedding_dim = args.input_embedding_dim
        self.num_classes = num_classes
        self.input_dims = input_dims
        
        # 1. Setup Modality Encoders dynamically
        self.encoders = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.encoders[mod] = TabularEncoder(dim, self.input_embedding_dim)

        self.predict_token = nn.Parameter(torch.randn(1, self.input_embedding_dim))

        # 2. Setup LLM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        
        # FIX: Dynamically load attention implementation from args, default to flash_attention_2 if not found
        attn_impl = getattr(args, "attn_implementation", "flash_attention_2")
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model_name, quantization_config=bnb_config,
            trust_remote_code=True, device_map="auto", attn_implementation=attn_impl
        )
        self.llm.gradient_checkpointing_enable()

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        if args.use_lora:
            # Replaced hardcoded target_modules with "all-linear".
            # This safely supports both standard transformers and Qwen3.5's hybrid DeltaNet/MoE architecture.
            lora_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(self.llm, lora_config)
            
        self.llm_hidden_dim = self.llm.config.hidden_size

        # 3. Projection & Output Heads
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_embedding_dim, self.llm_hidden_dim),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, 512),
            nn.ReLU(), nn.Dropout(0.5),  
            nn.Linear(512, self.num_classes)
        )

    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to filter out bitsandbytes 4-bit quantization artifacts"""
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(x in key for x in ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state']):
                continue
            filtered_state_dict[key] = value
        
        # Setting strict=False is required so PyTorch doesn't crash 
        # when it realizes we removed the quantization keys
        return super().load_state_dict(filtered_state_dict, strict=False)
        
    def forward(self, padded_raw_vectors, modality_ids, attention_mask, predict_mask):
        B, S, max_dim = padded_raw_vectors.shape
        device = padded_raw_vectors.device
        
        # Initialize an empty sequence for the embedded features
        embeddings = torch.zeros((B, S, self.input_embedding_dim), device=device, dtype=torch.bfloat16)

        # Vectorized routing: Encode each modality all at once
        for mod_str, dim in self.input_dims.items():
            mod_id = _MODALITY_ID_MAP[mod_str]
            mask = (modality_ids == mod_id)
            
            if mask.any():
                # Slice out only the relevant dimensions for this specific encoder
                vectors = padded_raw_vectors[mask][:, :dim]
                encoded = self.encoders[mod_str](vectors.float())
                embeddings[mask] = encoded.to(torch.bfloat16)

        # Inject Predict Token
        embeddings = torch.where(predict_mask.unsqueeze(-1), self.predict_token.to(torch.bfloat16), embeddings)

        # Pass through LLM
        inputs_embeds = self.input_projection(embeddings).to(self.llm.dtype)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        last_hidden_states = outputs.hidden_states[-1]
        
        return self.classification_head(last_hidden_states.float())

def map_precomputed_sample(sample):
    raw_events = sample['events.pkl']
    events = pickle.loads(raw_events) if isinstance(raw_events, bytes) else raw_events

    raw_labels = sample['labels.npy']
    if isinstance(raw_labels, bytes):
        labels = pickle.loads(raw_labels) if raw_labels.startswith(b'\x80') else np.frombuffer(raw_labels, dtype=np.float32)
    elif isinstance(raw_labels, np.ndarray):
        labels = raw_labels
    elif isinstance(raw_labels, list):
        labels = np.array(raw_labels, dtype=np.float32)
    else:
        labels = np.array([-1.0], dtype=np.float32)

    return {'stay_id': int(sample['__key__']), 'events': events, 'labels': torch.tensor(labels)}

class E2ECollateFn:
    def __init__(self, max_dim):
        self.max_dim = max_dim

    def __call__(self, batch):
        sequences_raw_vectors, sequences_labels, sequences_predict_masks, sequences_mod_ids = [], [], [], []
        final_labels_list = []

        for item in batch:
            events = item['events']
            final_label = item['labels']
            if final_label.ndim == 0: final_label = final_label.unsqueeze(0)
            final_labels_list.append(final_label)
            masked_label = torch.full((final_label.shape[0],), -100, dtype=final_label.dtype)

            curr_raw, curr_labels, curr_masks, curr_mod_ids = [], [], [], []

            for event in events:
                e_type = event['type']
                if e_type == '[PREDICT]':
                    curr_raw.append(torch.zeros(self.max_dim))
                    curr_labels.append(final_label)
                    curr_masks.append(True)
                    curr_mod_ids.append(torch.tensor(0, dtype=torch.long))
                else:
                    vec = torch.from_numpy(event['raw_vector']).squeeze(0)
                    padded_vec = torch.zeros(self.max_dim)
                    padded_vec[:vec.shape[0]] = vec  # Pad to global max_dim
                    
                    curr_raw.append(padded_vec)
                    curr_labels.append(masked_label)
                    curr_masks.append(False)
                    curr_mod_ids.append(torch.tensor(_MODALITY_ID_MAP.get(e_type, 0), dtype=torch.long))

            if not curr_raw: continue
            sequences_raw_vectors.append(torch.stack(curr_raw))
            sequences_labels.append(torch.stack(curr_labels))
            sequences_predict_masks.append(torch.tensor(curr_masks))
            sequences_mod_ids.append(torch.stack(curr_mod_ids))

        if not sequences_raw_vectors: return {}

        padded_raw = pad_sequence(sequences_raw_vectors, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)
        padded_predict_mask = pad_sequence(sequences_predict_masks, batch_first=True, padding_value=False)
        padded_mod_ids = pad_sequence(sequences_mod_ids, batch_first=True, padding_value=0)

        lengths = [len(s) for s in sequences_raw_vectors]
        max_len = padded_raw.shape[1]
        att_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
        for i, l in enumerate(lengths): att_mask[i, :l] = 1

        return {
            'padded_raw_vectors': padded_raw, 'padded_labels': padded_labels,
            'predict_mask': padded_predict_mask, 'attention_mask': att_mask,
            'modality_ids': padded_mod_ids, 'final_labels': torch.stack(final_labels_list)
        }

def get_dataloaders(args, max_dim):
    train_pattern = os.path.join(args.data_dir, "train_{000000..00018}.tar")
    val_pattern = os.path.join(args.data_dir, "val_{000000..000003}.tar")
    
    collate_fn = E2ECollateFn(max_dim)
    train_ds = webdataset.WebDataset(list(braceexpand(train_pattern))).shuffle(1000).decode().map(map_precomputed_sample)
    val_ds = webdataset.WebDataset(list(braceexpand(val_pattern))).decode().map(map_precomputed_sample)
    
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2),
        torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    )

def train_epoch(model, loader, optimizer, scaler, args, device, epoch):
    model.train()
    total_loss, steps = 0, 0
    
    # Initialize tqdm with the total number of samples
    with tqdm(total=args.num_train_samples, desc=f"Train Ep {epoch}", unit="sample") as pbar:
        for batch in loader:
            if not batch: continue
            
            # Extract actual batch size for accurate progress updating
            actual_batch_size = batch['padded_raw_vectors'].size(0)
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    batch['padded_raw_vectors'].to(device),
                    batch['modality_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['predict_mask'].to(device)
                )
                labels = batch['padded_labels'].to(device)

                flat_out = outputs.view(-1, outputs.shape[-1])
                flat_lbl = labels.view(-1)
                mask = (flat_lbl != -100)
                active_out = flat_out[mask].squeeze(-1)
                active_lbl = flat_lbl[mask]
                
                if active_out.numel() == 0: 
                    pbar.update(actual_batch_size)
                    continue
                
                if args.task == 'los_task':
                    loss = nn.MSELoss()(active_out, torch.log1p(active_lbl))
                else:
                    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.76]).to(device))(active_out, active_lbl)
                
                loss = loss / args.accumulation_steps
            
            loss.backward()
            
            if (steps + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.accumulation_steps
            steps += 1
            
            # Update the progress bar by the number of samples processed
            pbar.update(actual_batch_size)
            pbar.set_postfix({"loss": f"{loss.item() * args.accumulation_steps:.4f}"})
            
    return total_loss / max(steps, 1)

def validate(model, loader, args, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with tqdm(total=args.num_val_samples, desc="Validating", unit="sample") as pbar:
        with torch.no_grad():
            for batch in loader:
                if not batch: continue
                
                actual_batch_size = batch['padded_raw_vectors'].size(0)
                
                with autocast(dtype=torch.bfloat16):
                    outputs = model(
                        batch['padded_raw_vectors'].to(device),
                        batch['modality_ids'].to(device),
                        batch['attention_mask'].to(device),
                        batch['predict_mask'].to(device)
                    )
                    labels = batch['padded_labels'].to(device)
                    
                    for i in range(outputs.size(0)): 
                        valid_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                        if len(valid_indices) > 0:
                            last_idx = valid_indices[-1] 
                            all_preds.append(outputs[i, last_idx].squeeze(-1).cpu())
                            all_targets.append(labels[i, last_idx].cpu())
                
                pbar.update(actual_batch_size)
                
    if not all_preds: return {}
        
    preds = torch.stack(all_preds).float().numpy().ravel()
    targets = torch.stack(all_targets).float().numpy().ravel()
    
    if args.task == 'los_task':
        preds = np.clip(preds, a_min=-5.0, a_max=10.0)
        preds_hours = np.expm1(preds) 
        
        # Guard against zero-variance batches
        if len(np.unique(targets)) > 1 and len(np.unique(preds_hours)) > 1:
            pearson_corr, _ = pearsonr(targets, preds_hours)
            spearman_corr, _ = spearmanr(targets, preds_hours)
        else:
            pearson_corr, spearman_corr = 0.0, 0.0
            
        return {
            'val_mse': mean_squared_error(targets, preds_hours),
            'val_mae': mean_absolute_error(targets, preds_hours),  
            'val_pearson': pearson_corr,
            'val_spearman': spearman_corr
        }
    else:
        probs = torch.sigmoid(torch.tensor(preds)).numpy().ravel()
        if len(np.unique(targets)) > 1:
            auroc = roc_auc_score(targets, probs)
            auprc = average_precision_score(targets, probs)
        else:
            auroc, auprc = 0.0, 0.0
            
        return {
            'val_auroc': auroc,
            'val_auprc': auprc
        }
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mortality_task')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--llm_model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--input_embedding_dim', type=int, default=256)
    parser.add_argument('--wandb_project', type=str, default=None) 
    parser.add_argument('--save_prefix', type=str, default="eicu_e2e")
    
    # New arguments for tqdm sample tracking
    parser.add_argument('--num_train_samples', type=int, default=36934)
    parser.add_argument('--num_val_samples', type=int, default=7922)
    
    args = parser.parse_args()
    
    # Dynamically set WandB project based on task if not provided
    wandb_proj = args.wandb_project if args.wandb_project else f"eicu_e2e_{args.task}"
    wandb.init(project=wandb_proj, config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load input dimensions saved by precompute_vectors.py
    with open(os.path.join(args.data_dir, 'input_dims.json'), 'r') as f:
        input_dims = json.load(f)
    max_dim = max(input_dims.values())
    
    train_loader, val_loader = get_dataloaders(args, max_dim)
    
    model = EndToEndBUT(args, input_dims, num_classes=1).to(device)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    best_val = float('inf') if args.task == 'los_task' else -float('inf')
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
        metrics = validate(model, val_loader, args, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"Epoch {epoch} Metrics: {metrics}")
        
        current_metric = metrics.get('val_mse' if args.task == 'los_task' else 'val_auroc')
        
        if current_metric is None:
            continue
            
        is_best = (current_metric < best_val) if args.task == 'los_task' else (current_metric > best_val)
        
        if is_best:
            best_val = current_metric
            epochs_no_improve = 0
            save_path = f"{args.save_prefix}_{args.task}_best.pth"
            print(f"New best model found ({current_metric:.4f}). Saving to {save_path}...")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

if __name__ == "__main__":
    main()