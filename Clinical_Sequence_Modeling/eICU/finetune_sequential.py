import argparse
import logging
import os
import pickle
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_squared_error
import bitsandbytes as bnb
from tqdm import tqdm
import webdataset
import wandb
from braceexpand import braceexpand
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from flash_attn import flash_attn_func
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# eICU Modality Mapping
_MODALITY_ID_MAP = {
    '[PREDICT]': 0, 'demographics': 1, 'diagnosis': 2, 
    'treatment': 3, 'medication': 4, 'lab': 5, 'aps': 6
}

def check_tensor_health(tensor, name="Tensor"):
    if tensor is None:
        return
    
    is_nan = torch.isnan(tensor).any()
    is_inf = torch.isinf(tensor).any()
    
    if is_nan or is_inf:
        print(f"\n[!] CRITICAL: {name} contains {'NaN' if is_nan else ''} {'Inf' if is_inf else ''}")
        return False # Unhealthy
    return True # Healthy


class BeliefUpdateTransformer(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.llm_model_name = args.llm_model_name
        self.input_embedding_dim = args.input_embedding_dim
        self.num_classes = num_classes
        self.use_evidential_head = args.use_evidential_head

        # 1. Load the LLM with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto", # Automatically handle device placement
            attn_implementation=args.attn_implementation
        )

        self.llm.gradient_checkpointing_enable()
        print("Gradient Checkpointing ENABLED (Slower, Memory Efficient)")

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # 2. Setup LoRA (PEFT) if requested
        if args.use_lora:
            # Find common target modules (can be expanded)
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(self.llm, lora_config)
            print("LoRA (PEFT) enabled.")
            self.llm.print_trainable_parameters()
        
        # Get LLM hidden dimension
        self.llm_hidden_dim = self.llm.config.hidden_size

        # 3. Input Projection Layer
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_embedding_dim, self.llm_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim)
        )

        self.predict_token = nn.Parameter(
            torch.randn(1, self.input_embedding_dim)
        )

        # 4. Output Classification Head
        if self.use_evidential_head:
            self.classification_head = EvidentialHead(self.llm_hidden_dim, self.num_classes)
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(self.llm_hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),  
                nn.Linear(512, self.num_classes)
            )

    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle quantized weights"""
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(x in key for x in ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state']):
                continue
            filtered_state_dict[key] = value
        
        return super().load_state_dict(filtered_state_dict, strict=False)
        
    def forward(self, modality_embeddings_sequence, attention_mask):
        # 1. Project modality embeddings into the LLM's hidden space
        inputs_embeds = self.input_projection(modality_embeddings_sequence) 
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        # 2. Pass the sequence of embeddings through the LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
        
        # Get the hidden states from the last layer
        last_hidden_states = outputs.hidden_states[-1]

        if torch.isnan(last_hidden_states).any():
            print("[!] CRITICAL: NaNs generated inside LLM layers!")
        
        # 3. Pass *every* time-step's hidden state through the classification head
        final_outputs = self.classification_head(last_hidden_states.float())
        
        return final_outputs


def evidential_loss_function(alpha, labels_one_hot, current_epoch, max_epochs, annealing_steps=10):
    """
    Type-II Maximum Likelihood Loss for Evidential Deep Learning.
    """
    # 1. Calculate the Dirichlet strength (S)
    dirichlet_strength = torch.sum(alpha, dim=1, keepdim=True) # [N, 1]

    # 2. Calculate the main cross-entropy-like loss
    log_likelihood = torch.sum(labels_one_hot * (torch.log(dirichlet_strength) - torch.log(alpha)), dim=1)
    loss_ce = torch.mean(log_likelihood)

    # 3. Calculate the KL divergence term (annealed)
    annealing_coeff = torch.min(
        torch.tensor(1.0, device=alpha.device),
        torch.tensor(current_epoch / annealing_steps, device=alpha.device)
    )

    # Prior (uniform Dirichlet)
    alpha_0 = torch.ones_like(alpha)
    
    kl_div = torch.sum(
        (alpha - alpha_0) * (torch.digamma(alpha) - torch.digamma(dirichlet_strength)), 
        dim=1
    )
    kl_div_loss = torch.mean(kl_div)
    
    return loss_ce + (annealing_coeff * kl_div_loss)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_precomputed_sample(sample):
    """
    Robustly decodes events and labels.
    """
    # 1. Handle Events (Pickle)
    raw_events = sample['events.pkl']
    if isinstance(raw_events, bytes):
        events = pickle.loads(raw_events)
    else:
        events = raw_events

    # 2. Handle Labels (Numpy)
    raw_labels = sample['labels.npy']

    if isinstance(raw_labels, bytes):
        if raw_labels.startswith(b'\x80'):
            labels = pickle.loads(raw_labels)
        else:
            labels = np.frombuffer(raw_labels, dtype=np.float32)
            
    elif isinstance(raw_labels, np.ndarray):
        labels = raw_labels
        
    elif isinstance(raw_labels, list):
        labels = np.array(raw_labels, dtype=np.float32)
        
    else:
        labels = np.array([-1.0], dtype=np.float32)

    #print(f"labels: {labels}")

    return {
        'stay_id': int(sample['__key__']),
        'events': events, 
        'labels': torch.tensor(labels) 
    }
    
def sequential_collate_fn(batch):
    sequences_embeddings, sequences_labels, sequences_predict_masks, sequences_mod_ids = [], [], [], []
    final_labels_list = []
    embedding_dim = 256 
    dummy_predict_token = torch.zeros(embedding_dim)

    for item in batch:
        events = item['events']

        final_label = item['labels']
        if final_label.ndim == 0: final_label = final_label.unsqueeze(0)
        final_labels_list.append(final_label)
        masked_label = torch.full((final_label.shape[0],), -100, dtype=final_label.dtype)

        curr_embeddings, curr_labels, curr_masks, curr_mod_ids = [], [], [], []

        for event in events:
            e_type = event['type']
            if e_type == '[PREDICT]':
                curr_embeddings.append(dummy_predict_token)
                curr_labels.append(final_label)
                curr_masks.append(True)
                curr_mod_ids.append(torch.tensor(0, dtype=torch.long))
            else:
                emb = torch.from_numpy(event['embedding']).squeeze(0)
                if emb.shape[0] != embedding_dim: continue 
                curr_embeddings.append(emb)
                curr_labels.append(masked_label)
                curr_masks.append(False)
                curr_mod_ids.append(torch.tensor(_MODALITY_ID_MAP.get(e_type, 0), dtype=torch.long))

        if not curr_embeddings: continue
        sequences_embeddings.append(torch.stack(curr_embeddings))
        sequences_labels.append(torch.stack(curr_labels))
        sequences_predict_masks.append(torch.tensor(curr_masks))
        sequences_mod_ids.append(torch.stack(curr_mod_ids))

    if not sequences_embeddings: return {}

    padded_embeddings = pad_sequence(sequences_embeddings, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)
    padded_predict_mask = pad_sequence(sequences_predict_masks, batch_first=True, padding_value=False)
    padded_mod_ids = pad_sequence(sequences_mod_ids, batch_first=True, padding_value=0)

    lengths = [len(s) for s in sequences_embeddings]
    max_len = padded_embeddings.shape[1]
    att_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, l in enumerate(lengths): att_mask[i, :l] = 1

    return {
        'padded_embeddings': padded_embeddings, 'padded_labels': padded_labels,
        'predict_mask': padded_predict_mask, 'attention_mask': att_mask,
        'modality_ids': padded_mod_ids, 'final_labels': torch.stack(final_labels_list)
    }

def get_dataloaders(args):
    data_path = os.path.join(args.data_dir, args.task, "precomputed")
    # Adjust these ranges based on your actual file counts
    train_pattern = os.path.join(data_path, "train_{000000..00018}.tar")
    val_pattern = os.path.join(data_path, "val_{000000..000003}.tar")
    
    print(f"Loading from: {train_pattern}")
    
    # PERFORMANCE FIX: Added persistent_workers and prefetch_factor
    train_ds = webdataset.WebDataset(list(braceexpand(train_pattern))).shuffle(1000).decode().map(map_precomputed_sample)
    val_ds = webdataset.WebDataset(list(braceexpand(val_pattern))).decode().map(map_precomputed_sample)
    
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=sequential_collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2),
        torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=sequential_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    )

def apply_modality_dropout(embeddings, modality_ids, drop_prob=0.0):
    if drop_prob == 0.0: return embeddings
    droppable = [2, 3, 4, 5, 6] # Diagnosis, Tx, Med, Lab, APS
    decision_lut = torch.ones((embeddings.shape[0], 7), device=embeddings.device)
    probs = torch.rand((embeddings.shape[0], len(droppable)), device=embeddings.device)
    decision_lut[:, droppable] = (probs > drop_prob).float()
    mask = decision_lut.gather(1, modality_ids.clamp(max=6)).unsqueeze(-1)
    return embeddings * mask

def train_epoch(model, loader, optimizer, scaler, args, device, epoch):
    model.train()
    # Hardcoded total samples for accurate progress tracking
    num_train_samples = 36934 
    num_train_batches = num_train_samples // args.batch_size
    
    total_loss = 0
    steps = 0
    predict_token_emb = model.predict_token.squeeze(0).to(device)
    
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", total=num_train_batches)
    
    for batch in pbar:
        if not batch: continue
        
        embeddings = apply_modality_dropout(batch['padded_embeddings'].to(device), batch['modality_ids'].to(device), args.modality_dropout)
        # Inject Learnable Predict Token
        embeddings = torch.where(batch['predict_mask'].to(device).unsqueeze(-1), predict_token_emb, embeddings)
        
        # PERFORMANCE FIX: Explicit bfloat16 autocast
        with autocast(dtype=torch.bfloat16):
            outputs = model(embeddings, batch['attention_mask'].to(device))
            labels = batch['padded_labels'].to(device)

            # --- 1. HEALTH CHECK: Did the model die on the PREVIOUS batch? ---
            # Check a raw trainable weight to see if it's already NaN
            if torch.isnan(model.input_projection[0].weight).any():
                print(f"\n[!] CRITICAL: Model weights corrupted BEFORE processing Step {steps}!")
                print("The previous batch caused a gradient explosion. You need to lower your learning rate or increase gradient clipping.")
                break # Stop training, the model is dead
    
            # --- 2. HEALTH CHECK: Are the inputs corrupted? ---
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print(f"\n[!] CRITICAL: Poisoned Embeddings detected at Step {steps}!")
                torch.save(batch, "poison_batch_inputs.pt") # Save the evidence
                print("Saved 'poison_batch_inputs.pt' for offline inspection.")
                continue # Skip this batch so it doesn't kill the model
                
            if (batch['attention_mask'].sum(dim=1) == 0).any():
                print(f"\n[!] CRITICAL: Completely empty attention mask detected at Step {steps}!")
                continue
    
            with autocast(dtype=torch.bfloat16):
                outputs = model(embeddings, batch['attention_mask'].to(device))
                labels = batch['padded_labels'].to(device)
                
                # --- 3. HEALTH CHECK: Did the forward pass generate NaNs? ---
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n[!] CRITICAL: Forward pass generated NaNs at Step {steps}!")
                    # Save the exact embeddings that caused the LLM to choke
                    torch.save({'embeddings': embeddings, 'mask': batch['attention_mask']}, "poison_forward_pass.pt")
                    print("Saved 'poison_forward_pass.pt'. Check for extreme outlier values in the embeddings.")
                    break 
                    
            # --- FIX: FLATTEN AND MASK (Anytime Prediction) ---
            # 1. Flatten [Batch, Seq] -> [Batch*Seq]
            flat_out = outputs.view(-1, outputs.shape[-1])
            flat_lbl = labels.view(-1)
            
            # 2. Select only valid steps
            mask = (flat_lbl != -100)
            active_out = flat_out[mask].squeeze(-1)
            active_lbl = flat_lbl[mask]
            # --------------------------------------------------
            
            if active_out.numel() == 0: continue
            
            if args.use_evidential_head:
                loss = evidential_loss_function(active_out, active_lbl.float(), epoch, args.annealing_steps)
            elif args.task == 'los_task':
                log_lbl = torch.log1p(active_lbl)
                loss = nn.MSELoss()(active_out, log_lbl)
            else:
                # Binary Mortality (BCE expects [N, 1])
                pos_weight = torch.tensor([10.76]).to(device)
                loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(active_out, active_lbl)
            
            loss = loss / args.accumulation_steps
        
        loss.backward()
        
        if (steps + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss_val = loss.item() * args.accumulation_steps
        total_loss += current_loss_val
        steps += 1
        
        avg_loss = total_loss / max(steps, 1)
        pbar.set_postfix({"loss": f"{current_loss_val:.4f}", "avg_loss": f"{avg_loss:.4f}"})
        
    return total_loss / max(steps, 1)
    

# def validate(model, loader, args, device):
#     model.eval()
#     all_preds, all_targets = [], []
#     predict_token_emb = model.predict_token.squeeze(0).to(device)
    
#     num_val_samples = 7922
#     num_val_batches = num_val_samples // args.batch_size
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Validating", total=num_val_batches):
#             if not batch: continue
            
#             embeddings = batch['padded_embeddings'].to(device)
#             embeddings = torch.where(batch['predict_mask'].to(device).unsqueeze(-1), predict_token_emb, embeddings)
            
#             # PERFORMANCE FIX: Explicit bfloat16 autocast
#             with autocast(dtype=torch.bfloat16):
#                 outputs = model(embeddings, batch['attention_mask'].to(device))
#                 labels = batch['padded_labels'].to(device)
                
#                 # --- FIX: FLATTEN AND MASK ---
#                 flat_out = outputs.view(-1, outputs.shape[-1])
#                 flat_lbl = labels.view(-1)
                
#                 mask = (flat_lbl != -100)
#                 active_out = flat_out[mask].squeeze(-1)
#                 active_lbl = flat_lbl[mask]
#                 # -----------------------------
            
#             if active_out.numel() == 0: continue
            
#             all_preds.append(active_out.cpu())
#             all_targets.append(active_lbl.cpu())
            
#     if not all_preds: 
#         print("Warning: No valid predictions in validation set.")
#         return {}
        
#     preds = torch.cat(all_preds).float().numpy()
#     targets = torch.cat(all_targets).float().numpy()
    
#     if args.task == 'los_task':
#         mse = mean_squared_error(targets, preds)
#         mae = mean_absolute_error(targets, preds)
        
#         # Guard against zero-variance batches which return NaN for correlation
#         if len(np.unique(targets)) > 1 and len(np.unique(preds)) > 1:
#             pearson_corr, _ = pearsonr(targets, preds)
#             spearman_corr, _ = spearmanr(targets, preds)
#         else:
#             pearson_corr, spearman_corr = 0.0, 0.0
            
#         return {
#             'val_mse': mse,
#             'val_mae': mae,
#             'val_pearson': pearson_corr,
#             'val_spearman': spearman_corr
#         }
#     else:
#         # Check for single-class batches to avoid NaN
#         if len(np.unique(targets)) < 2:
#             return {'val_auroc': 0.0, 'val_auprc': 0.0}

#         # Apply Sigmoid for Binary Logits
#         probs = torch.sigmoid(torch.tensor(preds)).numpy().ravel()
#         targets = targets.ravel()
        
#         try:
#             return {
#                 'val_auroc': roc_auc_score(targets, probs), 
#                 'val_auprc': average_precision_score(targets, probs)
#             }
#         except Exception as e:
#             print(f"Metric Error: {e}")
#             return {}

def validate(model, loader, args, device):
    model.eval()
    all_preds, all_targets = [], []
    predict_token_emb = model.predict_token.squeeze(0).to(device)
    
    num_val_samples = 7922
    num_val_batches = num_val_samples // args.batch_size
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", total=num_val_batches):
            if not batch: continue
            
            # FIX 1: Sanitize validation embeddings just like the train loop
            embeddings = batch['padded_embeddings'].to(device)
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=10.0, neginf=-10.0)
            
            embeddings = torch.where(batch['predict_mask'].to(device).unsqueeze(-1), predict_token_emb, embeddings)
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(embeddings, batch['attention_mask'].to(device))
                labels = batch['padded_labels'].to(device)
                
                for i in range(outputs.size(0)): 
                    valid_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    
                    if len(valid_indices) > 0:
                        last_idx = valid_indices[-1] 
                        
                        pred = outputs[i, last_idx].squeeze(-1) 
                        lbl = labels[i, last_idx]
                        
                        all_preds.append(pred.cpu())
                        all_targets.append(lbl.cpu())
                
    if not all_preds: 
        print("Warning: No valid predictions in validation set.")
        return {}
        
    # FIX 2: Force strict 1D shapes to prevent Scikit-Learn broadcasting matrices
    preds = torch.stack(all_preds).float().numpy().ravel()
    targets = torch.stack(all_targets).float().numpy().ravel()
    
    if args.task == 'los_task':
        # FIX 3: Clamp the log-space predictions before exponentiating. 
        # (log(30 days in hours) is ~6.5, so clamping at 10.0 is very safe and prevents Inf/NaN overflows)
        preds = np.clip(preds, a_min=-5.0, a_max=10.0)
        preds_hours = np.expm1(preds) 
        
        mse = mean_squared_error(targets, preds_hours)
        mae = mean_absolute_error(targets, preds_hours)
        
        # We guard against constant arrays to prevent Pearsonr/Spearmanr warnings
        if len(np.unique(targets)) > 1 and len(np.unique(preds_hours)) > 1:
            pearson_corr, _ = pearsonr(targets, preds_hours)
            spearman_corr, _ = spearmanr(targets, preds_hours)
        else:
            pearson_corr, spearman_corr = 0.0, 0.0
            
        return {
            'val_mse': mse,
            'val_mae': mae,
            'val_pearson': pearson_corr,
            'val_spearman': spearman_corr
        }
    else:
        if len(np.unique(targets)) < 2:
            return {'val_auroc': 0.0, 'val_auprc': 0.0}

        probs = torch.sigmoid(torch.tensor(preds)).numpy().ravel()
        
        try:
            return {
                'val_auroc': roc_auc_score(targets, probs), 
                'val_auprc': average_precision_score(targets, probs)
            }
        except Exception as e:
            print(f"Metric Error: {e}")
            return {}
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mortality_task')
    parser.add_argument('--data_dir', type=str, default="/users/awang463/data/awang463/missingness/eICU/data/finetune_seq")
    parser.add_argument('--llm_model_name', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--modality_dropout', type=float, default=0.2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--annealing_steps', type=int, default=10) 
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_evidential_head', action='store_true')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help="Enable for memory savings, disable for speed")
    parser.add_argument('--input_embedding_dim', type=int, default=256)
    parser.add_argument('--wandb_project', type=str, default="eicu_finetune") 
    parser.add_argument('--wandb_name', type=str, default=None) 
    parser.add_argument('--save_prefix', type=str, default="eicu_but")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    train_loader, val_loader = get_dataloaders(args)
    # Mortality/LoS are single output tasks (logits or regression value)
    model = BeliefUpdateTransformer(args, num_classes=1) 
    model.to(device)
    
    # --- OPTIMIZER WITH WEIGHT DECAY ---
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    print(f"Starting training for {args.task} | LR: {args.learning_rate} | WD: {args.weight_decay}")
    
    # --- EARLY STOPPING & SAVING VARIABLES ---
    best_val_metric = -float('inf') if args.task != 'los_task' else float('inf')
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        # 1. Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
        
        # 2. Validate
        metrics = validate(model, val_loader, args, device)
        
        # 3. Log
        log_dict = {"epoch": epoch, "train_loss": train_loss}
        log_dict.update(metrics)
        wandb.log(log_dict)
        print(f"Ep {epoch}: {log_dict}")
        
        # 4. Check Metric for Saving & Early Stopping
        current_metric = None
        is_improvement = False
        
        if args.task == 'los_task':
            # Minimize MSE
            if 'val_mse' in metrics:
                current_metric = metrics['val_mse']
                if current_metric < best_val_metric:
                    is_improvement = True
                    best_val_metric = current_metric
        else:
            # Maximize AUROC
            if 'val_auroc' in metrics:
                current_metric = metrics['val_auroc']
                if current_metric > best_val_metric:
                    is_improvement = True
                    best_val_metric = current_metric
        
        # 5. Save & Stop Logic
        if is_improvement:
            epochs_no_improve = 0
            save_path = f"{args.save_prefix}_{args.task}_best.pth"
            print(f"New Best Metric ({best_val_metric:.4f})! Saving to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break

if __name__ == "__main__":
    main()