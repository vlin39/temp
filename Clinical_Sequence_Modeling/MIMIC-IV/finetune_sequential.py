import argparse
import logging
import pprint
import time
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import bitsandbytes as bnb

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

import wandb

from transformers import AutoTokenizer
import os

# picme_src imports
from picme_src.models import (
    BeliefUpdateTransformer,
    evidential_loss_function,
    MultiModalContrastiveModel,
)
from picme_src.data import get_sequential_pheno_dataloaders, get_sequential_mortality_dataloaders
from picme_src.argparser import finetune_arg_parser
from picme_src import data, models, picme_utils
from picme_src.data import * # Cleanup CUDA mem at startup
torch.cuda.empty_cache()
gc.collect()

_DATA_DIR = "/users/awang463/data/awang463/missing_modalities/everything"
_FULL_DATA_DIR = "/users/awang463/data/awang463/missing_modalities/full"

_MODALITY_SHAPES = {"demo": 44, "ts": 76, "projection": 256}
_RENAME_MODALITY = {"mortality": "in-hospital-mortality"}

_TASK_WEIGHTS = {
    "in-hospital-mortality": data._MORTALITY_CLASS_WEIGHTS,
    "phenotyping": data._PHENO_CLASS_WEIGHTS,
}

_TASK_CLASSES = {
    "mortality": 2,
    "phenotyping": 25,
}

TOKENIZER = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# --- TRANSFORMS (AUGMENTATION REMOVED) ---
# We use the standard deterministic transform for everything now
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_modality_dropout(embeddings, modality_ids, drop_prob=0.20):
    """
    Memory-efficient modality dropout using Lookup Table & In-place multiplication.
    """
    if drop_prob == 0.0: 
        return embeddings
    
    batch_size, seq_len = modality_ids.shape
    device = embeddings.device
    
    droppable_ids = [2, 3, 4, 5]
    max_id = 7 
    decision_lut = torch.ones((batch_size, max_id), device=device, dtype=embeddings.dtype)
    
    random_probs = torch.rand((batch_size, len(droppable_ids)), device=device)
    keep_decisions = (random_probs > drop_prob).to(dtype=embeddings.dtype)
    
    decision_lut[:, droppable_ids] = keep_decisions
    
    safe_ids = modality_ids.clamp(max=max_id - 1)
    keep_mask = decision_lut.gather(1, safe_ids)
    
    embeddings.mul_(keep_mask.unsqueeze(-1))
    
    return embeddings

# -------------------------------

def encode_batch_on_fly(raw_batch, encoder_model, args, device, predict_token_embedding):
    """
    Encodes data using encoder_model with CPU Shuffle (OOM Fix).
    Augmentation has been removed for stability.
    """
    sequences_of_embeddings = []
    sequences_of_labels = []
    sequences_of_predict_masks = []
    
    token_key_map = {
        'cxr': 'img', 'ds_note': 'text_ds', 'rad_note': 'text_rad',
        'demo': 'demo', 'ehr_bin': 'ts'
    }
    
    REQUIRED_MODALITIES = {'img', 'text_ds', 'text_rad', 'demo', 'ts'}

    encoder_model.eval()

    for patient_item in raw_batch:
        patient_events = patient_item['events']
        
        # --- FILTER INCOMPLETE PATIENTS ---
        if getattr(args, 'filter_incomplete', False):
            present_modalities = set()
            for event in patient_events:
                e_type = event['type']
                if e_type in token_key_map:
                    present_modalities.add(token_key_map[e_type])
            
            if not REQUIRED_MODALITIES.issubset(present_modalities):
                continue
        # ---------------------------------------

        # Label extraction
        labels_raw = patient_item.get('labels', None) 
        if labels_raw is None and 'labels' in patient_item:
            labels_raw = np.array(patient_item['labels'])
        if labels_raw is None:
            labels_raw = np.zeros(_TASK_CLASSES[args.task])

        # --- FIX: Ensure Mortality Labels are 1D, not scalars ---
        final_labels = torch.tensor(labels_raw)
        if final_labels.ndim == 0:
            final_labels = final_labels.unsqueeze(0)
        # --------------------------------------------------------
        patient_label_vector = final_labels
        masked_label_vector = torch.full((final_labels.shape[0],), -100, dtype=patient_label_vector.dtype)

        current_sequence_embeddings = []
        current_sequence_labels = []
        current_predict_mask = []

        for event in patient_events:
            event_type = event['type']
            
            if event_type == '[PREDICT]':
                # Append CPU tensor placeholder
                current_sequence_embeddings.append(torch.zeros(1, 256)) 
                current_sequence_labels.append(patient_label_vector)
                current_predict_mask.append(True)
                continue

            # --- Encoding Logic ---
            event_data = event['data']
            emb = None
            token_key = token_key_map.get(event_type)

            if not token_key: continue

            try:
                with torch.no_grad(): 
                    # ENCODING HAPPENS ON GPU
                    if args.model_name == 'baseline':
                        if token_key == 'demo':
                            data_t = torch.tensor(event_data, dtype=torch.float).unsqueeze(0).to(device)
                            emb = encoder_model.encoders['demo'](data_t)
                        elif token_key == 'ts':
                            data_t = torch.tensor(event_data, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
                            lens = torch.tensor([1], dtype=torch.long).cpu()
                            emb = encoder_model.encoders['ts'](data_t, lens)
                            if emb.dim() == 1: emb = emb.unsqueeze(0)
                        elif token_key == 'img':
                            img = Image.open(event_data).convert('RGB')
                            img_t = IMG_TRANSFORM(img).unsqueeze(0).to(device)
                            emb = encoder_model.encoders['img'](img_t)
                        elif token_key in ['text_ds', 'text_rad']:
                            inputs = TOKENIZER(event_data, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
                            emb = encoder_model.encoders[token_key](inputs['input_ids'], inputs['attention_mask'])
                    else:
                        # Contrastive logic
                        if token_key == 'demo':
                            data_t = torch.tensor(event_data, dtype=torch.float).unsqueeze(0).to(device)
                            emb = encoder_model.demo_encoder(data_t)
                        elif token_key == 'ts':
                            data_t = torch.tensor(event_data, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
                            lens = torch.tensor([1], dtype=torch.long).cpu()
                            emb = encoder_model.ts_encoder(data_t, lens)
                            if emb.dim() == 1: emb = emb.unsqueeze(0)
                        elif token_key == 'img':
                            img = Image.open(event_data).convert('RGB')
                            img_t = IMG_TRANSFORM(img).unsqueeze(0).to(device)
                            emb = encoder_model.image_encoder(img_t)
                        elif token_key in ['text_ds', 'text_rad']:
                            inputs = TOKENIZER(event_data, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
                            emb = encoder_model.text_encoder(inputs['input_ids'], inputs['attention_mask'])
                
                if emb is not None:
                      emb = encoder_model.projection(emb)

            except Exception as e:
                pass

            if emb is None:
                if getattr(args, 'filter_incomplete', False):
                      continue 
                
                if hasattr(encoder_model, 'modality_token_map'):
                      emb = encoder_model.modality_token_map[token_key].unsqueeze(0)
                else:
                      continue 

            # --- THE CPU SHUFFLE ---
            # 1. Move to CPU immediately
            # 2. Detach from graph
            current_sequence_embeddings.append(emb.detach().cpu())
            
            # 3. Explicit deletion to free GPU mem for next event
            del emb
            if 'img_t' in locals(): del img_t
            if 'inputs' in locals(): del inputs
            if 'data_t' in locals(): del data_t
            
            current_sequence_labels.append(masked_label_vector)
            current_predict_mask.append(False)

        if not current_sequence_embeddings: continue

        sequences_of_embeddings.append(torch.cat(current_sequence_embeddings, dim=0))
        sequences_of_labels.append(torch.stack(current_sequence_labels))
        sequences_of_predict_masks.append(torch.tensor(current_predict_mask))

    if not sequences_of_embeddings: return {}

    # Padding (On CPU)
    padded_embeddings = pad_sequence(sequences_of_embeddings, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(sequences_of_labels, batch_first=True, padding_value=-100)
    predict_mask = pad_sequence(sequences_of_predict_masks, batch_first=True, padding_value=False)

    lengths = [len(seq) for seq in sequences_of_embeddings]
    max_len = padded_embeddings.shape[1]
    attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    return {
        'padded_embeddings': padded_embeddings,
        'padded_labels': padded_labels,
        'predict_mask': predict_mask,
        'attention_mask': attention_mask
    }
    
def check_tensor_health(tensor, name="Tensor"):
    if tensor is None:
        return
    
    is_nan = torch.isnan(tensor).any()
    is_inf = torch.isinf(tensor).any()
    
    if is_nan or is_inf:
        print(f"\n[!] CRITICAL: {name} contains {'NaN' if is_nan else ''} {'Inf' if is_inf else ''}")
        return False 
    return True 

def train_sequential_model(dataloaders_dict, criterion, model, config, path, args, device, encoder_model=None):
    save_dir = f"/users/awang463/scratch/{path}"
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    best_val_metric = -1.0 
    patience_counter = 0 
    predict_token_embedding = model.predict_token.squeeze(0).to(device)

    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        # --- Training ---
        model.train()
        running_train_loss = 0.0
        train_loader = dataloaders_dict["train"]
        train_pbar = tqdm(train_loader, total=1420, desc="Training", leave=False)
        
        train_all_preds = []
        train_all_targets = []
        
        for i, batch in enumerate(train_pbar):
            if args.compute_embeddings_on_fly:
                # ENCODE ON THE FLY (Augmentation removed inside function)
                processed_batch = encode_batch_on_fly(
                    batch, encoder_model, args, device, predict_token_embedding
                )
                if not processed_batch: continue
                
                padded_embeddings = processed_batch['padded_embeddings'].to(device)
                padded_labels = processed_batch['padded_labels']
                predict_mask = processed_batch['predict_mask']
                padded_mask = processed_batch['attention_mask']
            else:
                padded_embeddings_cpu = batch['padded_embeddings']
                padded_labels = batch['padded_labels'].to(device)
                predict_mask = batch['predict_mask'].to(device)
                padded_mask = batch['attention_mask'].to(device)
                modality_ids = batch['modality_ids'].to(device)
                
                if padded_embeddings_cpu.shape[0] == 0: continue

                padded_embeddings = padded_embeddings_cpu.to(device)

                # --- APPLY MODALITY DROPOUT ---
                # Only apply during training!
                padded_embeddings = apply_modality_dropout(
                    padded_embeddings, 
                    modality_ids, 
                    drop_prob=0.20 
                )
                # ------------------------------

                predict_mask_expanded = predict_mask.unsqueeze(-1) 
                predict_token_expanded = predict_token_embedding.view(1, 1, -1)
                
                padded_embeddings = torch.where(
                    predict_mask_expanded,
                    predict_token_expanded,
                    padded_embeddings 
                )

            with autocast():
                current_seq_len = padded_embeddings.shape[1]
                # if current_seq_len > 2000: # Adjust threshold based on your expectations
                #     print(f"WARNING: Large Sequence Length detected: {padded_embeddings.shape}")
                #     del padded_embeddings, padded_labels, predict_mask, padded_mask
                
                #     # 2. Delete auxiliary tensors if they exist in local scope
                #     if 'modality_ids' in locals(): del modality_ids
                #     if 'predict_mask_expanded' in locals(): del predict_mask_expanded
                #     if 'predict_token_expanded' in locals(): del predict_token_expanded
                    
                #     # 3. Clear cache to be safe and move to next iteration
                #     torch.cuda.empty_cache()
                #     continue
                outputs = model(padded_embeddings, padded_mask)
                
                valid_loss_mask = (padded_labels != -100).all(dim=-1)
                active_outputs = outputs[valid_loss_mask].to(device)
                del outputs
                active_labels = padded_labels[valid_loss_mask]
                
                if active_outputs.numel() == 0: continue

                # --- START CHANGE: Loss Calculation Logic ---
                if args.use_evidential_head:
                    loss = criterion(active_outputs, active_labels.float(), epoch, args.annealing_steps)
                else:
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        loss = criterion(active_outputs.to(device), active_labels.squeeze(-1).long().to(device))
                    elif isinstance(criterion, nn.BCEWithLogitsLoss):
                        # Special handling for Mortality with BCE:
                        # Model outputs [N, 2], BCE needs [N, 1] (the positive class logit)
                        if active_outputs.shape[-1] == 2:
                            # Slice to get the positive class (index 1) and keep dimension
                            active_outputs = active_outputs[:, 1].unsqueeze(-1)
                        
                        loss = criterion(active_outputs.to(device), active_labels.float().to(device))
                    else:
                        # Fallback for other potential losses
                        loss = criterion(active_outputs.to(device), active_labels.float().to(device))
                # --- END CHANGE ---
                
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_train_loss += loss.item() * args.accumulation_steps
            train_pbar.set_postfix(loss=loss.item() * args.accumulation_steps)
            train_all_preds.append(active_outputs.detach().cpu())
            train_all_targets.append(active_labels.detach().cpu())

            del padded_embeddings, padded_labels, predict_mask, padded_mask, loss, active_outputs, active_labels
            if not args.compute_embeddings_on_fly:
                del modality_ids
            

        avg_train_loss = running_train_loss / 1420

        # Calculate Training AUROC
        train_auroc = None
        train_auprc = None
        train_f1_macro = None
        
        if train_all_targets:
            train_all_preds = torch.cat(train_all_preds, dim=0).float()
            train_all_targets = torch.cat(train_all_targets, dim=0).numpy()
            
            if train_all_targets.ndim == 2 and train_all_targets.shape[1] == 1:
                train_all_targets = train_all_targets.ravel()
            
            if args.use_evidential_head:
                dirichlet_strength = torch.sum(train_all_preds, dim=1, keepdim=True)
                train_probs = (train_all_preds / dirichlet_strength).numpy()
            elif args.task == 'phenotyping':
                train_probs = torch.sigmoid(train_all_preds).numpy()
            else:
                # --- START CHANGE: Handle Binary Mortality converted to BCE ---
                # If we used BCE, preds are [N, 1], so use Sigmoid.
                # If we used CrossEntropy, preds are [N, 2], so use Softmax.
                if train_all_preds.shape[1] == 1:
                    train_probs = torch.sigmoid(train_all_preds).numpy().ravel()
                else:
                    train_probs = torch.softmax(train_all_preds, dim=1).numpy()[:, 1]
            
            try:
                train_auroc = roc_auc_score(train_all_targets, train_probs, average="macro")
                train_auprc = average_precision_score(train_all_targets, train_probs, average="macro")
                
                train_preds_binary = (train_probs > 0.5).astype(int)
                train_f1_macro = f1_score(train_all_targets, train_preds_binary, average="macro", zero_division=0)
                
                print(f"  Train AUROC:  {train_auroc:.4f}")
                print(f"  Train AUPRC:  {train_auprc:.4f}")
                print(f"  Train F1 (Ma): {train_f1_macro:.4f}")

                del train_all_preds, train_all_targets
                
            except ValueError as e:
                print(f"  Could not compute training metrics: {e}")
        else:
            print("  Training set yielded no predictions for metrics.")

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_loader = dataloaders_dict["val"]
        
        all_preds = []
        all_targets = []

        val_pbar = tqdm(val_loader, desc="Validation", total=310, leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                if args.compute_embeddings_on_fly:
                    processed_batch = encode_batch_on_fly(
                        batch, encoder_model, args, device, predict_token_embedding
                    )
                    
                    if not processed_batch: 
                        continue

                    padded_embeddings = processed_batch['padded_embeddings']
                    padded_labels = processed_batch['padded_labels']
                    predict_mask = processed_batch['predict_mask']
                    padded_mask = processed_batch['attention_mask']
                    
                else:
                    padded_embeddings_cpu = batch['padded_embeddings']
                    
                    if padded_embeddings_cpu.shape[0] == 0:
                        continue
                    
                    padded_embeddings = padded_embeddings_cpu.to(device)
                    padded_labels = batch['padded_labels'].to(device)
                    predict_mask = batch['predict_mask'].to(device)
                    padded_mask = batch['attention_mask'].to(device)

                predict_mask_expanded = predict_mask.unsqueeze(-1) 
                predict_token_expanded = predict_token_embedding.view(1, 1, -1)
                
                padded_embeddings = torch.where(
                    predict_mask_expanded.to(device),
                    predict_token_expanded.to(device),
                    padded_embeddings.to(device) 
                ).to(device)

                outputs = model(padded_embeddings, padded_mask)
                
                valid_loss_mask = (padded_labels != -100).all(dim=-1)
                active_outputs = outputs[valid_loss_mask]
                active_labels = padded_labels[valid_loss_mask]

                if active_labels.shape[0] == 0:
                    continue
                    
                if args.use_evidential_head:
                    loss = criterion(
                        active_outputs, 
                        active_labels.float(), 
                        epoch, 
                        args.annealing_steps
                    )
                else:
                    if isinstance(criterion, nn.CrossEntropyLoss):
                         loss = criterion(active_outputs.to(device), active_labels.squeeze(-1).long().to(device))
                    elif isinstance(criterion, nn.BCEWithLogitsLoss):
                         if active_outputs.shape[-1] == 2:
                             active_outputs = active_outputs[:, 1].unsqueeze(-1)
                         loss = criterion(active_outputs.to(device), active_labels.float().to(device))
                    else:
                         loss = criterion(active_outputs.to(device), active_labels.float().to(device))
                # --- END CHANGE ---
                
                running_val_loss += loss.item()

                all_preds.append(active_outputs.cpu())
                all_targets.append(active_labels.cpu())
                
                del padded_embeddings, padded_labels, predict_mask, padded_mask, outputs, loss, active_outputs, active_labels
                #if not args.compute_embeddings_on_fly:
                #    del modality_ids

        avg_val_loss = running_val_loss / 310
        
        if not all_targets:
            print("Validation set yielded no predictions. Skipping metrics.")
            if train_auroc is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_auroc": train_auroc,
                    "train_auprc": train_auprc,
                    "train_f1_macro": train_f1_macro,
                }
                wandb.log(log_dict)
            continue
            
        all_preds = torch.cat(all_preds, dim=0).float()
        all_targets = torch.cat(all_targets, dim=0).numpy() 

        if all_targets.ndim == 2 and all_targets.shape[1] == 1:
             all_targets = all_targets.ravel()

        if args.use_evidential_head:
            dirichlet_strength = torch.sum(all_preds, dim=1, keepdim=True)
            probs = (all_preds / dirichlet_strength).numpy()
        elif args.task == 'phenotyping':
            probs = torch.sigmoid(all_preds).numpy()
        else:
            # --- START FIX ---
            # Check dimensions to decide between Sigmoid (1 output) and Softmax (2 outputs)
            if all_preds.shape[1] == 1:
                # Binary Case (BCE): Output is [N, 1]. Use Sigmoid.
                probs = torch.sigmoid(all_preds).numpy().ravel()
            else:
                # Multiclass Case (CE): Output is [N, 2]. Use Softmax and take class 1.
                probs = torch.softmax(all_preds, dim=1).numpy()[:, 1]

        try:
            val_auroc = roc_auc_score(all_targets, probs, average="macro")
            val_auprc = average_precision_score(all_targets, probs, average="macro")
            
            preds_binary = (probs > 0.5).astype(int)
            val_f1_macro = f1_score(all_targets, preds_binary, average="macro", zero_division=0)
            val_f1_micro = f1_score(all_targets, preds_binary, average="micro", zero_division=0)

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            if train_auroc is not None:
                print(f"  Train AUROC: {train_auroc:.4f}")
            print(f"  Val Loss:    {avg_val_loss:.4f}")
            print(f"  Val AUROC:   {val_auroc:.4f}")
            print(f"  Val AUPRC:   {val_auprc:.4f}")
            print(f"  Val F1 (Ma): {val_f1_macro:.4f}")
            
            log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "val_f1_macro": val_f1_macro,
                "val_f1_micro": val_f1_micro,
            }
            
            if train_auroc is not None:
                log_dict.update({
                    "train_auroc": train_auroc,
                    "train_auprc": train_auprc,
                    "train_f1_macro": train_f1_macro
                })
            
            wandb.log(log_dict)

            current_metric = log_dict.get(f"val_{args.objective}", -1.0)
            
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                save_path = os.path.join(save_dir, f"{path}_best.pth")
                print(f"  New best {args.objective}! Saving model to {save_path}")
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                
            try:
                del padded_embeddings, padded_labels, predict_mask, padded_mask
                del outputs, loss, active_outputs, active_labels
                
                # These might not exist depending on the path taken
                if 'predict_mask_expanded' in locals(): del predict_mask_expanded
                if 'predict_token_expanded' in locals(): del predict_token_expanded
                if 'modality_ids' in locals(): del modality_ids
                
            except UnboundLocalError:
                # This just means the variables weren't defined, which is fine.
                pass
            torch.cuda.empty_cache()

        except ValueError as e:
            print(f"Could not compute metrics: {e}")
            log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            if train_auroc is not None:
                log_dict.update({
                    "train_auroc": train_auroc,
                    "train_auprc": train_auprc,
                    "train_f1_macro": train_f1_macro
                })
            wandb.log(log_dict)

        lr_scheduler.step()
        
        next_lr = optimizer.param_groups[0]['lr']
        print(f"  Updated Learning Rate for next epoch: {next_lr:.6f}")
        if patience_counter >= 5:
            print(f"EARLY STOPPING: No improvement in 5 epochs.")
            break

    print("--- Training Complete ---")
    print(f"Best Validation {args.objective}: {best_val_metric:.4f}")
    
def train_model(
    dataloaders_dict,
    criterion,
    model_contrastive,
    config,
    path,
    args,
    device,
):
    torch.cuda.empty_cache()
    picme_utils.set_seed(config.seed_number)

    projection_dim = _MODALITY_SHAPES["projection"]

    # Initialize the separated classification head
    classifier = models.ClassificationHead(
        projection_dim=projection_dim,
        num_classes=_TASK_CLASSES[args.task],
        fusion_method=args.fusion_method,
        num_modalities=len(args.modalities),
        modality_lambdas=args.modality_lambdas,
    )
    classifier = classifier.to(device)

    num_epochs = config.epochs
    
    lr = config.learning_rate if isinstance(config.learning_rate, float) else config.learning_rate
    print(f"Optimizer: {'adamW'}, LR: {lr}")

    # Setup Optimizer to handle unified architectures
    params_to_optimize = list(classifier.parameters())
    if args.model_name == "baseline":
        # Baseline trains encoders end-to-end
        params_to_optimize += list(model_contrastive.parameters())
    elif not args.freeze:
        # Contrastive trains encoders end-to-end only if not frozen
        params_to_optimize += list(model_contrastive.parameters())
        
    # Standard PyTorch AdamW to handle parameter lists
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=0.01)

    since = time.time()
    best_metric = 0.0
    patience = 5
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            classifier.train() if phase == "train" else classifier.eval()
            batch_iterator = tqdm(dataloaders_dict[phase], total=len(dataloaders_dict[phase]), desc=f"Iterating {phase}")

            running_loss = 0
            all_logits = []
            all_preds = []
            all_labels = []
            num_batches = 0

            for batch_idx, batch in enumerate(batch_iterator):
                optimizer.zero_grad(set_to_none=True)

                modalities_data, modalities_type, mask_rad, mask_ds, ts_lens, labels, present_mask_batch = \
                        picme_utils.prep_data(args.modalities, batch, device, missing=True)

                if args.freeze and args.model_name != "baseline":
                    model_contrastive.eval()
                elif phase == "train" and (not args.freeze or args.model_name == "baseline"):
                    model_contrastive.train()

                with torch.set_grad_enabled(phase == "train"):
                    # 1. Get Embeddings (works for BOTH baseline and contrastive now)
                    embeddings = model_contrastive(
                        modalities_data, modalities_type, mask_rad, mask_ds, ts_lens
                    )
                    
                    # 2. Apply Missingness Tokens (Fair application for BOTH models)
                    final_embeddings = []
                    for i, modality_name in enumerate(args.modalities):
                        emb = embeddings[i]
                        token = model_contrastive.modality_token_map[modality_name]
                        mask = present_mask_batch[:, i].unsqueeze(1).to(device)
                        final_emb = torch.where(mask, emb, token)
                        final_embeddings.append(final_emb)

                    # 3. Fuse and Classify
                    concatenated_embeddings = picme_utils.secure_fusion(
                        final_embeddings, device, args.fusion_method
                    )
                    outputs = classifier(concatenated_embeddings)
                    
                    labels = labels.float()
                    labels = picme_utils.extract_labels(labels, args.task).to(device)
                    outputs = outputs.to(torch.float)
                    
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                    preds = picme_utils.predict(outputs, task=args.task)
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    all_logits.extend(outputs.cpu().detach().numpy())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                    del outputs, loss, final_embeddings, embeddings

                num_batches += 1

            epoch_loss = running_loss / num_batches
            epoch_metrics = picme_utils.compute_epoch_metrics(
                args.metrics, args.task, all_labels, all_preds, all_logits, phase
            )
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            
            epoch_out = f"Epoch: {epoch}, Loss: {epoch_loss:.4f}, "
            for metric in args.metrics:
                epoch_out += f"{metric}: {epoch_metrics[f'{phase}_{metric}']:0.4f}, "
            print(epoch_out)

            wandb.log(epoch_metrics)
            epoch_objective = epoch_metrics[f"{phase}_{args.objective}"]
            
            if phase == "val":
                acc_dict[epoch] = epoch_objective

                if epoch_objective > best_metric:
                    best_metric = epoch_objective
                    
                    # Save both encoder and classifier states to ensure resuming works
                    state_dict_to_save = {
                        'classifier': classifier.state_dict(),
                        'encoder': model_contrastive.state_dict()
                    }
                    torch.save(state_dict_to_save, path + f"_{args.objective}_best.pth")
                    print(f"New best model saved with {args.objective}: {best_metric:.4f}")

                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                         print("Early stopping.")
                         return model_contrastive, classifier
                else:
                    trigger = 0

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val {args.objective}: {best_metric:4f}")

    return model_contrastive, classifier
    

        

def main(args: argparse.Namespace):
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    args.batch_size = args.batch_size[0]
    args.learning_rate = args.learning_rate[0]
    config = args 

    task = args.task
    task_name = (
        args.task if args.task not in _RENAME_MODALITY else _RENAME_MODALITY[args.task]
    )
    num_classes = _TASK_CLASSES[args.task]
    print(f"Fine-Tuning for task {task_name}!")
    print(f"Model: {args.sequential_model}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # --- Model and Dataloader Setup ---
    if args.sequential_model:
        # --- 1. Load Dataloaders ---
        if args.compute_embeddings_on_fly:
            print("MODE: Computing embeddings on-the-fly from raw data.")
            dataloaders_dict = get_raw_sequential_dataloaders(args, config, _FULL_DATA_DIR)
            
            # --- 2. Load Encoder Model for On-the-fly ---
            print(f"Loading {args.model_name} encoder for feature extraction...")
            if args.model_name == 'baseline':
                model_args = argparse.Namespace(
                    modalities=["img", "text_rad", "text_ds", "ts", "demo"],
                    task=args.task,
                    fusion_method="modality_lstm", 
                    modality_lambdas=[1, 1, 1, 1, 1]
                )
                encoder_model = models.MultiModalBaseline(
                    ts_input_dim=_MODALITY_SHAPES["ts"],
                    demo_input_dim=_MODALITY_SHAPES["demo"],
                    projection_dim=_MODALITY_SHAPES["projection"],
                    args=model_args,
                    device=device
                ).to(device)
            else:
                encoder_model = models.MultiModalContrastiveModel(
                    ts_input_dim=_MODALITY_SHAPES["ts"],
                    demo_input_dim=_MODALITY_SHAPES["demo"],
                    projection_dim=_MODALITY_SHAPES["projection"],
                    num_modalities=5,
                    training_strategy='masked_global'
                ).to(device)
                print(f"Loading encoder weights from {args.state_dict}")
                encoder_model.load_state_dict(torch.load(args.state_dict, map_location=device))
                
            encoder_model.eval() # Freeze encoder
            
        else:
            print("MODE: Using precomputed embeddings.")
            if args.task == 'phenotyping':
                dataloaders_dict = get_sequential_pheno_dataloaders(args, config, _DATA_DIR, args.task, num_classes)
            else:
                dataloaders_dict = get_sequential_mortality_dataloaders(args, config, _DATA_DIR, args.task, num_classes)

        model = BeliefUpdateTransformer(args, num_classes=num_classes)
        model.to(device)
        
        # 4. Setup Loss Function
        if args.use_evidential_head:
            print("Using Evidential Deep Learning loss.")
            criterion = evidential_loss_function 
        else:
            print("Using standard loss.")
            if args.task == "phenotyping":
                print("Using Weighted BCEWithLogitsLoss.")
                pos_counts = torch.tensor([62487,  9155,  9983, 41633, 23660, 14644, 57487,  7415, 30942, 27660, 19402, 15228, 33515, 53496, 67119, 10981, 22608, 25878, 17296,  5717, 18265, 43025, 76549, 39855, 52615]).float().to(device)
                    
                total_samples = 118785 
                neg_counts = total_samples - pos_counts
                pos_weight = neg_counts / (pos_counts + 1e-5)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
            elif args.task in ["mortality", "in-hospital-mortality"]:
                print("Using BCEWithLogitsLoss for Mortality.")
                # You can add pos_weight here if you want to handle class imbalance
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 5. Define Save Path
        llm_name_clean = args.llm_model_name.replace('/', '_')
        path = (
            f"{args.save_prefix}_BUT_{task}_"
            f"{llm_name_clean}_"
            f"lr{config.learning_rate}_r{config.lora_r}_a{config.lora_alpha}_"
            f"seed{config.seed_number}"
        )
        
        # 6. Call NEW training function
        print(f"Starting sequential training... Log path: {path}")
        train_sequential_model(
            dataloaders_dict,
            criterion,
            model, 
            config,
            path,
            args,
            device,
            encoder_model=model
        )

    else:
        # --- STATIC (Late-Fusion) PATH ---
        
        # 1. Get Static Dataloaders
        print(f"Loading static dataloaders for task: {task_name}")
        train_input_list, val_input_list, _ = data.get_dataloaders(
            args.modalities, args.batch_size, _DATA_DIR, task_name, test=False, 
            dataset_tag="missing", evaluate_with_missing=True
        )
        dataloaders_dict = {
            "train": data.MergeLoader(train_input_list),
            "val": data.MergeLoader(val_input_list),
        }

        # 2. Build (Encoder) Model
        if args.model_name == "baseline":
            print("Using fully fine-tuned baseline!")
            model_args = argparse.Namespace(
            modalities=["img", "text_rad", "text_ds", "ts", "demo"],
            task=task_name,
            fusion_method="vanilla_lstm", 
            modality_lambdas=[1, 1, 1, 1, 1]
            )
            model_contrastive = models.MultiModalBaseline(
                ts_input_dim=_MODALITY_SHAPES["ts"],
                demo_input_dim=_MODALITY_SHAPES["demo"],
                projection_dim=_MODALITY_SHAPES["projection"],
                args=model_args,
                device=device
            ).to(device)
        else:
            model_contrastive = models.MultiModalContrastiveModel(
                ts_input_dim=_MODALITY_SHAPES["ts"],
                demo_input_dim=_MODALITY_SHAPES["demo"],
                projection_dim=256,
                num_modalities=len(args.modalities),
                training_strategy='masked_global'
            ).to(device)
        
        if args.model_name != "baseline":
            model_contrastive.load_state_dict(torch.load(args.state_dict))
            model_contrastive.to(device)
            print("Successfully loaded contrastively learned model.")
             
        if args.freeze:
            print("Freezing encoder weights.")
            for param in model_contrastive.parameters():
                param.requires_grad = False
        
        # 3. Setup Loss Function
        if args.weigh_loss:
            print("Using weighted CrossEntropyLoss.")
            weights = torch.Tensor(_TASK_WEIGHTS[task_name]).to(device).float()
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            print("Using standard CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
        
        # 4. Define Save Path
        path = (
            f"{args.save_prefix}_{args.model_name}_{task_name}_"
            f"{args.fusion_method}_{'_'.join(args.modalities)}_"
            f"lr{config.learning_rate}_seed{config.seed_number}"
        )
        
        # 5. Call ORIGINAL training function
        print(f"Starting static training... Log path: {path}")
        train_model(
            dataloaders_dict,
            criterion,
            model_contrastive,
            config, # config is just args
            path,
            args,
            device
        )

    print("--- Run Complete ---")
    wandb.finish()    


if __name__ == "__main__":
    args = finetune_arg_parser()
    if not hasattr(args, 'compute_embeddings_on_fly'):
        args.compute_embeddings_on_fly = False

    if args.compute_embeddings_on_fly:
        #print("Enabling strict filtering: Only patients with ALL 5 modalities will be used.")
        args.filter_incomplete = True
    else:
        args.filter_incomplete = False
        
    main(args)

