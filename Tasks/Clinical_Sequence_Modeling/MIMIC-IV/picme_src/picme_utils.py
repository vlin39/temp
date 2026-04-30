import os
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             roc_auc_score)

import torch.nn.functional as F


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def build_optimizer(network, optimizer, learning_rate, momentum):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer == "adamW":
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer


def prep_data(modalities, batch, device, missing=False):
    modalities_data = []
    modalities_type = []
    mask_rad, mask_ds, ts_lengths = None, None, None
    labels = None
    present_mask_list = [] 
    
    if missing:
        for modality, data in zip(modalities, batch):
            present_flag = None
            #print(f"modality: {modality}")
            #print(f"data: {data}")
            #print(f"len(data): {len(data)}")
            
            # Handle different dataset return formats
            # TextDatasetMissing returns 4 items: (text, mask, labels, present)
            if len(data) == 4:  
                if modality in ["text_rad", "text_ds"]:
                    data, mask, labels, present_flag = data
                    if modality == "text_rad":
                        mask_rad = mask.to(device) if mask is not None else None
                    else:  # text_ds
                        mask_ds = mask.to(device) if mask is not None else None
            # Other Missing datasets return 3 items: (data, labels, present)
            elif len(data) == 3:  
                #print(f"len(data) == 3 for : {modality}")
                #print(f"raw data going into the ts branch in prep data: {data}")
                data, labels, present_flag = data
                if modality == "ts":
                    #print(f"modality; {modality}")
                    #print(f"data dimensions for ts: {data.shape}")
                    #print(f"label: {labels}")
                    ts_lengths = labels # Yeah I know, this is weird
                    batch_size = data.size(0) if torch.is_tensor(data) else len(data)
                    present_flag = torch.ones(batch_size, dtype=torch.bool)
    
            # Original datasets return 2 items: (data, labels)
            elif len(data) == 2:  
                data, labels = data
                # For original datasets, assume all modalities are present
                batch_size = data.size(0) if torch.is_tensor(data) else len(data)
                present_flag = torch.ones(batch_size, dtype=torch.bool)
            
            modalities_data.append(data.to(device))
            modalities_type.append(modality)
            #print(f"modality: {modality}, present_flag.shape: {present_flag.shape}")
            present_mask_list.append(present_flag.to(device))
        
        # Convert present_mask_list to tensor: (batch_size, num_modalities)
        if present_mask_list:
            present_mask_batch = torch.stack(present_mask_list, dim=1)
        else:
            batch_size = modalities_data[0].size(0) if modalities_data else 0
            present_mask_batch = torch.ones(batch_size, len(modalities), dtype=torch.bool).to(device)
    else:
        # No missing data: we are in the create_embeddings phase
        # Assuming each batch is a list of tuples (data, optional mask) for each modality
        for modality, data in zip(modalities, batch):
            if modality == "text_rad":
                data, mask_rad, labels = data
                mask_rad = mask_rad.to(device)
            elif modality == "text_ds":
                data, mask_ds, labels = data
                mask_ds = mask_ds.to(device)
            elif modality == "ts":
                data, ts_lengths, labels = data
                ts_lengths = ts_lengths.to(device)
            else:  # This includes "img" and "demo"
                data, labels = data
    
            modalities_data.append(data.to(device))
            modalities_type.append(modality)

        # Create present_mask_batch tensor: (batch_size, num_modalities)
        batch_size = modalities_data[0].size(0) if modalities_data else 0
        present_mask_batch = torch.ones(batch_size, len(modalities), dtype=torch.bool).to(device)
    
    return modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths, labels, present_mask_batch

def prep_training_data(modalities, batch, device):
    modalities_data = []
    modalities_type = []
    mask_rad, mask_ds, ts_lengths = None, None, None
    labels = None
    present_mask_list = [] 

    for modality, data in zip(modalities, batch):
        present_flag = None
        
        # Text returns 4 items: (data, mask, labels, present_flag)
        if modality in ["text_rad", "text_ds"]: 
            data, mask, labels, present_flag = data
            if modality == "text_rad":
                mask_rad = mask.to(device) if mask is not None else None
            else: # text_ds
                mask_ds = mask.to(device) if mask is not None else None

        # Image and Demo return 3 items: (data, labels, present_flag)
        elif modality in ["img", "demo"]:
            data, labels, present_flag = data
        
        # Time-series (from seq_collate) returns 2 items: (padded_data, lengths)
        else:
            assert(modality == 'ts')
            data, ts_lengths = data  # Unpacks (padded_data, lengths)
            ts_lengths = ts_lengths.to(device)
            
            # --- THIS IS THE FIX ---
            # A sample is "present" if its length is greater than 0.
            present_flag = (ts_lengths > 0).to(device)
            # --- END OF FIX ---

        modalities_data.append(data.to(device))
        modalities_type.append(modality)
        present_mask_list.append(present_flag.to(device))
            
    # Convert present_mask_list to tensor: (batch_size, num_modalities)
    if present_mask_list:
        print(f"modalities_type: {modalities_type}")
        present_mask_batch = torch.stack(present_mask_list, dim=1)
    else:
        batch_size = modalities_data[0].size(0) if modalities_data else 0
        present_mask_batch = torch.ones(batch_size, len(modalities), dtype=torch.bool).to(device)
    
    return modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths, labels, present_mask_batch
    
    
def secure_fusion(embeddings, device, fusion_method):
    safe_embeddings = []
    
    # 1. Standardize dimensions (ensure all are at least 2D)
    for embedding in embeddings:
        if embedding.dim() == 1:
            safe_embeddings.append(torch.unsqueeze(embedding, 0))
        else:
            safe_embeddings.append(embedding)

    # 2. ALIGN BATCH SIZES (The Fix)
    # Check the batch dimension (dim 0) of all embeddings
    batch_sizes = [e.shape[0] for e in safe_embeddings]
    min_batch_size = min(batch_sizes)

    # If there is a mismatch (e.g., [32, 32, 32, 17, 32]), slice everyone to the minimum
    if any(bs != min_batch_size for bs in batch_sizes):
        print(f"Warning: Batch size mismatch detected {batch_sizes}. Truncating to {min_batch_size}.")
        safe_embeddings = [e[:min_batch_size] for e in safe_embeddings]

    # 3. Perform Fusion
    if fusion_method == "concatenation":
        # Shapes are now all [min_batch, dim], safe to cat
        fused_embeddings = torch.cat(safe_embeddings, dim=1).to(device)
        
    elif fusion_method in ["vanilla_lstm", "modality_lstm"]:
        expanded_embeddings = []
        for embedding in safe_embeddings:
            expanded_embeddings.append(embedding[:, None, :])
        # Shapes are now all [min_batch, 1, dim], safe to cat
        fused_embeddings = torch.cat(expanded_embeddings, dim=1).to(device)

    return fused_embeddings

def extract_labels(raw_labels, task):
    if task == "mortality":
        return raw_labels.long()
    elif task == "phenotyping":
        return raw_labels


def predict(outputs, task):
    if task == "mortality":
        _, preds = torch.max(outputs, axis=1)
        return preds.cpu().detach().numpy()
    elif task == "phenotyping":
        return (outputs.sigmoid() > 0.5).cpu().detach().numpy()
    else:
        return ValueError("Invalid task given.")


def compute_epoch_metrics(metrics, task, all_labels, all_preds, all_logits, phase, 
                          modality_counts=None, present_masks=None):
    """
    Compute evaluation metrics, optionally stratified by modality count.
    Includes AUPRC Baseline (Prevalence) calculation.
    """
    epoch_metrics = dict()
    
    all_labels, all_preds, all_logits = (
        np.asarray(all_labels),
        np.asarray(all_preds),
        np.asarray(all_logits),
    )

    if task == "mortality":
        # Ensure 1D array for labels if necessary
        if all_labels.ndim > 1: all_labels = all_labels.squeeze()
        # For binary tasks, usually we take the positive class probability
        if all_logits.ndim > 1 and all_logits.shape[1] == 2:
            all_logits = all_logits[:, 1]
    
    # --- HELPER: Calculate AUPRC Baseline (Prevalence) ---
    def get_auprc_baseline(labels, task_name):
        if len(labels) == 0: return 0.0
        
        if task_name == "mortality":
            # Binary prevalence: Count(Positives) / Total
            return np.mean(labels)
        else:
            # Multi-label prevalence (Macro Average)
            # 1. Calculate prevalence for each of the 25 classes
            class_prevalence = np.mean(labels, axis=0)
            # 2. Average them to match the "macro" averaging of the metric
            return np.mean(class_prevalence)

    # --- 1. Compute Overall Metrics ---
    for metric in metrics:
        if metric == "hamming" and task != "mortality":
            epoch_metrics[f"{phase}_hamming"] = hamming_loss(all_labels, all_preds)
            
        elif metric == "auprc":
            epoch_metrics[f"{phase}_auprc"] = average_precision_score(
                all_labels, all_logits, average="macro"
            )
            # <--- NEW: Calculate Global Baseline
            epoch_metrics[f"{phase}_auprc_baseline"] = get_auprc_baseline(all_labels, task)
            
        elif metric == "auroc":
            try:
                epoch_metrics[f"{phase}_auroc"] = roc_auc_score(
                    all_labels, all_logits, average="macro"
                )
            except ValueError:
                 epoch_metrics[f"{phase}_auroc"] = np.nan
                 
        elif metric == "f1":
            epoch_metrics[f"{phase}_f1"] = f1_score(
                all_labels, all_preds, average="weighted"
            )

    # Accuracy
    if task == "phenotyping":
        epoch_metrics[f"{phase}_accuracy"] = np.mean(
            (np.sum((all_labels == all_preds), axis=1) / 25)
        )
    elif task == "mortality":
        epoch_metrics[f"{phase}_accuracy"] = np.mean(all_preds == all_labels)
    
    # --- 2. Compute Stratified Metrics (by Modality Count) ---
    if modality_counts is not None:
        modality_counts = np.asarray(modality_counts)
        unique_counts = np.unique(modality_counts)
        
        for count in unique_counts:
            mask = modality_counts == count
            if np.sum(mask) > 0:  # Only compute if we have samples
                count_labels = all_labels[mask]
                count_preds = all_preds[mask]
                count_logits = all_logits[mask]
                
                # Accuracy
                if task == "phenotyping":
                    acc = np.mean((np.sum((count_labels == count_preds), axis=1) / 25))
                else:
                    acc = np.mean(count_preds == count_labels)
                epoch_metrics[f"{phase}_accuracy_{count}mod"] = acc
                
                # Other Metrics
                for metric in metrics:
                    suffix = f"{phase}_{metric}_{count}mod"
                    
                    if metric == "hamming" and task != "mortality":
                        epoch_metrics[suffix] = hamming_loss(count_labels, count_preds)
                        
                    elif metric == "auprc":
                        try:
                            epoch_metrics[suffix] = average_precision_score(
                                count_labels, count_logits, average="macro"
                            )
                        except:
                            epoch_metrics[suffix] = np.nan
                        
                        # <--- NEW: Calculate Baseline for this specific subset
                        # This is crucial because 1-modality patients might have different 
                        # disease rates than 5-modality patients
                        baseline_suffix = f"{phase}_auprc_baseline_{count}mod"
                        epoch_metrics[baseline_suffix] = get_auprc_baseline(count_labels, task)
                            
                    elif metric == "auroc":
                        try:
                            epoch_metrics[suffix] = roc_auc_score(
                                count_labels, count_logits, average="macro"
                            )
                        except:
                            epoch_metrics[suffix] = np.nan
                            
                    elif metric == "f1":
                        try:
                            epoch_metrics[suffix] = f1_score(
                                count_labels, count_preds, average="weighted"
                            )
                        except:
                            epoch_metrics[suffix] = np.nan
                
                epoch_metrics[f"{phase}_samples_{count}mod"] = np.sum(mask)

    return epoch_metrics
    

def compute_accuracy(image_output, text_output):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Compute the similarity score matrix
    scores = torch.matmul(image_output, text_output.t())

    # Get the indices that would sort the scores
    sorted_indices = torch.argsort(scores, dim=1, descending=True)

    # Get the rank of the correct text for each image
    targets = torch.arange(image_output.size(0)).to(device)
    ranks = (sorted_indices == targets.view(-1, 1)).nonzero()[:, 1]

    top1 = torch.sum(ranks < 1).item() / image_output.size(0)
    top5 = torch.sum(ranks < 5).item() / image_output.size(0)

    return top1, top5


def compute_multimodal_accuracy(*embeddings):
    n_modalities = len(embeddings)
    batch_size = embeddings[0].size(0)

    # Create a similarity score matrix for every pair of modalities
    score_matrices = [
        torch.matmul(embeddings[i], embeddings[j].t())
        for i in range(n_modalities)
        for j in range(n_modalities)
        if i != j
    ]

    targets = torch.arange(batch_size).to(embeddings[0].device)

    top1s, top5s, top10s = [], [], []  # Add list for top10s
    for scores in score_matrices:
        # Get the indices that would sort the scores
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        # Get the rank of the correct pairing
        ranks = (sorted_indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1]

        top1 = torch.sum(ranks < 1).item() / batch_size
        top5 = torch.sum(ranks < 5).item() / batch_size
        top10 = torch.sum(ranks < 10).item() / batch_size  # Compute top10

        top1s.append(top1)
        top5s.append(top5)
        top10s.append(top10)  # Append top10 result to list

    avg_top1 = sum(top1s) / len(top1s)
    avg_top5 = sum(top5s) / len(top5s)
    avg_top10 = sum(top10s) / len(top10s)  # Calculate average top10

    return avg_top1, avg_top5, avg_top10  # Return top10 as well

def compute_masked_retrieval(features_tensor, present_mask, k_vals=[1, 5, 10], min_samples=16):
    """
    Computes retrieval metrics ONLY on valid intersecting subsets of patients
    where the intersection size is large enough to be statistically meaningful.
    """
    device = features_tensor.device
    B, M, D = features_tensor.shape
    
    metrics_accumulator = {f"R@{k}": [] for k in k_vals}
    valid_pair_count = 0
    
    # Iterate over every unique pair of modalities
    for i in range(M):
        for j in range(M):
            if i == j: 
                continue
                
            # 1. Find intersection: Patients having BOTH Modality i and Modality j
            valid_mask = (present_mask[:, i] == 1) & (present_mask[:, j] == 1)
            num_valid = valid_mask.sum().item()
            
            # --- CRITICAL FIX: Threshold for statistical significance ---
            if num_valid < min_samples:
                continue
                
            valid_pair_count += 1
            
            # 2. Extract Valid Embeddings
            features_A = features_tensor[valid_mask, i, :]
            features_B = features_tensor[valid_mask, j, :]
            
            # 3. Compute Similarity Matrix [N_sub, N_sub]
            logits = torch.matmul(features_A, features_B.T)
            
            # 4. Compute Targets
            batch_size_sub = logits.shape[0]
            targets = torch.arange(batch_size_sub, device=device)
            
            # 5. Sort
            _, sorted_indices = logits.sort(dim=1, descending=True)
            hits = (sorted_indices == targets.view(-1, 1))
            
            # 6. Calculate R@K
            for k in k_vals:
                # Cap k at the number of available samples to prevent index errors
                current_k = min(k, batch_size_sub)
                recall = hits[:, :current_k].any(dim=1).float().mean().item()
                metrics_accumulator[f"R@{k}"].append(recall)

    # Return 0 if no pairs had enough overlap to compute metrics
    if valid_pair_count == 0:
        return 0.0, 0.0, 0.0
        
    final_metrics = {}
    for k in k_vals:
        final_metrics[k] = sum(metrics_accumulator[f"R@{k}"]) / valid_pair_count
        
    return final_metrics[1], final_metrics[5], final_metrics[10]
    

def impute_embeddings(model, modalities_data, modalities_type, present_mask, embedding_bank, device, 
                      mask_rad=None, mask_ds=None, ts_lengths=None):
    """
    Encodes raw modalities data and imputes missing modality embeddings using nearest-neighbor approach.
    """
    batch_size = present_mask.shape[0]
    
    # --- Step 1: Encode all available modalities ---
    encoded_embeddings = []
    for i, (modality_data, modality_name) in enumerate(zip(modalities_data, modalities_type)):
        if present_mask[:, i].all():  # If all samples have this modality
            if modality_name == "ts":
                emb = model.encode_modality(
                    modality_data, "ts", att_mask=None, lengths=ts_lengths
                )
            elif modality_name == "text_rad":
                emb = model.encode_modality(
                    modality_data, "text_rad", att_mask=mask_rad, lengths=None
                )
            elif modality_name == "text_ds":
                emb = model.encode_modality(
                    modality_data, "text_ds", att_mask=mask_ds, lengths=None
                )
            else:
                emb = model.encode_modality(
                    modality_data, modality_name, att_mask=None, lengths=None
                )
        else:
            # Handle mixed presence - encode sample by sample
            # Note: We need the dim from the bank to initialize zeros
            emb_dim = embedding_bank[modality_name].shape[1]
            emb = torch.zeros(batch_size, emb_dim, device=device)
            
            for j in range(batch_size):
                if present_mask[j, i]:  # If this sample has the modality
                    single_data = modality_data[j].unsqueeze(0) if modality_data.dim() > 1 else modality_data[j:j+1]
                    
                    if modality_name == "ts":
                        single_lengths = ts_lengths[j:j+1] if ts_lengths is not None else None
                        single_emb = model.encode_modality(
                            single_data, "ts", att_mask=None, lengths=single_lengths
                        )
                    elif modality_name == "text_rad":
                        single_mask = mask_rad[j:j+1] if mask_rad is not None else None
                        single_emb = model.encode_modality(
                            single_data, "text_rad", att_mask=single_mask, lengths=None
                        )
                    elif modality_name == "text_ds":
                        single_mask = mask_ds[j:j+1] if mask_ds is not None else None
                        single_emb = model.encode_modality(
                            single_data, "text_ds", att_mask=single_mask, lengths=None
                        )
                    else:
                        single_emb = model.encode_modality(
                            single_data, modality_name, att_mask=None, lengths=None
                        )
                    
                    emb[j] = single_emb.squeeze(0)
        
        encoded_embeddings.append(emb)
    
    # --- Step 2: Prepare Embedding Bank for Similarity ---
    # Pre-normalize the bank once to save computation
    embedding_bank_norm = {
        modality: F.normalize(tensor.to(device), dim=-1)
        for modality, tensor in embedding_bank.items()
    }
    
    # Keep raw bank on device for final retrieval
    embedding_bank_device = {
        modality: tensor.to(device)
        for modality, tensor in embedding_bank.items()
    }
    
    modality_names = list(embedding_bank.keys())
    num_bank_samples = embedding_bank[modality_names[0]].shape[0]
    
    # --- Step 3: Impute Missing Embeddings ---
    imputed_embeddings = list(encoded_embeddings)
    
    for i in range(batch_size):
        if not present_mask[i].all():  # If any modality is missing for this sample
            
            # Identify which modalities this patient HAS
            available_indices = present_mask[i].nonzero().squeeze(-1)
            
            # Initialize total scores
            total_similarity_scores = torch.zeros(num_bank_samples, device=device)
            
            # Accumulate similarity scores from available modalities
            for mod_idx in available_indices:
                modality_name = modality_names[mod_idx]
                
                # [1, 256]
                sample_emb = imputed_embeddings[mod_idx][i].unsqueeze(0)
                sample_emb_norm = F.normalize(sample_emb, dim=-1)
                
                # [N, 256]
                bank_embs_norm = embedding_bank_norm[modality_name]
                if modality_name == "ts":
                    bank_embs_norm = bank_embs_norm.squeeze(1)
                #print(f"modality_name: {modality_name}")
                #print(f"bank_embs_norm: {bank_embs_norm.shape}")
                
                # --- FIX: Matrix Multiplication ---
                # [1, 256] @ [256, N] -> [1, N]
                # We transpose the bank to align dimensions
                sims = torch.mm(sample_emb_norm, bank_embs_norm.t()).squeeze(0)
                
                total_similarity_scores += sims
            
            # Find best neighbor (index in the bank)
            best_neighbor_idx = torch.argmax(total_similarity_scores)
            
            # Identify which modalities this patient is MISSING
            missing_indices = (~present_mask[i]).nonzero().squeeze(-1)
            
            # Impute the missing modalities using the best neighbor's data
            for mod_idx in missing_indices:
                modality_name = modality_names[mod_idx]
                imputed_embeddings[mod_idx][i] = embedding_bank_device[modality_name][best_neighbor_idx]

    return imputed_embeddings
