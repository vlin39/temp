import os
import sys
import pickle
import random
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
import webdataset 
import json 
import pickle 
import numpy as np 
from braceexpand import braceexpand
from torchvision import transforms

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Map string types to Integers
_MODALITY_ID_MAP = {
    '[PREDICT]': 0,
    'demo': 1,
    'ehr_bin': 2,   # Time-series
    'cxr': 3,       # Image
    'rad_note': 4,  # Text 1
    'ds_note': 5,   # Text 2
    # Add others if necessary
}

_MORTALITY_CLASS_WEIGHTS = [8377, 1357] # Will need to change this
_PHENO_CLASS_WEIGHTS = [3117,638,899,3726,2453,1676,2178,1199,2914,3675,1369,2006,4780,4909,4320,814,2562,1488,1117,587,780,1407,1840,1826,1405]        
def augment_timeseries_tensor(ts_tensor, sigma=0.03, mask_prob=0.15):
    """
    Applies Jittering and Time-step Masking to a Time-Series Tensor.
    Args:
        ts_tensor (torch.Tensor): Shape [Seq_Len, Features]
    """
    # 1. Jitter (Add Gaussian Noise)
    if random.random() < 0.5:
        noise = torch.normal(mean=0, std=sigma, size=ts_tensor.shape)
        ts_tensor = ts_tensor + noise

    # 2. Random Time-Step Masking (Zero out entire rows)
    if random.random() < 0.5:
        seq_len = ts_tensor.shape[0]
        # Calculate how many steps to mask
        num_mask = int(seq_len * mask_prob)
        if num_mask > 0:
            mask_indices = torch.randperm(seq_len)[:num_mask]
            ts_tensor[mask_indices, :] = 0.0

    return ts_tensor

def augment_text_ids(input_ids, mask_token_id, vocab_size, mask_prob=0.15):
    """
    Applies BERT-style masking to token IDs.
    1. 80% chance -> replace with [MASK]
    2. 10% chance -> replace with random word
    3. 10% chance -> keep original (identity)
    """
    # Create a mask for tokens to augment (excluding special tokens usually 0, 101, 102)
    # Assuming 101=CLS, 102=SEP, 0=PAD. Adjust if ClinicalBERT differs.
    probability_matrix = torch.full(input_ids.shape, mask_prob)
    special_tokens_mask = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% replace with random word
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(input_ids), vocab_size, input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The remaining 10% are kept original (Identity)
    return input_ids

def combine_sample(sample):
    """
    Combines the decoded .json (events) and .npy (labels) parts
    from the RAW WebDataset shards.
    """
    # sample['json'] is a dict: {'stay_id': ..., 'events': ...}
    # sample['npy'] is a numpy array (the labels)
    
    data_dict = sample['json']
    # Ensure we cast to tensor here so the collate function receives tensors
    data_dict['labels'] = torch.tensor(sample['npy']) 
    
    return data_dict

def map_precomputed_sample(sample):
    """
    Decodes the .pkl and .npy files from the PRECOMPUTED shards.
    """
    events = sample['events.pkl'] 
    labels = sample['labels.npy']
    
    return {
        'stay_id': int(sample['__key__']),
        'events': events, 
        'labels': torch.tensor(labels) 
    }

# --- LOADERS ---

def get_raw_sequential_dataloaders(args, config, data_dir):
    """
    Loads the RAW WebDataset (JSONs with paths/text + NPY labels) 
    for on-the-fly encoding.
    """
    # --- FIX: Dynamic Directory Selection based on Task ---
    if args.task == 'phenotyping':
        subdir = "sequential_pheno"
        # Phenotyping usually has 1420 train shards / 310 val shards
        train_len = 1420
        val_len = 310
        train_shards = 141
        val_shards = 30
    else:
        # Assuming your raw mortality data is in "picme_sequential_mortality" 
        # (based on your precompute script)
        subdir = "picme_sequential_mortality" 
        # You might need to adjust these lengths/shards for your specific mortality dataset
        # Based on your precomputed loader, it seems smaller:
        train_len = 910  # Example from your mortality loader
        val_len = 190    # Example from your mortality loader
        train_shards = 90
        val_shards = 18

    sequence_data_dir = os.path.join(data_dir, subdir) 

    # Define shard ranges
    train_shards_list = "{" + f"000000..{train_shards:06d}" + "}" 
    val_shards_list = "{" + f"000000..{val_shards:06d}" + "}"

    train_url_pattern = os.path.join(sequence_data_dir, f"train_{train_shards_list}.tar")
    val_url_pattern = os.path.join(sequence_data_dir, f"val_{val_shards_list}.tar")

    train_urls = list(braceexpand(train_url_pattern))
    val_urls = list(braceexpand(val_url_pattern))

    print(f"[Raw Loader] Loading from: {subdir}")
    print(f"[Raw Loader] Found {len(train_urls)} train shards.")
    print(f"[Raw Loader] Found {len(val_urls)} val shards.")

    train_dataset = (
        webdataset.WebDataset(train_urls, nodesplitter=webdataset.split_by_worker)
        .shuffle(1000)
        .decode()
        .map(combine_sample) 
        .with_length(train_len)
    )

    val_dataset = (
        webdataset.WebDataset(val_urls, nodesplitter=webdataset.split_by_worker)
        .decode()
        .map(combine_sample) 
        .with_length(val_len)
    )

    def raw_collate_fn(batch):
        return batch

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size, 
        collate_fn=raw_collate_fn, 
        num_workers=4, # Increased for better throughput
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=raw_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    return {"train": train_loader, "val": val_loader}
    
    
def get_sequential_pheno_dataloaders(args, config, data_dir, task, num_classes):
    # This path is correct
    #sequence_data_dir = os.path.join(data_dir, "sequential_pheno_baseline_precomputed") 
    sequence_data_dir = os.path.join(data_dir, "picme_sequential_pheno_precomputed") 
    #sequence_data_dir = os.path.join(data_dir, "og_picme_sequential_pheno_precomputed") 
    #sequence_data_dir = os.path.join(data_dir, "picme_sequential_mortality_precomputed") 
    
    # Train: 1420 / 50 = 29 shards (0-28)
    # Val:   310 / 50 = 7 shards (0-6)
    train_shards_list = "{" + f"000000..{28:06d}" + "}"
    val_shards_list = "{" + f"000000..{6:06d}" + "}"
    # train_shards_list = "{" + f"000000..{90:06d}" + "}"
    # val_shards_list = "{" + f"000000..{18:06d}" + "}"

    train_url_pattern = os.path.join(sequence_data_dir, f"train_{train_shards_list}.tar")
    val_url_pattern = os.path.join(sequence_data_dir, f"val_{val_shards_list}.tar")

    train_urls = list(braceexpand(train_url_pattern))
    val_urls = list(braceexpand(val_url_pattern))

    print(f"Found {len(train_urls)} train shards.")
    print(f"Found {len(val_urls)} val shards.")

    num_train = 1420
    num_val = 310
    #num_train = 910
    #num_val = 190

    train_dataset = (
        webdataset.WebDataset(train_urls, nodesplitter=webdataset.split_by_worker)
        .shuffle(1000)
        .decode("l")
        .map(map_precomputed_sample) 
        .with_length(num_train)
    )

    val_dataset = (
        webdataset.WebDataset(val_urls, nodesplitter=webdataset.split_by_worker)
        .decode("l")
        .map(map_precomputed_sample)  
        .with_length(num_val)
    )

    # --- Create DataLoaders ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=sequential_collate_fn, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=sequential_collate_fn,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    return {"train": train_loader, "val": val_loader}


def get_sequential_mortality_dataloaders(args, config, data_dir, task, num_classes):
    # This path is correct
    #sequence_data_dir = os.path.join(data_dir, "sequential_pheno_baseline_precomputed") 
    #sequence_data_dir = os.path.join(data_dir, "picme_sequential_pheno_precomputed") 
    #sequence_data_dir = os.path.join(data_dir, "og_picme_sequential_pheno_precomputed") 
    sequence_data_dir = os.path.join(data_dir, "picme_sequential_mortality_precomputed") 
    #sequence_data_dir = os.path.join(data_dir, "og_picme_sequential_mortality_precomputed") 
    
    # Train: 1420 / 50 = 29 shards (0-28)
    # Val:   310 / 50 = 7 shards (0-6)
    #train_shards_list = "{" + f"000000..{28:06d}" + "}"
    #val_shards_list = "{" + f"000000..{6:06d}" + "}"
    train_shards_list = "{" + f"000000..{90:06d}" + "}"
    val_shards_list = "{" + f"000000..{18:06d}" + "}"

    train_url_pattern = os.path.join(sequence_data_dir, f"train_{train_shards_list}.tar")
    val_url_pattern = os.path.join(sequence_data_dir, f"val_{val_shards_list}.tar")

    train_urls = list(braceexpand(train_url_pattern))
    val_urls = list(braceexpand(val_url_pattern))

    print(f"Found {len(train_urls)} train shards.")
    print(f"Found {len(val_urls)} val shards.")

    #num_train = 1420
    #num_val = 310
    num_train = 910
    num_val = 190

    train_dataset = (
        webdataset.WebDataset(train_urls, nodesplitter=webdataset.split_by_worker)
        .shuffle(1000)
        .decode("l")
        .map(map_precomputed_sample) 
        .with_length(num_train)
    )

    val_dataset = (
        webdataset.WebDataset(val_urls, nodesplitter=webdataset.split_by_worker)
        .decode("l")
        .map(map_precomputed_sample)  
        .with_length(num_val)
    )

    # --- Create DataLoaders ---
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=sequential_collate_mortality_fn, 
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=sequential_collate_mortality_fn,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    return {"train": train_loader, "val": val_loader}

def sequential_collate_mortality_fn(batch):
    """
    Pads sequences for the BUT model (Mortality Task).
    Fixes the IndexError by ensuring scalar labels become 1D tensors.
    """
    
    sequences_of_embeddings = []
    sequences_of_labels = []
    sequences_of_predict_masks = []
    final_labels_list = []
    sequences_of_modality_ids = []
    
    EMBEDDING_DIM = None
    DUMMY_PREDICT_TOKEN = None

    for item in batch:
        patient_events = item['events']
        
        # --- FIX START: Handle Scalar Labels for Mortality ---
        # Mortality labels are often saved as scalar floats (0-d).
        # We must convert them to 1-d tensors (shape [1]) to access .shape[0].
        final_labels = torch.tensor(item['labels'])
        if final_labels.ndim == 0:
            final_labels = final_labels.unsqueeze(0) # Shape becomes [1]
        # --- FIX END ---

        final_labels_list.append(final_labels)
        
        patient_label_vector = final_labels
        
        # Now final_labels.shape[0] will correctly return 1 instead of crashing
        masked_label_vector = torch.full(
            (final_labels.shape[0],), 
            -100, 
            dtype=patient_label_vector.dtype
        )

        current_sequence_embeddings = []
        current_sequence_labels = []
        current_predict_mask = []
        current_sequence_mod_ids = []
        
        if not patient_events:
            continue
            
        # --- Discover EMBEDDING_DIM from the first real data event ---
        if EMBEDDING_DIM is None:
            for event in patient_events:
                if event['type'] != '[PREDICT]':
                    EMBEDDING_DIM = event['embedding'].shape[-1]
                    DUMMY_PREDICT_TOKEN = torch.zeros(EMBEDDING_DIM)
                    break
        
        # If entire patient has no data events, skip
        if EMBEDDING_DIM is None:
            continue

        for event in patient_events:
            if event['type'] == '[PREDICT]':
                current_sequence_embeddings.append(DUMMY_PREDICT_TOKEN)
                current_sequence_labels.append(patient_label_vector)
                current_predict_mask.append(True)
                current_sequence_mod_ids.append(torch.tensor(0, dtype=torch.long))
            else:
                embedding = torch.from_numpy(event['embedding']).squeeze(0)
                mod_id = _MODALITY_ID_MAP.get(event['type'], 0)
                current_sequence_mod_ids.append(torch.tensor(mod_id, dtype=torch.long))
                
                if embedding.shape[0] != EMBEDDING_DIM:
                    print(f"WARNING: Skipping event. Mismatched dim. Got {embedding.shape[0]}, expected {EMBEDDING_DIM}")
                    continue
                    
                current_sequence_embeddings.append(embedding)
                current_sequence_labels.append(masked_label_vector)
                current_predict_mask.append(False)

        sequences_of_modality_ids.append(torch.stack(current_sequence_mod_ids))

        if not current_sequence_embeddings:
            continue

        sequences_of_embeddings.append(torch.stack(current_sequence_embeddings))
        sequences_of_labels.append(torch.stack(current_sequence_labels))
        sequences_of_predict_masks.append(torch.tensor(current_predict_mask))

    # Handle edge case where ENTIRE batch was empty/skipped
    if EMBEDDING_DIM is None:
        return {
            'padded_embeddings': torch.empty(0),
            'padded_labels': torch.empty(0),
            'predict_mask': torch.empty(0),
            'attention_mask': torch.empty(0),
            'final_labels_batch': torch.empty(0),
            'modality_ids': torch.empty(0)
        }
        
    # --- Padding ---
    padded_embeddings = rnn_utils.pad_sequence(
        sequences_of_embeddings, batch_first=True, padding_value=0.0
    )
    padded_labels = rnn_utils.pad_sequence(
        sequences_of_labels, batch_first=True, padding_value=-100
    )
    predict_mask = rnn_utils.pad_sequence(
        sequences_of_predict_masks, batch_first=True, padding_value=False
    )
    padded_modality_ids = rnn_utils.pad_sequence(
        sequences_of_modality_ids, batch_first=True, padding_value=0
    )

    # --- Create Attention Mask ---
    lengths = [len(seq) for seq in sequences_of_embeddings]
    max_seq_len = padded_embeddings.shape[1]
    
    attention_mask = torch.zeros(len(lengths), max_seq_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    return {
        'padded_embeddings': padded_embeddings,
        'padded_labels': padded_labels,
        'predict_mask': predict_mask,
        'attention_mask': attention_mask,
        'final_labels_batch': torch.stack(final_labels_list),
        'modality_ids': padded_modality_ids 
    }
    

    
def sequential_collate_fn(batch):
    """
    Pads sequences for the BUT model.
    This runs in the dataloader workers.
    
    Returns:
    - padded_embeddings: [B, S, D], padded with 0.0
    - padded_labels: [B, S, C], padded with -100
    - predict_mask: [B, S], boolean mask, True where [PREDICT] should be
    - attention_mask: [B, S], 0/1 mask for transformer
    - final_labels_batch: [B, C], the static labels for the batch
    """
    
    sequences_of_embeddings = []
    sequences_of_labels = []
    sequences_of_predict_masks = []
    final_labels_list = []
    sequences_of_modality_ids = []
    
    # --- FIX: Initialize Dim and Token as None ---
    EMBEDDING_DIM = None
    DUMMY_PREDICT_TOKEN = None
    # --- END FIX ---

    for item in batch:
        patient_events = item['events']
        final_labels = torch.tensor(item['labels'])
        final_labels_list.append(final_labels)
        
        patient_label_vector = final_labels
        masked_label_vector = torch.full((final_labels.shape[0],), -100, 
                                         dtype=patient_label_vector.dtype)

        current_sequence_embeddings = []
        current_sequence_labels = []
        current_predict_mask = []
        current_sequence_mod_ids = []
        
        if not patient_events:
            continue
            
        # --- FIX: Discover EMBEDDING_DIM from the first real event in the batch ---
        if EMBEDDING_DIM is None:
            for event in patient_events:
                if event['type'] != '[PREDICT]':
                    # event['embedding'] shape is (1, E) or (E,). Use [1] or [-1]
                    EMBEDDING_DIM = event['embedding'].shape[-1] # Get 256
                    DUMMY_PREDICT_TOKEN = torch.zeros(EMBEDDING_DIM)
                    break
        
        # If this patient had no data (only [PREDICT]) and we don't know the dim yet,
        # we must skip them and hope the next patient has data.
        if EMBEDDING_DIM is None:
            continue
        # --- END FIX ---

        for event in patient_events:
            if event['type'] == '[PREDICT]':
                # --- FIX: Append the correctly-shaped dummy token ---
                current_sequence_embeddings.append(DUMMY_PREDICT_TOKEN)
                current_sequence_labels.append(patient_label_vector)
                current_predict_mask.append(True)
                current_sequence_mod_ids.append(torch.tensor(0, dtype=torch.long))
            else:
                embedding = torch.from_numpy(event['embedding']).squeeze(0)
                mod_id = _MODALITY_ID_MAP.get(event['type'], 0)
                current_sequence_mod_ids.append(torch.tensor(mod_id, dtype=torch.long))
                
                # Safety check
                if embedding.shape[0] != EMBEDDING_DIM:
                    print(f"WARNING: Skipping event. Mismatched dim. Got {embedding.shape[0]}, expected {EMBEDDING_DIM}")
                    continue
                    
                current_sequence_embeddings.append(embedding)
                current_sequence_labels.append(masked_label_vector)
                current_predict_mask.append(False)

        sequences_of_modality_ids.append(torch.stack(current_sequence_mod_ids))

        if not current_sequence_embeddings:
            continue

        # This stack will now work, as all tensors are shape [256]
        sequences_of_embeddings.append(torch.stack(current_sequence_embeddings))
        sequences_of_labels.append(torch.stack(current_sequence_labels))
        sequences_of_predict_masks.append(torch.tensor(current_predict_mask))

    # --- FIX: Handle edge case where ENTIRE batch had no data events ---
    if EMBEDDING_DIM is None:
        print("WARNING: Entire batch contained no data events. Returning empty.")
        return {
            'padded_embeddings': torch.empty(0),
            'padded_labels': torch.empty(0),
            'predict_mask': torch.empty(0),
            'attention_mask': torch.empty(0),
            'final_labels_batch': torch.empty(0)
        }
    # --- END FIX ---
        
    # --- Padding ---
    padded_embeddings = rnn_utils.pad_sequence(
        sequences_of_embeddings, batch_first=True, padding_value=0.0
    )
    padded_labels = rnn_utils.pad_sequence(
        sequences_of_labels, batch_first=True, padding_value=-100
    )
    predict_mask = rnn_utils.pad_sequence(
        sequences_of_predict_masks, batch_first=True, padding_value=False # Pad with False
    )
    
    padded_modality_ids = rnn_utils.pad_sequence(
        sequences_of_modality_ids, batch_first=True, padding_value=0
    )

    # --- Create Attention Mask ---
    lengths = [len(seq) for seq in sequences_of_embeddings]
    max_seq_len = padded_embeddings.shape[1]
    
    # Use len(lengths) instead of len(batch) in case some items were skipped
    attention_mask = torch.zeros(len(lengths), max_seq_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    return {
        'padded_embeddings': padded_embeddings,
        'padded_labels': padded_labels,
        'predict_mask': predict_mask,
        'attention_mask': attention_mask,
        'final_labels_batch': torch.stack(final_labels_list),
        'modality_ids': padded_modality_ids 
    }
    
class TextDataset(Dataset):
    def __init__(self, dataframe, target_cols, tokenizer, type_text, max_token_len=512):
        self.tokenizer = tokenizer
        self.type_text = type_text
        self.dataframe = dataframe
        self.labels = dataframe[target_cols].values
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]

        text_data = data_row[self.type_text]

        encoding = self.tokenizer(
            text_data,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text = encoding["input_ids"].flatten()
        att_mask = encoding["attention_mask"].flatten()
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, att_mask, labels

# class TextDatasetMissing(Dataset):
#     def __init__(self, dataframe, target_cols, tokenizer, type_text, max_token_len=512):
#         self.tokenizer = tokenizer
#         self.type_text = type_text
#         #self.dataframe = dataframe
#         self.target_cols = target_cols
#         self.labels = dataframe[target_cols].values if target_cols[0] in dataframe.columns else np.zeros((len(dataframe), len(target_cols)))
#         self.max_token_len = max_token_len
#         self.text_data_col = dataframe[type_text].values
#         self.labels = dataframe[target_cols].values if target_cols[0] in dataframe.columns else np.zeros((len(dataframe), len(target_cols)))
#         self._len = len(dataframe)

#     def __len__(self):
#         return self._len

#     def __getitem__(self, idx):
#         text_data = self.text_data_col[idx]
#         labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
#         if pd.isna(text_data):
#             # Return placeholders consistent with base class structure + presence flag
#             return (torch.zeros(self.max_token_len, dtype=torch.long), 
#                     torch.zeros(self.max_token_len, dtype=torch.long), 
#                     labels, 
#                     False)  

#         encoding = self.tokenizer.encode_plus(
#             text_data,
#             add_special_tokens=True,
#             max_length=self.max_token_len,
#             return_token_type_ids=False,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#         return (encoding['input_ids'].flatten(), 
#                 encoding['attention_mask'].flatten(), 
#                 labels, 
#                 True)  



class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data, is_train=False):
        self.time_series_data = time_series_data
        self.is_train = is_train

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        time_series = self.time_series_data[idx]
        ts = torch.tensor(time_series, dtype=torch.float)
        
        # Apply Augmentation only during training
        if self.is_train:
            ts = augment_timeseries_tensor(ts)
            
        return ts

class DemographicsDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.features = (
            dataframe.values
        )  # .drop(labels, axis=1).values  # Assuming labels are not part of your features dataframe
        self.labels = labels  # Directly using the passed list of labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        demo = torch.tensor(self.features[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return demo, labels


# class DemographicsDatasetMissing(Dataset):
#     def __init__(self, dataframe, labels):
#         self.features = dataframe.values
#         self.labels = labels
#         self.output_dim = dataframe.shape[1]

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         feature_row = self.features[idx]
#         labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        
#         if np.isnan(feature_row).any():
#             return (torch.zeros(self.output_dim, dtype=torch.float), 
#                     labels_tensor, 
#                     False)
        
#         return (torch.tensor(feature_row, dtype=torch.float), 
#                 labels_tensor, 
#                 True)




class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, target_cols, img_col, transform=None):
        self.dataframe = dataframe
        self.img_col = img_col
        self.labels = dataframe[target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        img_path = self.dataframe.iloc[idx][self.img_col]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        return image, labels

# class MedicalImageDatasetMissing(Dataset):
#     def __init__(self, dataframe, target_cols, img_col, transform=None):
#         self.img_paths = dataframe[img_col].values # Store only this column
#         self.labels = dataframe[target_cols].values if target_cols[0] in dataframe.columns else np.zeros((len(dataframe), len(target_cols)))
#         self._len = len(dataframe)
#         self.img_col = img_col
#         self.labels = dataframe[target_cols].values if target_cols[0] in dataframe.columns else np.zeros((len(dataframe), len(target_cols)))
#         self.transform = transform
#         # Get output shape from transform or use default
#         self.placeholder_shape = (3, 224, 224)  # Should match transform output

#     def __len__(self):
#         return self._len

#     def __getitem__(self, idx):
#         labels = torch.tensor(self.labels[idx], dtype=torch.float)
#         img_path = self.img_paths[idx]
        
#         if pd.isna(img_path):
#             return (torch.zeros(self.placeholder_shape), 
#                     labels, 
#                     False)
        
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return (image, 
#                 labels, 
#                 True)


# --- Replace these classes in picme_src/data.py ---

class TextDatasetMissing(Dataset):
    def __init__(self, dataframe, target_cols, tokenizer, type_text, max_token_len=512, is_train=False):
        self.tokenizer = tokenizer
        self.type_text = type_text
        self.target_cols = target_cols
        self.is_train = is_train # <--- New Flag
        
        if target_cols[0] in dataframe.columns:
            self.labels = dataframe[target_cols].values
        else:
            self.labels = np.zeros((len(dataframe), len(target_cols)))
            
        self.max_token_len = max_token_len
        self.text_data_col = dataframe[type_text].values
        self._len = len(dataframe)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        text_data = self.text_data_col[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # Missing Case
        if pd.isna(text_data):
            return (torch.zeros(self.max_token_len, dtype=torch.long), 
                    torch.zeros(self.max_token_len, dtype=torch.long), 
                    labels, 
                    False)  

        # Present Case
        encoding = self.tokenizer(
            text_data,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].flatten()
        att_mask = encoding['attention_mask'].flatten()

        # Apply Augmentation (Masking) only during training
        if self.is_train:
            # Note: 103 is usually [MASK] for BERT/ClinicalBERT. 
            # Ideally verify with tokenizer.mask_token_id
            mask_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id else 103
            vocab_size = self.tokenizer.vocab_size
            input_ids = augment_text_ids(input_ids, mask_id, vocab_size)

        return (input_ids, 
                att_mask, 
                labels, 
                True)
        
class DemographicsDatasetMissing(Dataset):
    def __init__(self, dataframe, labels):
        self.features = dataframe.values
        self.labels = labels
        self.output_dim = dataframe.shape[1]
        
        # --- THE FIX ---
        self._len = len(self.features)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        feature_row = self.features[idx]
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        
        if np.isnan(feature_row).any():
            return (torch.zeros(self.output_dim, dtype=torch.float), 
                    labels_tensor, 
                    False)
        
        return (torch.tensor(feature_row, dtype=torch.float), 
                labels_tensor, 
                True)
        

class MedicalImageDatasetMissing(Dataset):
    def __init__(self, dataframe, target_cols, img_col, transform=None):
        self.img_paths = dataframe[img_col].values
        self.target_cols = target_cols
        # Handle cases where target columns might not exist in pretraining data
        if target_cols[0] in dataframe.columns:
            self.labels = dataframe[target_cols].values
        else:
            self.labels = np.zeros((len(dataframe), len(target_cols)))
            
        self.transform = transform
        self.placeholder_shape = (3, 224, 224) 
        self._len = len(dataframe)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        img_path = self.img_paths[idx]
        
        # Missing Modality Case
        if pd.isna(img_path):
            return (torch.zeros(self.placeholder_shape), 
                    labels, 
                    False) # Present = False
        
        # Present Modality Case
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return (image, labels, True) # Present = True
            
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Returning placeholder.")
            return (torch.zeros(self.placeholder_shape), 
                    labels, 
                    False)
            
class EHRdataset(Dataset):
    def __init__(self, listfile, preprocessed_dir):
        self.data_files = []
        self.feature_shape = -1 # Set default
        
        try:
            # 1. Load the dataframe
            self.df = pd.read_csv(listfile, low_memory=False)
            
            # 2. Set description for tqdm
            desc = f"Initializing {os.path.basename(listfile)}"
            if 'tqdm' in globals():
                 desc_fn = tqdm
            else:
                 desc_fn = lambda x, **kwargs: x

            # 3. Loop over the dataframe to build file list
            for idx, row in desc_fn(self.df.iterrows(), total=len(self.df), desc=desc):
                filename = row["filename"]
                
                if pd.isna(filename):
                    self.data_files.append(None) 
                    continue
                
                preprocessed_file = os.path.join(
                    preprocessed_dir,
                    os.path.basename(filename).replace(".csv", ".h5"),
                )
                
                if os.path.exists(preprocessed_file):
                    self.data_files.append(preprocessed_file)
                else:
                    self.data_files.append(None) 
            
            # 4. (FIX) Store length and delete df *inside the try block*
            self._len = len(self.data_files)
            del self.df # Now this is safe
        
        except Exception as e:
            # 5. (FIX) If anything fails, just set length to 0
            print(f"Error reading or processing {listfile}: {e}")
            self._len = 0
            
    def _get_feature_shape(self):
        # Find the first valid file and open it to get the feature shape
        first_valid_file = next((f for f in self.data_files if f is not None), None)
        if first_valid_file:
            try:
                with h5py.File(first_valid_file, "r") as hf:
                    self.feature_shape = hf["data"].shape[1]
            except Exception as e:
                print(f"Error opening {first_valid_file} to get shape: {e}")
                # Fallback to global var if file read fails
                self._get_shape_from_global()
        else:
            # No files in this dataset. We must use the global var.
            self._get_shape_from_global()
            
    def _get_shape_from_global(self):
        # Fallback: get shape from the 'discretizer_header' global variable
        try:
            global discretizer_header
            if 'discretizer_header' in globals() and discretizer_header:
                self.feature_shape = len(discretizer_header)
            else:
                raise NameError("discretizer_header not found")
        except NameError:
            print("Error: No valid files found and global 'discretizer_header' is not defined.")
            print("Cannot determine feature shape for empty placeholder.")
            self.feature_shape = 0 # This will likely cause a downstream error, but it's a last resort

    def __getitem__(self, index):
        filepath = self.data_files[index]
        
        # If file is missing, return a placeholder
        if filepath is None:
            # Get the feature shape if we haven't already
            if self.feature_shape == -1:
                self._get_feature_shape()
                
            # Return a tensor of shape (0, num_features).
            # The seq_collate function will see len(data) = 0 and handle it correctly.
            return torch.empty(0, self.feature_shape, dtype=torch.float)
            
        # If file is present, load it
        try:
            with h5py.File(filepath, "r") as hf:
                data = hf["data"][:]
            
            # Lazily set shape on first successful load
            if self.feature_shape == -1:
                self.feature_shape = data.shape[1]
                
            return torch.tensor(data, dtype=torch.float) # Ensure it's a tensor
        
        except Exception as e:
            print(f"Error loading file {filepath} at index {index}: {e}")
            # Fallback to placeholder if file is corrupted
            if self.feature_shape == -1:
                self._get_feature_shape()
            return torch.empty(0, self.feature_shape, dtype=torch.float)

    def __len__(self):
        # Length *must* be the length of the full dataframe for alignment
        return self._len

        

class EHRdatasetFinetune(Dataset):
    def __init__(self, listfile, preprocessed_dir, classes):
        self.data_files = []
        self.labels = []

        df = pd.read_csv(listfile)
        for idx, row in df.iterrows():
            # Load preprocessed data
            preprocessed_file = os.path.join(
                preprocessed_dir, os.path.basename(row["stay"]).replace(".csv", ".h5")
            )
            if os.path.exists(preprocessed_file):
                self.data_files.append(preprocessed_file)
                self.labels.append(row[classes].values)

    def __getitem__(self, index):
        try:
            with h5py.File(self.data_files[index], "r") as hf:
                data = hf["data"][:]
            label = self.labels[index]  # Adjust this based on how you handle labels
            return data, label

        except KeyError as e:
            print(f"Error loading data from file {self.data_files[index]}: {e}")
            raise

    def __len__(self):
        return len(self.data_files)


class EHRdatasetFinetuneIHM(Dataset):
    def __init__(self, listfile, preprocessed_dir):
        self.data_files = []
        self.labels = []

        df = pd.read_csv(listfile)
        for idx, row in df.iterrows():
            # Load preprocessed data
            preprocessed_file = os.path.join(
                preprocessed_dir, os.path.basename(row["stay"]).replace(".csv", ".h5")
            )
            if os.path.exists(preprocessed_file):
                self.data_files.append(preprocessed_file)
                self.labels.append(
                    row["y_true"]
                )  # Adjust based on your actual label handling

    def __getitem__(self, index):
        try:
            with h5py.File(self.data_files[index], "r") as hf:
                data = hf["data"][:]
            label = self.labels[index]  # Adjust this based on how you handle labels
            return data, label

        except KeyError as e:
            print(f"Error loading data from file {self.data_files[index]}: {e}")
            raise

    def __len__(self):
        return len(self.data_files)


# def load_dataset(file_path):
#     with open(file_path, "rb") as file:
#         dataset = pickle.load(file)
#     return dataset

def load_dataset(file_path):
    # --- PICKLE HACK FOR TRANSFORMERS VERSION MISMATCH ---
    # Attempt to trick pickle into mapping the old, hardcoded fast tokenizer path 
    # to the current transformers structure.
    try:
        import transformers.models.distilbert.tokenization_distilbert_fast
    except ModuleNotFoundError:
        import transformers.models.distilbert.tokenization_distilbert as fallback
        sys.modules['transformers.models.distilbert.tokenization_distilbert_fast'] = fallback
    # -----------------------------------------------------
    
    with open(file_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset
    
# def seq_collate(batch):
#     data = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     data = [torch.tensor(d, dtype=torch.float) for d in data]

#     data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
#     lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)

#     return data_padded, lengths, labels

def seq_collate(batch):
    # Handle different return formats
    
    # First, check if batch elements are tensors/arrays (not tuples/lists)
    if torch.is_tensor(batch[0]) or isinstance(batch[0], np.ndarray):
        # Case: batch contains individual tensors/arrays [tensor1, tensor2, ...]
        data = batch
        # Convert numpy arrays to tensors
        data = [torch.tensor(d, dtype=torch.float) if isinstance(d, np.ndarray) else d for d in data]
        data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
        return data_padded, lengths
    
    # If we get here, batch elements are tuples/lists
    if len(batch[0]) == 3:  
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        present_flags = [item[2] for item in batch] 
    elif len(batch[0]) == 1: # Pretraining data
        data = [item[0] for item in batch]
        # Ensure data is tensor (convert numpy arrays)
        data = [torch.tensor(d, dtype=torch.float) if isinstance(d, np.ndarray) else d for d in data]
        data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
        return data_padded, lengths
    else:  # Original format: (data, labels)
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
    
    # Ensure data is tensor (convert numpy arrays)
    data = [torch.tensor(d, dtype=torch.float) if isinstance(d, np.ndarray) else d for d in data]

    data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # For missing datasets, we need to return the presence flags too
    if len(batch[0]) == 3:
        present_flags = torch.stack(present_flags) if torch.is_tensor(present_flags[0]) else torch.tensor(present_flags)
        return data_padded, lengths, labels, present_flags
    else:
        return data_padded, lengths, labels
        

def shuffle_indices(dataset, seed):
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    return indices


def get_shuffled_subset(dataset, indices):
    return Subset(dataset, indices)


class MyIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    # def __next__(self):
    #     # When the shortest loader (the one with minimum number of batches)
    #     # terminates, this iterator will terminates.
    #     # The `StopIteration` raised inside that shortest loader's `__next__`
    #     # method will in turn gets out of this `__next__` method.
    #     batches = []
    #     for loader_iter in self.loader_iters:
    #         print(f"loader_iter: {loader_iter}")
    #         batches.append(next(loader_iter))
    #     return self.my_loader.combine_batch(batches)

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class MergeLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def get_dataloaders(modalities, batch_size, data_dir, task, test=False, dataset_tag=None, evaluate_with_missing=False):
    train_inputs = []
    val_inputs = []
    test_inputs = []

    # Determine which dataset suffix to use based on missing evaluation mode
    if evaluate_with_missing and dataset_tag == "missing":
        dataset_suffix = "missing"
    else:
        dataset_suffix = ""  # Use original datasets

    # Load the first modality to get the length of the datasets
    if evaluate_with_missing and dataset_tag == "missing":
        t_inputs = load_dataset(
            f"{data_dir}/{task}/missing_train_finetune_{modalities[0]}_dataset.pkl"
        )
        v_inputs = load_dataset(
            f"{data_dir}/{task}/missing_val_finetune_{modalities[0]}_dataset.pkl"
        )
    else:
        t_inputs = load_dataset(
            f"{data_dir}/{task}/train_finetune_{modalities[0]}_dataset.pkl"
        )
        v_inputs = load_dataset(
            f"{data_dir}/{task}/val_finetune_{modalities[0]}_dataset.pkl"
        )

    # Shuffle indices once
    seed = 42  # Set a seed for reproducibility
    train_indices = shuffle_indices(t_inputs, seed)

    for modality_name in modalities:
        if test:
            if evaluate_with_missing and dataset_tag == "missing":
                test_input = load_dataset(
                    f"{data_dir}/{task}/missing_test_finetune_{modality_name}_dataset.pkl"
                )
            else:
                test_input = load_dataset(
                    f"{data_dir}/{task}/test_finetune_{modality_name}_dataset.pkl"
                )
                
            if modality_name == "ts":
                curr_test = DataLoader(
                    test_input,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
            else:
                curr_test = DataLoader(test_input, batch_size=batch_size, shuffle=False)
            test_inputs.append(curr_test)
        else:
            if evaluate_with_missing and dataset_tag == "missing":
                t_inputs = load_dataset(
                    f"{data_dir}/{task}/missing_train_finetune_{modality_name}_dataset.pkl"
                )
                v_inputs = load_dataset(
                    f"{data_dir}/{task}/missing_val_finetune_{modality_name}_dataset.pkl"
                )
            else:
                t_inputs = load_dataset(
                    f"{data_dir}/{task}/train_finetune_{modality_name}_dataset.pkl"
                )
                v_inputs = load_dataset(
                    f"{data_dir}/{task}/val_finetune_{modality_name}_dataset.pkl"
                )

            t_inputs = get_shuffled_subset(t_inputs, train_indices)

            if modality_name == "ts":
                curr_t = DataLoader(
                    t_inputs,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
                curr_v = DataLoader(
                    v_inputs,
                    batch_size=batch_size,
                    collate_fn=seq_collate,
                    shuffle=False,
                )
            else:
                curr_t = DataLoader(t_inputs, batch_size=batch_size, shuffle=False)
                curr_v = DataLoader(v_inputs, batch_size=batch_size, shuffle=False)

            train_inputs.append(curr_t)
            #print(f"train_inputs: {len(curr_t)}")
            val_inputs.append(curr_v)
            #print(f"val_inputs: {len(curr_v)}")
            #print(f"Modality name: {modality_name}")

    if test:
        return None, None, test_inputs
    else:
        return train_inputs, val_inputs, None
