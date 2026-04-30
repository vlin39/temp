import math

import torch
import torch.nn as nn
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from picme_src.encoders import *
from picme_src.losses import *
from .picme_utils import secure_fusion
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from flash_attn import flash_attn_func

def check_tensor_health(tensor, name="Tensor"):
    if tensor is None:
        return
    
    is_nan = torch.isnan(tensor).any()
    is_inf = torch.isinf(tensor).any()
    
    if is_nan or is_inf:
        print(f"\n[!] CRITICAL: {name} contains {'NaN' if is_nan else ''} {'Inf' if is_inf else ''}")
        print(f"    - Min: {tensor.min().item()}")
        print(f"    - Max: {tensor.max().item()}")
        print(f"    - Mean: {tensor.mean().item()}")
        print(f"    - NaN count: {torch.isnan(tensor).sum().item()}")
        return False # Unhealthy
    return True # Healthy


class BaseContrastiveModel(nn.Module):
    def __init__(self, ts_input_dim, demo_input_dim, projection_dim, num_modalities):
        super(BaseContrastiveModel, self).__init__()
        self.image_encoder = ImageEncoder(projection_dim)
        self.text_encoder = TextEncoder(projection_dim)
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, projection_dim)
        self.demo_encoder = DemoEncoder(demo_input_dim, projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # self.projection = nn.Sequential(
        #     nn.Linear(projection_dim, projection_dim),
        #     nn.BatchNorm1d(projection_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),  # <--- Add this!
        #     nn.Linear(projection_dim, projection_dim),
        # )
        self.num_modalities = num_modalities
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():
            if isinstance(model, (nn.LSTM, nn.RNN, nn.GRU)):
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def encode_modality(self, data, modality_type, att_mask=None, lengths=None):
        if modality_type == "img":
            features = self.image_encoder(data)
        elif modality_type in ["text_rad", "text_ds"]:
            features = self.text_encoder(data, att_mask)
        elif modality_type == "ts":
            #print(f"data: {data.shape}")
            #print(f"lengths: {lengths}")
            batch_size = data.shape[0]
            
            # 1. Create a zero placeholder for the entire batch
            features = torch.zeros(batch_size, 
                                   self.ts_encoder.output_dim, 
                                   device=data.device, 
                                   dtype=data.dtype)
            
            # 2. Find which samples are actually present (length > 0)
            present_mask = lengths > 0
            num_present = present_mask.sum()

            # 3. If any samples are present, encode just them
            if num_present > 0:
                present_data = data[present_mask]
                present_lengths = lengths[present_mask]
                
                # 4. Run the encoder ONLY on the present data
                present_features = self.ts_encoder(present_data, present_lengths.cpu())
                
                # 5. Scatter the results back into the placeholder tensor
                features[present_mask] = present_features.to(dtype=features.dtype)
        elif modality_type == "demo":
            features = self.demo_encoder(data)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

        return self.projection(features)


class ContrastiveModel(BaseContrastiveModel):
    def __init__(self, ts_input_dim, demo_input_dim, projection_dim):
        super(ContrastiveModel, self).__init__(
            ts_input_dim, demo_input_dim, projection_dim
        )

    def forward(
        self,
        modality1,
        modality2,
        modality1_type,
        modality2_type,
        att_mask1=None,
        att_mask2=None,
        lengths=None,
    ):
        embeddings1 = self.encode_modality(
            modality1, modality1_type, att_mask1, lengths
        )
        embeddings2 = self.encode_modality(
            modality2, modality2_type, att_mask2, lengths
        )
        return embeddings1, embeddings2


class MultiModalContrastiveModel(BaseContrastiveModel):
    def __init__(
        self, ts_input_dim, demo_input_dim, projection_dim, num_modalities, training_strategy='all_pairs' 
    ):
        super(MultiModalContrastiveModel, self).__init__(
            ts_input_dim, demo_input_dim, projection_dim, num_modalities=num_modalities
        )
        self.training_strategy = training_strategy
        if self.training_strategy in ['ovo', 'masked_global', 'focal']:
            self.N_weights = nn.Parameter(
                torch.ones(self.num_modalities) / self.num_modalities
            )
        else: # 'all_pairs'
            num_pairs = self.num_modalities * (self.num_modalities - 1) // 2
            self.pair_weights = nn.Parameter(torch.ones(num_pairs) / num_pairs)

        self.missing_token_img = nn.Parameter(torch.randn(projection_dim))
        self.missing_token_text_rad = nn.Parameter(torch.randn(projection_dim))
        self.missing_token_text_ds = nn.Parameter(torch.randn(projection_dim))
        self.missing_token_ts = nn.Parameter(torch.randn(projection_dim))
        self.missing_token_demo = nn.Parameter(torch.randn(projection_dim))
        
        self.modality_token_map = {
            "img": self.missing_token_img,
            "text_rad": self.missing_token_text_rad,
            "text_ds": self.missing_token_text_ds,
            "ts": self.missing_token_ts,
            "demo": self.missing_token_demo
        }

        self.log_tau = nn.Parameter(torch.ones([]) * np.log(0.07))

    def get_temperature(self):
        # Clamp to avoid numerical instability
        return torch.exp(self.log_tau).clamp(min=0.07, max=0.5)

    def forward(self, modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths):
        embeddings = []
        for data, modality_type in zip(modalities_data, modalities_type):
            #print(f"data: {data}")
            #print(f"modality_type: {modality_type}")
            embeddings.append(self.encode_modality(
                    data,
                    modality_type,
                    (
                        mask_rad
                        if modality_type == "text_rad"
                        else mask_ds if modality_type == "text_ds" else None
                    ),
                    ts_lengths if modality_type == "ts" else None,
                )
                             )
        return embeddings

class MultiModalContrastiveOGModel(BaseContrastiveModel):
    def __init__(
        self, ts_input_dim, demo_input_dim, projection_dim, num_modalities, training_strategy='all_pairs' 
    ):
        super(MultiModalContrastiveOGModel, self).__init__(
            ts_input_dim, demo_input_dim, projection_dim, num_modalities=num_modalities
        )
        self.training_strategy = training_strategy
        if self.training_strategy in ['ovo', 'masked_global', 'focal']:
            self.N_weights = nn.Parameter(
                torch.ones(self.num_modalities) / self.num_modalities
            )
        else: # 'all_pairs'
            num_pairs = self.num_modalities * (self.num_modalities - 1) // 2
            self.pair_weights = nn.Parameter(torch.ones(num_pairs) / num_pairs)

    def forward(self, modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths):
        embeddings = []
        for data, modality_type in zip(modalities_data, modalities_type):
            #print(f"data: {data}")
            #print(f"modality_type: {modality_type}")
            embeddings.append(self.encode_modality(
                    data,
                    modality_type,
                    (
                        mask_rad
                        if modality_type == "text_rad"
                        else mask_ds if modality_type == "text_ds" else None
                    ),
                    ts_lengths if modality_type == "ts" else None,
                )
                             )
        return embeddings


class MultiModalBaseline(nn.Module):
    def __init__(
        self,
        ts_input_dim,
        demo_input_dim,
        projection_dim,
        args,  # args passed directly
        device,
    ):
        """
        Initialize encoders dynamically. Classification head is now separate.
        """
        super(MultiModalBaseline, self).__init__()
        self.device = device
        self.modalities = args.modalities
        self.num_modalities = len(self.modalities)
        self.args = args

        # Initialize encoders dynamically using ModuleDict
        self.encoders = nn.ModuleDict({
            "img": ImageEncoder(projection_dim),
            "text_rad": TextEncoder(projection_dim),
            "text_ds": TextEncoder(projection_dim),
            "ts": TimeSeriesEncoder(ts_input_dim, projection_dim),
            "demo": DemoEncoder(demo_input_dim, projection_dim),
        })

        # Give the baseline learnable tokens to ensure a fair ablation
        self.modality_token_map = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(projection_dim)) for mod in self.modalities
        })

    def encode_modality(self, data, modality_type, att_mask=None, lengths=None):
        """
        Encode a single modality using the appropriate encoder from ModuleDict.
        """
        if modality_type not in self.encoders:
            raise ValueError(f"Unknown modality type: {modality_type}")

        if modality_type in ["text_rad", "text_ds"]:
            features = self.encoders[modality_type](data, att_mask)
        elif modality_type == "ts":
            batch_size = data.shape[0]
            encoder = self.encoders[modality_type]
            
            # 1. Create a zero placeholder
            features = torch.zeros(batch_size, 
                                   encoder.output_dim, 
                                   device=data.device, 
                                   dtype=data.dtype)
            
            # 2. Find present samples
            present_mask = lengths > 0
            num_present = present_mask.sum()

            # 3. If any samples are present, encode just them
            if num_present > 0:
                present_data = data[present_mask]
                present_lengths = lengths[present_mask]
                
                # 4. Run the encoder ONLY on the present data
                present_features = encoder(present_data, present_lengths.cpu())
                
                # 5. Scatter the results back
                features[present_mask] = present_features
                
        else:
            features = self.encoders[modality_type](data)

        return features

    def forward(
        self,
        modalities_data,
        modalities_type,
        mask_rad=None,
        mask_ds=None,
        ts_lengths=None,
    ):
        """
        Purely returns embeddings, just like the Contrastive model.
        """
        embeddings = [
            self.encode_modality(
                data,
                modality_type,
                att_mask=(
                    mask_rad
                    if modality_type == "text_rad"
                    else mask_ds if modality_type == "text_ds" else None
                ),
                lengths=(ts_lengths if modality_type == "ts" else None),
            )
            for data, modality_type in zip(modalities_data, modalities_type)
        ]

        return embeddings
        


class ModalityEnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, modality_lambda=None, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(device),
                torch.zeros(bs, self.hidden_size).to(device),
            )
        else:
            h_t, c_t = init_states

        hs = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            lambda_t = 1 if not modality_lambda else modality_lambda[t]

            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :hs]),  # input
                torch.sigmoid(gates[:, hs : hs * 2]),  # forget
                torch.tanh(gates[:, hs * 2 : hs * 3]),  # candidate memory
                torch.sigmoid(gates[:, hs * 3 :]),  # output
            )

            modality_scale_factor = torch.zeros_like(g_t) + lambda_t
            c_t = f_t * c_t + ((i_t * g_t) * modality_scale_factor)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        projection_dim,
        num_classes,
        fusion_method,
        num_modalities,
        modality_lambdas,
        verbose=True,
    ):
        super(ClassificationHead, self).__init__()
        self.fusion_method = fusion_method
        self.num_modalities = num_modalities
        self.modality_lambdas = modality_lambdas

        if fusion_method == "concatenation":
            if verbose:
                print(f"Building a {fusion_method} fusion classifier head.")
            self.classifier = nn.Sequential(
                nn.Linear(projection_dim * num_modalities, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        elif fusion_method == "vanilla_lstm":
            self.vanilla_lstm = nn.LSTM(
                projection_dim, projection_dim, batch_first=True, dropout=0.0
            )
            # REMOVED SIGMOID to allow CrossEntropyLoss/BCEWithLogitsLoss to function normally
            self.classifier = nn.Linear(projection_dim, num_classes)
        elif fusion_method == "modality_lstm":
            assert num_modalities == len(modality_lambdas)
            self.modality_lstm = ModalityEnhancedLSTM(projection_dim, projection_dim)
            # REMOVED SIGMOID
            self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, embeddings):
        if self.fusion_method == "concatenation":
            output = self.classifier(embeddings)
        elif self.fusion_method == "vanilla_lstm":
            seq_lengths = np.array([self.num_modalities] * len(embeddings))
            feats = torch.nn.utils.rnn.pack_padded_sequence(
                embeddings, seq_lengths, batch_first=True, enforce_sorted=False
            )
            _, (ht, _) = self.vanilla_lstm(feats)
            lstm_out = ht.squeeze(0)
            output = self.classifier(lstm_out)
        elif self.fusion_method == "modality_lstm":
            _, (ht, _) = self.modality_lstm(
                embeddings, modality_lambda=self.modality_lambdas
            )
            output = self.classifier(ht)

        return output
        

class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning head for multi-class classification.
    Outputs the parameters (alpha) of a Dirichlet distribution.
    """
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Hidden states from the LLM [B, S, H] or [B*S, H]
        Returns:
            torch.Tensor: Alpha parameters [B, S, C] or [B*S, C], where alpha > 1
        """
        evidence = F.softplus(self.linear(x)) # Use softplus for non-negativity
        alpha = evidence + 1
        return alpha

def evidential_loss_function(alpha, labels_one_hot, current_epoch, max_epochs, annealing_steps=10):
    """
    Type-II Maximum Likelihood Loss for Evidential Deep Learning.
    
    Args:
        alpha (torch.Tensor): Output from EvidentialHead (alpha > 1) [N, C]
        labels_one_hot (torch.Tensor): One-hot encoded labels [N, C]
        current_epoch (int): The current training epoch for annealing.
        max_epochs (int): Total epochs for annealing.
        annealing_steps (int): Number of epochs to ramp up the KL term.
    """
    # 1. Calculate the Dirichlet strength (S)
    dirichlet_strength = torch.sum(alpha, dim=1, keepdim=True) # [N, 1]

    # 2. Calculate the main cross-entropy-like loss
    log_likelihood = torch.sum(labels_one_hot * (torch.log(dirichlet_strength) - torch.log(alpha)), dim=1)
    loss_ce = torch.mean(log_likelihood)

    # 3. Calculate the KL divergence term (annealed)
    # This regularizes the model to prevent it from predicting 0 uncertainty
    # for out-of-distribution samples.
    
    # Annealing coefficient (ramps up from 0 to 1)
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

# class PositionalEncoding(nn.Module):
#     """
#     Standard sinusoidal positional encoding.
#     From PyTorch tutorials: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#     """
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
        
#         # pe shape is [max_len, 1, d_model]. 
#         # We transpose to [1, max_len, d_model] to be batch_first compatible
#         pe = pe.transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, embedding_dim]
#         """
#         # x is [B, S, E]. self.pe is [1, max_len, E].
#         # Slicing `self.pe[:, :x.size(1)]` gives [1, S, E].
#         # This broadcasts perfectly with x.
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)

        
class BeliefUpdateTransformer(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.llm_model_name = args.llm_model_name
        self.input_embedding_dim = args.input_embedding_dim
        self.num_classes = num_classes
        self.use_evidential_head = args.use_evidential_head
        #self.pos_encoder = PositionalEncoding(d_model=self.input_embedding_dim, dropout=0.1, max_len=10000)

        # Defaults to flash_attention_2 for backwards compatibility in training
        attn_impl = getattr(args, "attn_implementation", "flash_attention_2")

        # 1. Load the LLM with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(self.llm_model_name, trust_remote_code=True)
        
        if "phi-3" in self.llm_model_name.lower():
            attn_impl="eager"
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                rope_type = config.rope_scaling.get("rope_type") or config.rope_scaling.get("type")
                if rope_type == "default":
                    config.rope_scaling = None
                elif "type" not in config.rope_scaling:
                    config.rope_scaling["type"] = config.rope_scaling.get("rope_type", "su")
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            config=config,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=attn_impl
        )

        self.llm.gradient_checkpointing_enable()

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
        # Maps your [B, S, 256] modality embeddings to [B, S, 4096] (or LLM hidden dim)
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
        """Custom load_state_dict to handle quantized weights with verbose logging"""
        # Filter out the quantized weight parameters
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if any(x in key for x in ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state']):
                continue
            filtered_state_dict[key] = value
        
        # --- CHANGE START ---
        # Capture the result instead of ignoring it
        result = super().load_state_dict(filtered_state_dict, strict=False)
        
        # Explicitly check for critical missing keys
        if result.missing_keys:
            print("\n[WARNING] Missing Keys during loading (These weights stayed Random):")
            # Filter out the quantization keys from the warning to see the real issues
            real_missing = [k for k in result.missing_keys if 'quant' not in k and 'absmax' not in k]
            if real_missing:
                for k in real_missing:
                    print(f"  - {k}")
            else:
                print("  (Only quantization keys were missing, which is expected.)")

        if result.unexpected_keys:
            print("\n[WARNING] Unexpected Keys in state_dict:")
            for k in result.unexpected_keys:
                print(f"  - {k}")
        # --- CHANGE END ---
        
        return result
        
    # --- FIX 2: Add output_attentions=False to signature ---
    def forward(self, modality_embeddings_sequence, attention_mask, output_attentions=False):
        """
        [Your existing docstring...]
        """
        inputs_embeds = self.input_projection(modality_embeddings_sequence) 
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        if not check_tensor_health(inputs_embeds, "inputs_embeds"):
            print(f"inputs_embeds: {inputs_embeds}")
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions, # <--- Pass it to the LLM
            use_cache=False
        )
        
        last_hidden_states = outputs.hidden_states[-1]

        if torch.isnan(last_hidden_states).any():
            print("[!] CRITICAL: NaNs generated inside LLM layers!")
        
        final_outputs = self.classification_head(last_hidden_states.float())
        
        # --- FIX 3: Conditionally return the attentions tuple ---
        if output_attentions:
            return final_outputs, outputs.attentions
            
        return final_outputs
        