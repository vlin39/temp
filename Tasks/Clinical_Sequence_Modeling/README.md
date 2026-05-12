# Clinical Sequence Modeling

This task trains a **Belief Update Transformer (BUT)** on sequential clinical events to predict
in-hospital mortality at any point during a patient's ICU stay. The core research question:
do hybrid LLM backbones (interleaving MHA and linear/gated-recurrent attention) better capture
long-range dependencies in medical time series than pure-transformer baselines?

Two datasets are supported: **MIMIC-IV** (multi-modal, 5 input types) and **eICU** (tabular,
7 event types).

> Both datasets require institutional data access agreements (PhysioNet). The scripts assume
> access to pre-processed data in WebDataset shard format; preparing these shards requires
> institutional access and is not covered by this repository.

---

## Datasets

### MIMIC-IV

| Property | Value |
|----------|-------|
| Task | In-hospital mortality; phenotyping (multilabel) |
| Train / val split | 910 / 190 (mortality); 1420 / 310 (phenotyping) |
| Modalities | CXR images, radiology notes, discharge summaries, vitals/labs, demographics |
| Embedding dim | 256 per modality |
| Input format | Precomputed embeddings in WebDataset `.tar` shards |

Each patient record is a chronological sequence of clinical events; each event carries a 256-dim
modality embedding and a type label. `[PREDICT]` marker events are interleaved to trigger
anytime mortality predictions.

### eICU

| Property | Value |
|----------|-------|
| Tasks | In-hospital mortality (binary); length-of-stay (regression) |
| Event types | Demographics, diagnosis, treatment, medication, lab, APS, vital signs |
| Input format | Raw tabular vectors, padded to max feature dimension |

---

## Model Architecture

The Belief Update Transformer processes clinical events as a token sequence fed into a
quantized LLM backbone.

1. **Input** — 256-dim modality embeddings (or raw tabular vectors) interleaved with learnable
   `[PREDICT]` tokens.
2. **LLM backbone** — 4-bit quantized causal language model (OLMo, Qwen, Llama, Mistral).
   LoRA fine-tuning is applied to keep training tractable.
3. **Classification head** — Either a standard 3-layer MLP (→ ReLU → Dropout → logits) or an
   **Evidential** head (Dirichlet parameter predictor) for uncertainty quantification via
   evidential deep learning.
4. **Output** — Per-timestep mortality logits at every `[PREDICT]` position, enabling
   *anytime prediction* as the sequence grows.

See `Figs/Clinical-Sequence-Modeling/fig_clinical_architecture_diagram.jpeg` for a visual overview.

---

## MIMIC-IV Training

```bash
cd Tasks/Clinical_Sequence_Modeling

python MIMIC-IV/finetune_sequential.py \
    --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sequential_model \
    --task mortality \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --use_lora \
    --state_dict <pretrained_encoder.pth>
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm_model_name` | `mistralai/Mistral-7B-v0.1` | HuggingFace LLM backbone |
| `--sequential_model` | off | Use the sequential BUT pipeline (required) |
| `--task` | `mortality` | `mortality` or `phenotyping` |
| `--batch_size` | `4` | Training batch size |
| `--learning_rate` | `5e-5` | AdamW learning rate |
| `--epochs` | `10` | Maximum training epochs |
| `--use_lora` | off | Enable LoRA fine-tuning |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA scaling factor |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--input_embedding_dim` | `256` | Modality embedding dimension |
| `--use_evidential_head` | off | Use Evidential DL head for uncertainty |
| `--annealing_steps` | — | KL annealing epochs for evidential training |
| `--compute_embeddings_on_fly` | off | Encode raw data during training instead of loading precomputed embeddings |
| `--state_dict` | — | Path to pretrained encoder weights |
| `--objective` | `val_auroc` | Metric to optimize (must be in `--metrics`) |
| `--metrics` | `auroc auprc f1` | Metrics to track |
| `--accumulation_steps` | `1` | Gradient accumulation steps |

---

## eICU Training

```bash
python eICU/finetune_sequential.py \
    --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --task mortality \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --use_lora
```

For length-of-stay regression, use `--task los`. Loss switches automatically to MSE on
log-transformed targets.

Baseline (non-LLM) models are in `eICU/finetune_sequential_baseline.py` and use a lighter
CNN/LSTM encoder stack without the LLM backbone.

---

## CLI Differences: MIMIC-IV vs eICU

> MIMIC-IV training delegates all argument parsing to `MIMIC-IV/picme_src/argparser.py`.
> eICU training defines its own `argparse` parameters directly in each script. The two
> interfaces are not identical — use each script's `--help` for the authoritative argument
> list. In particular, the epoch count argument is `--num_train_epochs` in the MIMIC scripts
> and `--epochs` in the eICU scripts.

---

## Evaluation Across Time

`evaluate_across_time.py` measures AUROC (and optionally predictive entropy) as a function of
how many clinical events have been observed — revealing how quickly the model reaches a reliable
prediction.

```bash
python MIMIC-IV/evaluate_across_time.py \
    --state_dict_path <trained_model.pth> \
    --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --output_dir results/
```

**Outputs:**
- CSV with columns `num_events`, `auroc`, `mean_entropy`, `count`
- PNG plots of AUROC and uncertainty vs. sequence length

Pre-rendered evaluation plots are in `Figs/Clinical-Sequence-Modeling/fig_clinical_*_auroc_vs_seq_len.jpeg`.

---

## Output Format

| Output | Description |
|--------|-------------|
| `{prefix}_BUT_{task}_{model}_best.pth` | Best model checkpoint (saved by validation AUROC) |
| W&B run | Per-epoch AUROC, AUPRC, loss (when `--wandb_project` is set) |
| `results/{model}_{task}_metrics.csv` | Evaluation CSV from `evaluate_across_time.py` |
| `results/{model}_{task}_auroc.png` | AUROC-vs-sequence-length plot |
