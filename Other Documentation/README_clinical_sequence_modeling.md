# Clinical Sequence Modeling

The `Tasks/Clinical_Sequence_Modeling/` directory contains **multi‑modal sequence modeling** experiments built on real electronic health record (EHR) data.  These tasks complement the synthetic retrieval and state‑tracking probes by challenging a model to integrate **heterogeneous, time‑ordered patient events** and predict clinically meaningful outcomes.

Two datasets are supported:

* **MIMIC‑IV** – a large publicly accessible EHR covering ICU stays at Beth Israel Deaconess Medical Center.  The provided scripts operate on pre‑processed WebDataset shards containing events such as demographic vectors, time‑series features, radiology images, discharge summary notes and radiology reports.  Tasks include **in‑hospital mortality** (binary) and **phenotyping** (25‑class multi‑label).  See the `MIMIC‑IV` subfolder for code.
* **eICU** – an EHR from multiple hospitals.  Scripts in `eICU` demonstrate training both multi‑modal models and simpler tabular baselines on tasks similar to those in MIMIC‑IV.

These experiments were used in the project to test whether hybrid language models can learn to update a latent belief state from long‑horizon, multi‑modal clinical sequences.

## Folder structure

````text
Clinical_Sequence_Modeling/
├── MIMIC‑IV/
│   ├── finetune_sequential.py          # Multi‑modal Belief Update Transformer (BUT) training
│   ├── evaluate_across_time.py         # Evaluate model AUROC/entropy vs number of events
│   ├── extract_runs.py                 # W&B helper to fetch best runs
│   ├── extract_runs_across_time.py     # Evaluate runs across time (optional)
│   ├── finetune_sequential.sh          # Example SLURM job for hyperparameter sweep
│   ├── picme_src/                      # Support code for encoders, data loading and utilities
│   └── …                               # Additional helper scripts
└── eICU/
    ├── finetune_sequential.py          # Multi‑modal BUT training on eICU
    ├── finetune_sequential_baseline.py # Tabular baseline model
    ├── eval_baseline_sequential.py     # Evaluate baseline model on held‑out set
    ├── eval_seq_baseline.py            # Compute metrics vs number of events
    ├── extract_best_baseline.py        # Select best baseline runs from W&B
    ├── finetune_sequential.sh          # Example SLURM sweep script
    └── …                               # Additional helper scripts
````

## Data format

Both MIMIC‑IV and eICU experiments assume that pre‑processed event sequences are stored in **WebDataset** tar archives.  Each sample is a JSON dict of events sorted by time.  Events include:

| Event type | Key in sample | Description |
|-----------|---------------|-------------|
| `img` / `[IMG]`       | `event['type']` = `'cxr'` | Path to a chest X‑ray.  Encoded via a ResNet or Vision Transformer within the training script. |
| `text_ds` / `[DS_NOTE]` | `'ds_note'`             | Discharge summary note.  Encoded with a text encoder (e.g., ClinicalBERT). |
| `text_rad` / `[RAD_NOTE]` | `'rad_note'`            | Radiology report.  Encoded with a text encoder. |
| `demo` / `[DEMO]`      | `'demo'`                | Demographic vector (e.g. age, gender, race).  Encoded by a small MLP. |
| `ts` / `[EHR_BIN]`     | `'ehr_bin'`             | Time‑series features (labs/vitals) binned per hour.  Encoded with an RNN or temporal encoder. |
| `[PREDICT]`            | sentinel event          | Indicates the point in the sequence at which a prediction should be made.  The label for the task is attached here. |

Label vectors are stored under `labels` and have length equal to the number of classes for the selected task (2 for mortality, 25 for phenotyping).  A value of `-100` indicates a masked label (no prediction at this time step).

## Training scripts

### Multi‑Modal Belief Update Transformer (BUT)

For both MIMIC‑IV and eICU the primary model is a **Belief Update Transformer** which consumes sequences of modality‑embeddings and learns to update a latent state.  The training scripts wrap this model with **LoRA** adapters for efficiency and optionally use **gradient checkpointing** and **mixed precision**.

The exact command‑line interface varies between the MIMIC and eICU training scripts (MIMIC delegates to `picme_src/argparser.py`, whereas eICU defines its own `argparse` parameters).  In general you will need to specify:

| Argument | Description |
|---|---|
| `--model_name` or `--llm_model_name` | Hugging Face model ID for the base causal language model (e.g. `allenai/Olmo-Hybrid-Instruct-SFT-7B` or `meta-llama/Meta-Llama-3-8B`). |
| `--task` | Which prediction task to train.  For MIMIC this is `mortality` or `phenotyping`; for eICU it is `mortality_task` or `los_task`. |
| `--learning_rate`, `--weight_decay` | Optimizer hyperparameters.  MIMIC accepts lists of learning rates for sweeps via its finetune argparser; eICU accepts single floating‑point values. |
| `--epochs` or `--num_train_epochs` | Number of passes over the data.  The MIMIC script uses `--num_train_epochs` (via `picme_src.finetune_arg_parser`), whereas the eICU scripts use `--epochs`. |
| `--batch_size` and `--accumulation_steps` | Per‑GPU batch size and gradient accumulation steps. |
| `--data_dir` | Root directory containing WebDataset shards (`*_events.tar` for on‑the‑fly embedding, or precomputed shard directories). |
| `--save_prefix` | Prefix for saved model checkpoints and logs. |
| LoRA / PEFT flags | Parameters such as `--use_lora`, `--lora_r`, `--lora_alpha` and `--lora_dropout` control the rank and strength of the low‑rank adaptation. |
| Miscellaneous | Additional flags such as `--seed_number` for reproducibility and `--attn_implementation` to select attention kernels. |

### Example: fine‑tuning on MIMIC‑IV

The MIMIC training script is driven by the `picme_src` argument parser and supports sweeping over multiple hyperparameters.  As a simple example you can fine‑tune a hybrid model on the mortality task like this:

```bash
cd "Tasks/Clinical_Sequence_Modeling/MIMIC-IV"

python finetune_sequential.py \
  --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
  --task mortality \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --num_train_epochs 5 \
  --batch_size 2 \
  --data_dir /path/to/mimic_preprocessed \
  --save_prefix mimic_but_hybrid
```

The actual script accepts many more options (see `picme_src/argparser.py`), including lists of learning rates, LoRA ranks, gradient accumulation steps and the ability to freeze certain LLM layers.  The included `finetune_sequential.sh` demonstrates how to perform hyper‑parameter sweeps on a SLURM cluster.

### Example: fine‑tuning on eICU

For the eICU dataset the multi‑modal training script defines its own CLI.  You must specify the task name (`mortality_task` or `los_task`), the Hugging Face LLM, optimizer settings and LoRA parameters.  For example to train a hybrid model on the mortality task:

```bash
cd "Tasks/Clinical_Sequence_Modeling/eICU"

python finetune_sequential.py \
  --task mortality_task \
  --data_dir /path/to/eicu_precomputed \
  --llm_model_name meta-llama/Meta-Llama-3-8B \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --epochs 5 \
  --batch_size 4 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --save_prefix eicu_but_hybrid
```

Refer to the comments at the top of `eICU/finetune_sequential.py` and the provided `finetune_sequential.sh` for a full list of options and default values.  These scripts also enable gradient checkpointing and LoRA by default; remove `--use_lora` or change the ranks to disable or adjust the adaptation.

### Baseline tabular model (eICU only)

The eICU folder also includes `finetune_sequential_baseline.py`, which trains a **TabularEncoder + Belief Update Transformer** on pre‑computed tabular vectors.  Use this when GPU budget is limited or you wish to isolate the effect of tabular features.  The baseline script does not take a `--model_name`; instead you must supply the task, data directory and underlying LLM via `--llm_model_name`.  A minimal example is:

```bash
cd "Tasks/Clinical_Sequence_Modeling/eICU"

python finetune_sequential_baseline.py \
  --task mortality_task \
  --data_dir /path/to/eicu_precomputed \
  --llm_model_name meta-llama/Meta-Llama-3-8B \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --epochs 5 \
  --batch_size 16 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --save_prefix eicu_tabular_baseline
```

As with the multi‑modal script, additional options let you control LoRA dropout, patience for early stopping and the attention implementation.  Consult the top of `finetune_sequential_baseline.py` for details.

## Evaluation scripts

### Evaluating across time (MIMIC‑IV)

`evaluate_across_time.py` assesses a trained model’s performance as a function of how many events have been observed.  It operates on **pre‑computed embeddings** (WebDataset shards) and a trained model checkpoint.  The script will group predictions by the number of non‑`[PREDICT]` events encountered and compute AUROC and entropy curves.

Important options include:

| Option | Description |
|---|---|
| `--state_dict_path` | Path to the `.pth` checkpoint of the fine‑tuned model. |
| `--llm_model_name` | Hugging Face model ID used at training time. |
| `--data_dir` | Directory containing `picme_sequential_<task>_precomputed` shards. |
| `--batch_size` | Batch size during evaluation (default 1). |
| `--output_dir` | Directory where plots and CSVs will be saved. |
| `--sequential_model` | Set this flag if you are evaluating a sequential Belief Update Transformer (as opposed to a static late‑fusion model). |

Example:

```bash
cd "Tasks/Clinical_Sequence_Modeling/MIMIC-IV"

python evaluate_across_time.py \
  --state_dict_path /path/to/mimic_but_hybrid_best.pth \
  --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
  --data_dir /path/to/mimic_precomputed \
  --batch_size 16 \
  --sequential_model \
  --output_dir mimic_eval_time
```

The script writes a CSV summarizing AUROC and mean entropy for each event bucket and produces a PNG/Matplotlib figure if run in an environment with display support.

### Evaluating eICU models and baselines

The eICU folder provides several evaluation utilities.  Unlike the MIMIC evaluator, these scripts expect a **CSV file listing the best model checkpoints** (typically produced by `extract_best_baseline.py`) and will iterate over the listed runs.  The workflow is:

1. **Select the best runs from W&B**

   Use `extract_best_baseline.py` to query the Weights & Biases API for each task and find the model checkpoint with the highest validation AUROC (or lowest MSE for length‑of‑stay).  This writes a CSV with columns `Task`, `LLM`, `File_Path` and metrics:

   ```bash
   cd "Tasks/Clinical_Sequence_Modeling/eICU"
   python extract_best_baseline.py --entity <wandb_user> --output_csv best_eicu_models.csv
   ```

2. **Evaluate overall test performance**

   `eval_baseline_sequential.py` reads the CSV, loads each listed checkpoint into an `EndToEndBUT` model and evaluates it on the test split.  You must supply the base data directory containing the precomputed vectors:

   ```bash
   python eval_baseline_sequential.py \
     --input_csv best_eicu_models.csv \
     --base_data_dir /path/to/eicu_precomputed \
     --output_csv eicu_baseline_test_metrics.csv
   ```

   The script prints per‑run AUROC/MSE and writes a summary CSV.

3. **Compute metrics versus number of events**

   To understand how quickly the model’s confidence improves as more events are observed, run `eval_seq_baseline.py`.  It takes the same input CSV and base data directory and writes a metrics file per model:

   ```bash
   python eval_seq_baseline.py \
     --input_csv best_eicu_models.csv \
     --base_data_dir /path/to/eicu_precomputed \
     --output_dir eicu_metrics_vs_events
   ```

   The resulting CSVs can be plotted with the included helper functions to visualize AUROC or Spearman correlation as a function of sequence length.

### Selecting best runs with Weights & Biases

Both MIMIC‑IV and eICU experiments log metrics to Weights & Biases.  The helper scripts `extract_runs.py` (for MIMIC) and `extract_best_baseline.py` (for eICU) connect to the API, enumerate runs, filter by model family and pick the run with the highest validation AUROC (or lowest MSE).  They write their results to a CSV for convenient inspection.

Example usage for MIMIC:

```bash
cd "Tasks/Clinical_Sequence_Modeling/MIMIC-IV"
python extract_runs.py --entity <wandb_user> --project <project_name> --output_csv best_mimic_models.csv
```

## Environment setup

The clinical sequence modeling scripts have heavier dependencies than the retrieval/state‑tracking tasks.  A typical environment includes:

* Python 3.10
* PyTorch ≥ 2.0 with CUDA support
* Hugging Face `transformers` and `accelerate`
* `bitsandbytes` for 8‑bit optimizers
* `datasets` and `webdataset` for data loading
* `tqdm`, `numpy`, `pandas`, `matplotlib`, `seaborn`
* `wandb` for experiment tracking

One way to set up such an environment is to start from the general `Env/environment.yml` in the repo and then add any missing packages using `pip install -r requirements.txt`.  Be sure to install the appropriate CUDA toolkit for your GPU.

## Notes and caveats

* **Data access** – The scripts assume access to pre‑processed MIMIC‑IV/eICU data in WebDataset format.  Preparing these shards requires access to the original EHR datasets and is not covered by this repo.
* **Compute requirements** – Training the Belief Update Transformer on MIMIC‑IV or eICU requires multiple GPUs with at least 24 GB memory each.  The baseline tabular models can be trained on a single GPU.
* **LoRA/quantization** – The scripts default to LoRA adaptation and optional quantization for memory efficiency.  Adjust `--lora_r` and `--lora_alpha` to trade off memory vs capacity.  To train the full model without LoRA, pass `--llm_peft none`.

By following the examples above and adapting file paths and hyperparameters to your own setup, you should be able to reproduce the clinical sequence modeling experiments and extend them with your own models.