# Division of Labor in Hybrid Model Memory Tasks

Division of Labor in Hybrid Model Memory Tasks is a research codebase for probing **where memory-relevant computation lives inside hybrid language models**. The project compares pure transformer baselines against hybrid architectures that mix standard self-attention with state-space / linear-attention style components, with the goal of understanding whether similar end-task accuracy comes from similar internal organization or from a different division of labor across layers and module types.

The repository is organized around **three** evaluation families:

1. **In‑Context Retrieval**  
   Measures whether a model can recover information from a prompt or reason over small structured contexts. In this repo, that task family includes:
   - `colored_objects/` for controlled retrieval‑and‑reasoning probes on short synthetic contexts
   - `NIAH/` for long‑context retrieval pressure tests.  
   See the dedicated `README_in_context_retrieval.md` for details and run instructions.

2. **State Tracking**  
   Measures whether a model can maintain and update latent world state as boxes gain, lose and move objects through a sequence of operations.  
   See `README_state_tracking.md` for task design, dataset generation and evaluation instructions.

3. **Clinical Sequence Modeling**  
   Finetunes multi‑modal sequence models on real electronic health record (EHR) datasets (MIMIC‑IV and eICU).  
   Models must ingest a time‑ordered stream of heterogeneous events—structured tabular measurements, textual notes, radiology images, demographics and projection features—and predict clinical outcomes such as mortality or a phenotype label vector.  
   The tasks in this folder are representative of long‑horizon memory and reasoning problems in applied domains.  
   A new file `README_clinical_sequence_modeling.md` describes the data format, available training scripts and evaluation pipelines.

The project framing matches the accompanying abstract and introduction: the central question is **functional localization in hybrid language models**—that is, where task-relevant computation occurs, and whether hybrid models preserve transformer-like sensitivity patterns or rely on different internal structure.

## What is in this repository

```text
.
├── Env/
│   ├── colored_objects_environment.yml
│   ├── environment.yml
│   └── requirements.txt
├── Models/
│   └── model_util.py
├── Tasks/
│   ├── In‑Context Retrieval/
│   │   ├── README.md
│   │   ├── NIAH/
│   │   └── colored_objects/
│   ├── State‑Tracking/
│   │   ├── README.md
│   │   └── entity‑tracking‑lms/
│   └── Clinical_Sequence_Modeling/
│       ├── MIMIC‑IV/
│       └── eICU/
├── Figs/
├── Utils/
└── LICENSE
```

## Core components

### `Models/model_util.py`
This is the shared utility module for:
- loading Hugging Face causal language models and tokenizers
- preparing raw and chat-style inputs
- **attention-group ablations** on hybrids via:
  - `ablate_groups(model, "self_attn")`
  - `ablate_groups(model, "linear_attn")`
- **single-layer ablations** via:
  - `ablate_single_attn_module(model, layer_idx)`

That makes this file the bridge between the paper’s layerwise functional-ablation framing and the task-specific evaluation scripts.

### `Tasks/In-Context Retrieval/`
Contains the retrieval probes:
- `colored_objects/`: custom evaluation code and saved outputs
- `NIAH/`: vendored / adapted Needle-in-a-Haystack package and contexts

### `Tasks/State-Tracking/`
Contains the state-tracking benchmark, dataset generation code, metrics, sweep pipeline, and saved evaluation outputs.

## Models currently referenced in code

The shared model registry in `Models/model_util.py` includes:

- `allenai/Olmo-3-7B-Instruct-SFT`
- `allenai/Olmo-Hybrid-Instruct-SFT-7B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`

These correspond to the two main comparison families described in the project materials:
- **OLMo-3 vs OLMo-Hybrid**
- **Qwen3 vs Qwen3.5 hybrid-style counterparts**

## Results already included

This repo is not just scaffolding; it already contains saved result files.

### Retrieval artifacts
Under `Tasks/In-Context Retrieval/colored_objects/output/` you will find:
- 14 JSON result files
- 3 SVG comparison plots

Those plots line up with the project figures showing:
- near-ceiling performance on simple color lookup
- larger hybrid gains on harder retrieval / reasoning variants such as “neither color” and spatial reasoning
- mixed behavior on arithmetic-style variants

### State-tracking artifacts
Under `Tasks/State-Tracking/entity-tracking-lms/output/` you will find:
- 34 JSON result files
- outputs for multiple box-count settings and multiple model families

The included results show a clear pattern: on the OLMo family, the pure transformer is much stronger when **zero operations** are required, but the hybrid degrades much less as the number of state updates increases and is stronger for nontrivial operation counts. Across the `boxes2` through `boxes6` datasets, the hybrid OLMo model consistently outperforms the pure OLMo transformer in set accuracy.

## Environment and setup

### Recommended starting point
For lightweight reproduction of the custom retrieval experiments, start with:

```bash
conda env create -f Env/colored_objects_environment.yml
conda activate dol-colored-objects
```

### Important note on the other environment files
`Env/environment.yml` and `Env/requirements.txt` appear to be broader snapshots and are explicitly labeled as placeholders from a previous project. They may still be useful as reference manifests, but they should not be treated as guaranteed minimal or clean environments for fresh setup.

### Typical runtime dependencies
Across the repo, the main dependencies are:
- Python 3.10
- PyTorch
- Transformers
- Accelerate
- Datasets
- NumPy / pandas / tqdm
- PyYAML (for sweep configs)

## Quick start

### Run colored-objects evaluation
```bash
cd "Tasks/In-Context Retrieval/colored_objects"

python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --subtype what_color   --prompt_mode chat   --num_test_set 250   --seed 0
```

To sweep prompt modes or ablation groups:
```bash
python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --sweep_subtypes   --sweep_prompt_modes   --sweep_ablate_groups   --num_test_set 300   --seed 0
```

### Run state-tracking evaluation
```bash
cd "Tasks/State-Tracking/entity-tracking-lms"

python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --split test   --prompt_mode chat   --num_test_set_per_bin 300   --seed 0
```

### Run the state-tracking generation + eval pipeline
```bash
python scripts/run_boxes_eval_pipeline.py --config configs/boxes_pipeline.yaml
```

This pipeline:
1. generates the dataset if the expected split files are missing
2. sweeps over YAML-configured data/model settings
3. writes JSON outputs under `output/<dataset_slug>/`

## How the repo maps to the paper/project story

The accompanying project abstract emphasizes:
- hybrid-vs-transformer comparisons
- inference-time functional ablations
- layerwise sensitivity profiles
- two memory-centric task families: **in-context retrieval** and **state tracking**

This repository directly supports that story:
- **task code** lives under `Tasks/`
- **shared ablation utilities** live under `Models/model_util.py`
- **saved baseline outputs** already exist for both task families
- **figures** in the provided `images.pdf` summarize the main empirical trends that motivated these READMEs

What is not yet centralized is a single top-level “run everything” script for the full paper pipeline. Instead, the repo is best understood as:
- a shared model/ablation utility layer
- two task-specific experiment folders
- a mix of custom scripts, vendored benchmark code, and saved outputs

## Suggested reading order

If you are new to the project, the easiest path is:

1. Read `Tasks/In-Context Retrieval/README.md`
2. Read `Tasks/State-Tracking/README.md`
3. Inspect `Models/model_util.py`
4. Open the saved JSON outputs and SVG figures
5. Extend the task scripts with `ablate_single_attn_module(...)` if you want per-layer sensitivity experiments

## License

The root repository is released under the MIT License.  
Note that some subdirectories, especially `Tasks/In-Context Retrieval/NIAH/`, carry their own upstream provenance and licensing context, so check local files there before redistribution.
