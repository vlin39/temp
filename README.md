# Attention Ablation in Hybrid Language Models

This project investigates where task-relevant computation is localized in **hybrid LLMs** — models
that interleave standard multi-head self-attention (MHA) with linear/gated-recurrent attention
layers — versus matched pure-transformer baselines. The ablation methodology: monkey-patch one
attention family out of a loaded model (replace its `forward` with an identity pass), measure task
degradation, then restore and repeat for the other family. This isolates the contribution of each
component type without retraining.

Primary model pair: **OLMo-3-7B-Instruct-SFT** (pure transformer) vs.
**OLMo-Hybrid-Instruct-SFT-7B** (hybrid). Extended comparisons use Qwen3 / Qwen3.5 hybrids.
Three task families are studied: in-context retrieval, box entity state-tracking, and clinical
sequence modeling for mortality prediction.

---

## Repository Layout

```
.
├── Env/                                        # Python environment specs
├── Figs/                                       # All figures, organized by task
│   ├── In-Context-Retrieval/
│   ├── State-Tracking/
│   ├── Clinical-Sequence-Modeling/
│   ├── Misc/
│   └── figure_descriptions.md
├── Models/
│   └── model_util.py                           # Model loading + ablation utilities
└── Tasks/
    ├── In-Context Retrieval/                   # NIAH + Colored Objects
    │   ├── NIAH/
    │   └── colored_objects/
    ├── State-Tracking/
    │   └── entity-tracking-lms/                # Box entity tracking
    └── Clinical_Sequence_Modeling/             # MIMIC-IV + eICU mortality
        ├── MIMIC-IV/
        └── eICU/
```

---

## Models

| Key | HuggingFace ID | Type |
|-----|----------------|------|
| `olmo3_7b_instruct` | `allenai/Olmo-3-7B-Instruct-SFT` | Pure transformer |
| `olmo_hybrid_7b_instruct` | `allenai/Olmo-Hybrid-Instruct-SFT-7B` | Hybrid |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Hybrid |
| `qwen3-32b` | `Qwen/Qwen3-32B` | Hybrid |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | Hybrid |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | Hybrid |

Model IDs are also importable from `Models/model_util.py`:

```python
from Models.model_util import MODEL_IDS
```

---

## Ablation Design

Ablation utilities live in `Models/model_util.py`. The key function is `ablate_groups`, which
matches submodule names to the last path segment (`self_attn` or `linear_attn`) and replaces their
`forward` method with a no-op that returns the input tensor unchanged.

```python
from Models.model_util import (
    load_model_and_tokenizer,
    ablate_groups,
    restore_attention_ablation,
    ablate_single_attn_module,
)

model, tokenizer = load_model_and_tokenizer("allenai/Olmo-Hybrid-Instruct-SFT-7B")

# Ablate all standard self-attention blocks, leaving linear attention intact
patched = ablate_groups(model, "self_attn")   # returns list of patched module names

# ... run inference ...

restore_attention_ablation(model)             # undo all patches
```

| `group` value | Effect |
|---------------|--------|
| `"none"` | Remove any active ablation (restore) |
| `"self_attn"` | Zero out all MHA blocks; linear attention layers remain active |
| `"linear_attn"` | Zero out all linear/gated-recurrent blocks; MHA layers remain active |

For layer-by-layer ablation:

```python
# Ablate only decoder layer 12 (whichever attention type it has)
patched = ablate_single_attn_module(model, layer_idx=12)
```

---

## Environment Setup

> **Note:** `Env/environment.yml` and `Env/requirements.txt` are broader snapshots from a
> previous project and should not be treated as guaranteed minimal or clean environments.
> `Env/colored_objects_environment.yml` is the recommended starting point for lightweight
> reproduction.

```bash
# Recommended: conda env for colored objects / state tracking tasks
conda env create -f Env/colored_objects_environment.yml
conda activate dol-colored-objects

# pip (broader)
pip install -r Env/requirements.txt
```

Key dependencies: Python 3.10, PyTorch, Transformers, Accelerate, Datasets,
NumPy / pandas / tqdm, PyYAML.

The NIAH benchmark has a lighter self-contained environment:

```bash
pip install -r "Tasks/In-Context Retrieval/NIAH/requirements.txt"
```

---

## Quick Start

**Colored Objects (single run)**
```bash
cd "Tasks/In-Context Retrieval/colored_objects"
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color --prompt_mode chat --num_test_set 250 --seed 0
```

**Colored Objects (full ablation sweep)**
```bash
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sweep_ablate_groups --sweep_subtypes --sweep_prompt_modes --num_test_set 250
```

**Entity Tracking**
```bash
cd Tasks/State-Tracking/entity-tracking-lms
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --data_subdir boxes5_nso_exp2_max3_zero_shot \
    --prompt_mode chat --num_test_set_per_bin 300 --seed 0
```

**Clinical Sequence Modeling (MIMIC-IV)**
```bash
cd Tasks/Clinical_Sequence_Modeling
python MIMIC-IV/finetune_sequential.py \
    --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sequential_model --task mortality \
    --batch_size 4 --learning_rate 5e-5 --epochs 10 --use_lora
```

---

## Figures

All figures are in `Figs/`, organized into task subfolders. See
[`Figs/figure_descriptions.md`](Figs/figure_descriptions.md) for a description of every figure,
the model and configuration it represents, and the source script that generated it.

```
Figs/
├── In-Context-Retrieval/   ← colored objects + NIAH heatmaps
├── State-Tracking/         ← entity tracking bar/panel charts
├── Clinical-Sequence-Modeling/ ← AUROC-vs-sequence-length plots
└── Misc/                   ← reference figures from related work
```

---

## Pre-existing Results

This repo includes committed result files — it is not scaffolding only.

- **In-Context Retrieval**: 14 JSON output files + 3 SVG comparison plots in
  `Tasks/In-Context Retrieval/colored_objects/output/`
- **State Tracking**: 34 JSON result files across 5 box-count sweeps in
  `Tasks/State-Tracking/entity-tracking-lms/output/`

---

## Key Findings

From the saved OLMo results:

- At **0 operations** (pure retrieval, no state updates) the pure transformer leads clearly
  (set accuracy 0.953 vs 0.593 on the 6-box task).
- For **1+ operations** the hybrid leads across all operation counts, degrading far less as
  the sequence of state updates grows.

This is consistent with a division-of-labor hypothesis: linear/gated-recurrent attention
carries iterative state-update computation that standard MHA handles poorly under ablation.

---

## Suggested Reading Order

1. Read each task README for context on the benchmarks.
2. Inspect `Models/model_util.py` to understand the ablation primitives.
3. Open the committed JSON files in task `output/` directories to explore pre-computed results.
4. Extend with new model or ablation runs using the commands in each task README.

---

## Output Format

| Task | Output location | Format |
|------|----------------|--------|
| Colored Objects | `Tasks/In-Context Retrieval/colored_objects/output/` | JSON per `(model, subtype, prompt_mode, ablate_group)` |
| Entity Tracking | `Tasks/State-Tracking/entity-tracking-lms/output/` | JSON per `(model, dataset, prompt_mode)` |
| Clinical (MIMIC-IV) | configurable `--output_dir` | CSV + PNG per model |
| Clinical (eICU) | configurable `--output_dir` | CSV + PNG per model |

---

## Citation

```bibtex
@misc{hybrid-ablation-2026,
  title  = {Attention Ablation in Hybrid Language Models},
  year   = {2026},
}
```
