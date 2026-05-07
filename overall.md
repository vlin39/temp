# Attention Ablation in Hybrid Language Models — Project Overview

This document is the consolidated guide to the project. It integrates the
top-level project description and every task-level README, organised so a new
collaborator can read top-to-bottom and reproduce any result. Individual task
READMEs (linked at each section) remain authoritative for their specific
tasks; this document is a navigable summary.

---

## 1. Research Question

Where is task-relevant computation localised in **hybrid language models** —
models that interleave standard multi-head self-attention (MHA) with
linear / gated-recurrent attention layers — and how does this differ from
matched pure-transformer baselines?

**Methodology.** For a loaded model, monkey-patch one attention family out
(replace its `forward` with an identity pass), measure task degradation,
restore, and repeat for the other family. This isolates the contribution of
each component type without retraining.

**Primary model pair**

| | HuggingFace ID | Type |
|---|---|---|
| Pure | `allenai/Olmo-3-7B-Instruct-SFT` | 32-layer transformer |
| Hybrid | `allenai/Olmo-Hybrid-Instruct-SFT-7B` | MHA + gated-recurrent interleave |

Extended comparisons use Qwen3 (pure) / Qwen3.5 (hybrid) at 8B/9B and 27B/32B.

---

## 2. Repository Layout

```
.
├── overall.md                     ← this file
├── README.md                      ← top-level project README
├── Env/                           ← conda/pip environment specs
├── Models/
│   └── model_util.py              ← model loading + ablation primitives
├── Figs/                          ← all rendered figures, organised by task
│   ├── In-Context-Retrieval/
│   ├── State-Tracking/
│   ├── Clinical-Sequence-Modeling/
│   └── figure_descriptions.md
└── Tasks/
    ├── In-Context Retrieval/
    │   ├── colored_objects/       ← attribute Q&A over scenes
    │   └── NIAH/                  ← needle-in-a-haystack
    ├── State-Tracking/
    │   ├── entity-tracking-lms/   ← box contents over operations
    │   ├── web-of-lies/           ← boolean chain truth propagation
    │   └── dyck-languages/        ← typed-stack bracket completion
    └── Clinical_Sequence_Modeling/
        ├── MIMIC-IV/              ← mortality + phenotyping
        └── eICU/                  ← mortality + length-of-stay
```

---

## 3. Ablation Primitives

Defined in `Models/model_util.py`. The same primitives back every task.

```python
from Models.model_util import (
    load_model_and_tokenizer,
    ablate_groups,
    ablate_single_attn_module,
    list_decoder_attention_layer_indices,
    restore_attention_ablation,
)

model, tokenizer = load_model_and_tokenizer("allenai/Olmo-Hybrid-Instruct-SFT-7B")

# Group ablation — zero out one attention family across all decoder layers
ablate_groups(model, "self_attn")     # zero MHA blocks
ablate_groups(model, "linear_attn")   # zero linear/gated-recurrent blocks
ablate_groups(model, "none")          # restore

# Layerwise ablation — zero out the attention block in one specific decoder layer
ablate_single_attn_module(model, layer_idx=12)
restore_attention_ablation(model)
```

Matching is by submodule name (`.self_attn` vs `.linear_attn` last segment)
and works for OLMo Hybrid, Qwen3-Next, and Qwen3.5 hybrid stacks.

---

## 4. Models

| Key (in `MODEL_IDS`) | HuggingFace ID | Family |
|-----|----------------|------|
| `olmo3_7b_instruct` | `allenai/Olmo-3-7B-Instruct-SFT` | Pure |
| `olmo_hybrid_7b_instruct` | `allenai/Olmo-Hybrid-Instruct-SFT-7B` | Hybrid |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Pure |
| `qwen3-32b` | `Qwen/Qwen3-32B` | Pure |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | Hybrid |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | Hybrid |

```python
from Models.model_util import MODEL_IDS  # dict[str, str]
```

---

## 5. Environment

`Env/environment.yml` and `Env/requirements.txt` are broader snapshots from
a previous project — they install correctly but include packages this
project doesn't use. The minimal viable install is:

```bash
conda create -n dol python=3.10 -y
conda activate dol
pip install torch transformers datasets numpy tqdm accelerate
```

Or use the full environment file:

```bash
conda env create -f Env/environment.yml
conda activate dol
```

NIAH has its own lighter dependency set in
`Tasks/In-Context Retrieval/NIAH/requirements.txt`.

---

## 6. Tasks

Three task families, each summarised below. Each subsection links to the
canonical task README.

### 6.1 In-Context Retrieval

> Source: [`Tasks/In-Context Retrieval/README.md`](Tasks/In-Context%20Retrieval/README.md)

Two benchmarks probing retrieval and shallow reasoning over in-context
information. No state propagation required — the answer is *somewhere in
the prompt* and the model needs to find and use it.

#### Colored Objects

[Task README](Tasks/In-Context%20Retrieval/colored_objects/README.md)

The model is asked questions about the color, position, or count of objects
embedded in a natural-language scene. Five subtypes (`what_color`,
`yes_no_color`, `neither_color`, `spatial`, `arithmetic`); difficulty
controlled by `num_objects` ∈ {2, 3, 4, 5, 6}.

```bash
cd "Tasks/In-Context Retrieval/colored_objects"
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color --prompt_mode chat --num_test_set 250 --seed 0

# Full sweep
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sweep_ablate_groups --sweep_subtypes --sweep_prompt_modes --num_test_set 250
```

Output: `output/<model_slug>_<subtype>_<prompt_mode>[_<ablate_group>].json`.
Pre-computed JSONs and SVG comparison plots are committed.

#### NIAH

[Task README](Tasks/In-Context%20Retrieval/NIAH/README.md)

Needle-in-a-haystack at controlled depths (0–100%) and context lengths
(1K–65K tokens). Uses the upstream `needlehaystack` benchmark package
(API-based; local ablation is not applied). Project NIAH results are
rendered as heatmaps in `Figs/In-Context-Retrieval/fig_niah_*.jpeg`.

---

### 6.2 State Tracking

The central group of tasks. Each probes a different *state structure*:

| Task | State | Update operation | Difficulty axis |
|---|---|---|---|
| `entity-tracking-lms` | Set of items per box | put / move / remove | numops, num_boxes |
| `web-of-lies` | Single boolean | "X says Y lies/tells the truth" | chain depth |
| `dyck-languages` | Typed stack | bracket open / close | nesting depth, target len |

The pure transformer should keep up with the hybrid on retrieval-heavy
tasks; the hybrid should pull ahead on tasks that require iterative state
updates over many steps. The ablation comparison localises *where* that
state-tracking computation lives.

#### Entity Tracking (boxes)

[Task README](Tasks/State-Tracking/entity-tracking-lms/README.md)

Track box contents through a sequence of put / move / remove operations.
Adapted from Kim & Schuster (ACL 2023). Two metrics per run, stratified by
`numops` (0–5):

- **`exact_accuracy`** — predicted list matches gold list exactly
- **`set_accuracy`** — predicted set matches gold set (order-insensitive)

```bash
cd Tasks/State-Tracking/entity-tracking-lms
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --data_subdir boxes6_exp2_max3_nops12_zero_shot \
    --prompt_mode chat --num_test_set_per_bin 300 --seed 0
```

Pipeline sweep over box counts and models (Cartesian product) is via
`scripts/run_boxes_eval_pipeline.py` with a YAML config.

**Pre-computed key result** (`boxes6`, OLMo pair, chat, set accuracy):

| numops | OLMo-Hybrid | OLMo-3 (pure) |
|---|---|---|
| 0 | 0.593 | **0.953** |
| 1 | **0.447** | 0.160 |
| 2 | **0.300** | 0.197 |
| 3 | **0.233** | 0.137 |
| 4 | **0.217** | 0.063 |
| 5 | **0.210** | 0.097 |

The crossover at `numops ≥ 1` is the central state-tracking finding: pure
transformers are sharper at zero-update retrieval; hybrids degrade more
gracefully as state updates accumulate.

#### Web of Lies

[Task README](Tasks/State-Tracking/web-of-lies/README.md)

BBH boolean-chain task. Each example: "Yang tells the truth. Phoebe says
Yang lies. Sherrie says Phoebe lies. … Does Sue tell the truth?" The state
is a single boolean propagated through 5 statements. Binary `yes` / `no`.

```bash
cd Tasks/State-Tracking/web-of-lies
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat --sweep_ablate_groups
```

Adds `--sweep_layer_indices` for per-layer ablation and a paired-prompt
activation-patching script (`make_patching_pairs.py` + `run_patching.py`).
The corrupt prompt flips the first statement's truth verb, which
deterministically flips the final answer.

#### Dyck Languages

[Task README](Tasks/State-Tracking/dyck-languages/README.md)

BBH Dyck-4 completion task. The model is given an incomplete bracket
sequence (e.g. `"[ ( { < ( ) > }"`) and must produce the closing bracket
sequence (e.g. `"} ]"`). Recognising Dyck-n for n>1 is provably equivalent
to operating a typed stack — the sharpest state-tracking probe of the three.

Two metrics: **`exact_accuracy`** (token-for-token match) and
**`token_recall`** (longest matching prefix / target length). Difficulty
axes: `target_len` (number of closes needed) and `max_open_depth` (peak
nesting depth in the input).

```bash
cd Tasks/State-Tracking/dyck-languages
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat --sweep_ablate_groups
```

Same `--sweep_layer_indices` and patching pipeline as web-of-lies. The
patching corruption changes one bracket in the prefix, chosen so that the
required closing sequence becomes provably different from the clean target.

---

### 6.3 Clinical Sequence Modeling

[Task README](Tasks/Clinical_Sequence_Modeling/README.md)

Trains a **Belief Update Transformer** on sequential ICU events to predict
in-hospital mortality at any point during a stay. Two datasets:

| Dataset | Tasks | Modalities |
|---|---|---|
| MIMIC-IV | mortality, phenotyping | CXR images, radiology notes, discharge summaries, vitals/labs, demographics |
| eICU | mortality, length-of-stay | demographics, diagnosis, treatment, medication, lab, APS, vitals |

Both require institutional data access (PhysioNet); the repo ships training
code, not data.

```bash
cd Tasks/Clinical_Sequence_Modeling
python MIMIC-IV/finetune_sequential.py \
    --llm_model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sequential_model --task mortality \
    --batch_size 4 --learning_rate 5e-5 --epochs 10 --use_lora
```

---

## 7. Common Analysis Pipeline

The state-tracking tasks share a three-tier ablation analysis (used in
`web-of-lies` and `dyck-languages`; see the `colored_objects` and
`entity-tracking-lms` READMEs for their task-specific equivalents).

### Tier 1 — Group ablation
Zero out all `self_attn` or all `linear_attn` blocks; compare degradation
to the unablated baseline.

```bash
python run_evaluation.py --model_name <ID> --prompt_mode chat --sweep_ablate_groups
```

### Tier 2 — Layerwise ablation
Zero out one decoder layer at a time, sweeping all 32 layers. Reveals
positional gradient (early-layer vs. late-layer importance) and whether
critical layers cluster by attention type in hybrids.

```bash
python run_evaluation.py --model_name <ID> --prompt_mode chat --sweep_layer_indices
```

### Tier 3 — Activation patching
Generate clean/corrupt paired prompts that differ by minimal information
content. For each layer L, patch the clean residual into the corrupt
forward pass and measure recovery of the correct answer logit.

```bash
python make_patching_pairs.py --num_pairs 200 --seed 0
python run_patching.py --model_name <ID> --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json
```

### Visualisation
Each task ships a `make_comparison_plot.py` that renders SVG figures from
the JSON outputs (no GPU needed — pure stdlib, runs on a login node):

- Baseline overall + breakdown by task-specific difficulty axis
- Group ablation comparison (4 models × 3 ablation settings)
- Layerwise ablation (2×2 subplot grid; separate MHA/SSM curves for hybrids)
- Activation patching effect (mean logit-diff by layer)

Outputs land in each task's `output/*.svg` and copies are placed in
`Figs/State-Tracking/fig_*.svg`.

---

## 8. Running on OSCAR (Brown HPC)

```bash
# 1. Pull the branch
cd /path/to/repo
git fetch origin
git checkout claude/add-readme-documentation-uogMw && git pull

# 2. Activate or create the env (one-time)
micromamba activate <existing-env>           # if you already have one
# or:
micromamba create -n dol python=3.10 -y && micromamba activate dol
pip install torch transformers datasets numpy tqdm accelerate

# 3. Get a GPU node
salloc --partition=gpu --gres=gpu:1 --mem=40G --cpus-per-task=4 --time=4:00:00

# 4. Run the eval (example: web-of-lies, OLMo-Hybrid, all ablations)
cd Tasks/State-Tracking/web-of-lies
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat --sweep_ablate_groups

# 5. Render figures (no GPU needed; can be on the login node)
python make_comparison_plot.py --prompt_mode chat --skip_layerwise --skip_patching
```

A 200-example run with 3 ablation settings on a single A100 takes about
30–60 min per model.

---

## 9. Outputs and Figures

| Task | Output dir | Format |
|---|---|---|
| Colored Objects | `Tasks/In-Context Retrieval/colored_objects/output/` | JSON per `(model, subtype, prompt_mode, ablate_group)` |
| Entity Tracking | `Tasks/State-Tracking/entity-tracking-lms/output/` | JSON per `(model, dataset, prompt_mode)` |
| Web of Lies | `Tasks/State-Tracking/web-of-lies/output/` | JSON per `(model, prompt_mode, ablate_group OR layer)` |
| Dyck Languages | `Tasks/State-Tracking/dyck-languages/output/` | Same schema as web-of-lies |
| Clinical (MIMIC-IV) | `--output_dir` | CSV + PNG |
| Clinical (eICU) | `--output_dir` | CSV + PNG |

`Figs/figure_descriptions.md` documents every committed figure with
generating script and configuration.

---

## 10. Reading Order for New Collaborators

1. **This file (`overall.md`)** — high-level orientation.
2. **`Models/model_util.py`** — read end-to-end (it's short). The ablation
   primitives are the project's core abstraction.
3. Pick **one task README** matching your interest and run a single
   evaluation end-to-end before scaling up:
   - Mechanistic interpretability angle → `web-of-lies` or `dyck-languages`
   - Empirical state-tracking comparison → `entity-tracking-lms`
   - Retrieval baseline → `colored_objects`
4. **`Tasks/<your_task>/output/`** — open committed JSON files; they're
   the ground truth for what a successful run looks like.
5. Reproduce the ablation sweep for your task; render figures with
   `make_comparison_plot.py`.

---

## 11. Citation

```bibtex
@misc{hybrid-ablation-2026,
  title = {Attention Ablation in Hybrid Language Models},
  year  = {2026},
}
```

Per-task source citations (Kim & Schuster 2023 for boxes; Suzgun et al. 2022
for BBH; Merrill & Sabharwal 2024 for the Dyck/SSM theoretical motivation)
are inside the task READMEs.
