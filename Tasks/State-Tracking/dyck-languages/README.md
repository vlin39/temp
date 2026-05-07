# Dyck Languages (BBH)

Models are evaluated on their ability to recognise balanced bracket sequences — a
canonical test of stack-like state tracking. Example:

> "Is the following sequence valid? [ ( { < > } ) ]"

The model must answer **yes** (balanced) or **no** (unbalanced). Correct classification
requires tracking an implicit stack: each opening bracket pushes a frame, each closing
bracket pops and verifies the match. Errors anywhere in the chain produce an incorrect
answer.

This complements `web-of-lies` (boolean chain propagation) and `entity-tracking-lms`
(set-update tracking) by probing a different state structure — a depth-indexed stack
rather than a single boolean or entity set.

Source: BIG-Bench Hard `dyck_languages` (Suzgun et al. 2022) via `lukaemon/bbh` on
HuggingFace.

---

## Dataset

- **Source:** `datasets.load_dataset("lukaemon/bbh", "dyck_languages")`
- **Labels:** binary `yes` / `no`
- **Bracket types:** up to four — `()`, `[]`, `{}`, `<>`
- **Difficulty metrics** stored per example:
  - `seq_len` — total number of bracket tokens
  - `max_depth` — maximum nesting depth reached

---

## Single Evaluation

```bash
cd Tasks/State-Tracking/dyck-languages

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --num_test_set 200 \
    --seed 0
```

## Ablation Sweep

```bash
# All three ablation groups
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --sweep_ablate_groups
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | *(required)* | HuggingFace model ID or key from `Models.model_util.MODEL_IDS` |
| `--prompt_mode` | `continuation` | `continuation`, `chat`, or `chat_continual` |
| `--sweep_prompt_modes` | off | Run all three prompt modes |
| `--num_test_set` | `200` | Total examples; must be even |
| `--ablate_group` | `none` | `none`, `self_attn`, or `linear_attn` |
| `--sweep_ablate_groups` | off | Run all three ablation settings |
| `--sweep_layer_indices` | off | Ablate one decoder layer at a time |
| `--seed` | `0` | RNG seed |
| `--cache_dir` | `data/` | BBH dataset cache directory |

---

## Output JSON Structure

```json
{
  "model_name": "allenai/Olmo-Hybrid-Instruct-SFT-7B",
  "task": "dyck_languages",
  "prompt_mode": "chat",
  "ablate_group": "none",
  "overall": {"n": 200, "accuracy": 0.71},
  "by_answer": {
    "yes": {"n": 100, "accuracy": 0.80},
    "no":  {"n": 100, "accuracy": 0.62}
  },
  "by_depth":   {"1": {...}, "2": {...}, "3": {...}},
  "by_seq_len": {"4": {...}, "6": {...}, ...},
  "rows": [...]
}
```

Results are written to `output/<model_slug>_<prompt_mode>[_<ablate_group>].json`.

---

## Visualization

After running evaluations, render all comparison figures:

```bash
python make_comparison_plot.py --prompt_mode chat
```

Skip figures that require data not yet collected:

```bash
python make_comparison_plot.py --prompt_mode chat --skip_layerwise --skip_patching
```

| Output file | Contents |
|---|---|
| `output/dyck_languages_accuracy_by_answer.svg` | Baseline accuracy, split by answer label |
| `output/dyck_languages_ablation_comparison.svg` | Accuracy under each ablation group |
| `output/dyck_languages_accuracy_by_depth.svg` | Accuracy vs max nesting depth (dyck-specific) |
| `output/dyck_languages_layerwise_ablation.svg` | Per-layer ablation, 2×2 subplot grid |
| `output/dyck_languages_patching_effect.svg` | Activation patching effect by layer |
| `output/dyck_languages_summary.md` | Markdown table of all numbers |

SVGs are also copied to `Figs/State-Tracking/fig_dyck_languages_*.svg`.

---

## Layerwise Ablation Sweep

```bash
python run_evaluation.py \
    --model_name allenai/Olmo-3-7B-Instruct-SFT \
    --prompt_mode chat --sweep_layer_indices

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat --sweep_layer_indices
```

---

## Activation Patching

```bash
# Step 1: corrupt a valid sequence by flipping one bracket
python make_patching_pairs.py --num_pairs 200 --seed 0

# Step 2: sweep all layers for each model
python run_patching.py \
    --model_name allenai/Olmo-3-7B-Instruct-SFT \
    --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json

python run_patching.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json
```

The corrupt prompt is produced by flipping one bracket near the middle of a valid
sequence, changing a single character to make the sequence unbalanced. The patching
script measures how much each layer's clean residual recovers the correct answer.

---

## Saved Results

*(Populated after first run. The hypothesis: hybrid models should show greater degradation
under `linear_attn` ablation on deep sequences, since maintaining a stack across many
tokens favours recurrent state.)*

---

## Citation

```bibtex
@article{suzgun2022challenging,
  title={Challenging BIG-bench tasks and whether chain-of-thought can solve them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian
          and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V
          and Chi, Ed H and Zhou, Denny and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
```
