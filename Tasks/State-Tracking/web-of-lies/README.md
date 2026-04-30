# Web of Lies (BBH)

Models are evaluated on their ability to carry a single boolean truth-value through a chain of
meta-statements. Example:

> "Yang tells the truth. Phoebe says Yang lies. Sherrie says Phoebe lies.
> Mateo says Sherrie tells the truth. Sue says Mateo lies.
> Does Sue tell the truth?"

The model must answer **yes** or **no**. The "state" is each person's truth-bit, derived
recursively from the previous statement. This complements the boxes task (`entity-tracking-lms/`)
by exercising state propagation under negation rather than under set updates.

Source: BIG-Bench Hard `web_of_lies` (Suzgun et al. 2022) via `lukaemon/bbh` on HuggingFace.

---

## Dataset

- **Source:** `datasets.load_dataset("lukaemon/bbh", "web_of_lies")`
- **Size:** ~250 examples; all are 5-step chains; binary `yes` / `no` labels.
- **Cache:** downloaded once into `data/` on first run (configurable via `--cache_dir`).

---

## Single Evaluation

```bash
cd Tasks/State-Tracking/web-of-lies

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --num_test_set 200 \
    --seed 0
```

## Ablation Sweep

```bash
# Single ablation group
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --ablate_group self_attn

# Full sweep: all three ablation groups × all three prompt modes
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sweep_ablate_groups \
    --sweep_prompt_modes
```

> `--num_test_set` must be a positive even number (split equally into yes/no answer bins).

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | *(required)* | HuggingFace model ID or key from `Models.model_util.MODEL_IDS` |
| `--prompt_mode` | `continuation` | `continuation`, `chat`, or `chat_continual` |
| `--sweep_prompt_modes` | off | Run all three prompt modes |
| `--num_test_set` | `200` | Total examples evaluated; must be even |
| `--ablate_group` | `none` | `none`, `self_attn`, or `linear_attn` |
| `--sweep_ablate_groups` | off | Run all three ablation settings |
| `--seed` | `0` | RNG seed for stratified sampling |
| `--max_new_tokens` | `8` | Generation budget (`yes`/`no` + punctuation) |
| `--cache_dir` | `data/` | Directory to cache the BBH dataset download |
| `--output_json` | auto | Override output path (single configuration only) |

---

## Output JSON Structure

```json
{
  "model_name": "allenai/Olmo-Hybrid-Instruct-SFT-7B",
  "task": "web_of_lies",
  "num_test_set": 200,
  "n_per_answer_bin": 100,
  "seed": 0,
  "max_new_tokens": 8,
  "prompt_mode": "chat",
  "ablate_group": "none",
  "ablation_modules": [],
  "pool_sizes_by_answer": {"yes": 125, "no": 125},
  "available_chain_depths": [5],
  "overall": {"n": 200, "accuracy": 0.74},
  "by_answer": {
    "yes": {"n": 100, "accuracy": 0.79},
    "no":  {"n": 100, "accuracy": 0.69}
  },
  "by_chain_depth": {"5": {"n": 200, "accuracy": 0.74}},
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

To skip figures that require data not yet collected:

```bash
python make_comparison_plot.py --prompt_mode chat --skip_layerwise --skip_patching
```

| Output file | Contents |
|---|---|
| `output/web_of_lies_accuracy_by_answer.svg` | Baseline accuracy (pure vs. hybrid), split by answer label |
| `output/web_of_lies_ablation_comparison.svg` | Accuracy under `none` / `self_attn` / `linear_attn` ablation, per model |
| `output/web_of_lies_layerwise_ablation.svg` | Accuracy when a single layer is ablated, plotted by layer index |
| `output/web_of_lies_patching_effect.svg` | Mean logit-diff from residual-stream activation patching, by layer |
| `output/web_of_lies_summary.md` | Markdown table of all numbers in Figures 1–2 |

SVGs are also copied to `Figs/State-Tracking/fig_web_of_lies_*.svg`.

Colors follow `colored_objects/make_comparison_plot.py` (`#2ca02c` pure, `#d96bc0` hybrid).
Hybrid bars are diagonally hatched for readability under color-vision deficiency and in monochrome.

---

## Layerwise Ablation Sweep

To ablate one decoder layer at a time and produce the data for Figure 3:

```bash
python run_evaluation.py \
    --model_name allenai/Olmo-3-7B-Instruct-SFT \
    --prompt_mode chat \
    --sweep_layer_indices

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --sweep_layer_indices
```

Each run writes one JSON per layer: `output/<slug>_<mode>_layer<N>.json`. The JSON schema matches the group-ablation output with additional fields `ablate_mode`, `layer_idx`, and `layer_type` (`self_attn` or `linear_attn`).

---

## Activation Patching

To generate paired prompts and run the patching experiment (Figure 4):

```bash
# Step 1: create paired clean/corrupt prompts
python make_patching_pairs.py --num_pairs 200 --seed 0

# Step 2: run patching for each model
python run_patching.py \
    --model_name allenai/Olmo-3-7B-Instruct-SFT \
    --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json

python run_patching.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json
```

Each corrupt prompt is produced by flipping the first statement's truth value (e.g. "Yang tells the truth" → "Yang lies"), which deterministically flips the final answer. The patching script sweeps all decoder layers and reports `mean_logit_diff` (how much the clean activation at layer L recovers the correct answer) and `patched_accuracy`.

---

## Saved Results

*(Populated after first run. Expected to mirror the boxes pattern: the hybrid model's linear
attention layers should carry iterative state-propagation, showing greater degradation under
`linear_attn` ablation relative to `self_attn` ablation.)*

---

## Phase 2 (deferred) — Synthetic Depth Sweep

BBH `web_of_lies` has fixed depth 5, which prevents a depth-vs-accuracy crossover plot
analogous to the boxes `numops` sweep. A generator
(`src/dataset_generation/generate_web_of_lies.py`) producing chains of configurable depth
(3 / 5 / 7 / 9 / 11) would unlock that analysis. This is out of scope for the current phase
and will be added once the Phase 1 pipeline is validated.

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
