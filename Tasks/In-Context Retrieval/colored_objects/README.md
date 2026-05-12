# Colored Objects

Models are asked questions about the color (or spatial/arithmetic properties) of objects embedded
in a natural-language scene. Task difficulty is controlled by `num_objects` (2–6 distractors) and
five `subtype` variants:

| Subtype | Question type |
|---------|---------------|
| `what_color` | "What color is the X?" |
| `yes_no_color` | "Is the X [color]?" (yes/no) |
| `neither_color` | Multi-choice including "neither" option |
| `spatial` | Relative spatial position |
| `arithmetic` | Count or sum across objects |

Task examples are loaded from `task.json`. Results are stratified by `num_objects` and written
to `output/`.

---

## Single Evaluation

```bash
cd "Tasks/In-Context Retrieval/colored_objects"

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color \
    --prompt_mode chat \
    --num_test_set 250 \
    --seed 0
```

## Ablation Sweep

```bash
# Ablate one attention group
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color \
    --prompt_mode chat \
    --ablate_group self_attn \
    --num_test_set 250

# Full sweep: all ablation groups × all subtypes × all prompt modes
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --sweep_ablate_groups \
    --sweep_subtypes \
    --sweep_prompt_modes \
    --num_test_set 250
```

> **Note:** `--num_test_set` must be divisible by the number of available `num_objects` bins for
> the selected subtype. The script will report a clear error if this constraint is violated.

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | *(required)* | HuggingFace model ID or key from `Models/model_util.MODEL_IDS` |
| `--subtype` | `what_color` | Task subtype filter (`what_color`, `yes_no_color`, `neither_color`, `spatial`, `arithmetic`) |
| `--sweep_subtypes` | off | Run all available subtypes instead of a single one |
| `--prompt_mode` | `continuation` | `continuation`, `chat`, or `chat_continual` |
| `--sweep_prompt_modes` | off | Run all three prompt modes |
| `--num_test_set` | `100` | Total evaluated examples (must be divisible by available `num_objects` bins) |
| `--ablate_group` | `none` | Attention ablation: `none`, `self_attn`, or `linear_attn` |
| `--sweep_ablate_groups` | off | Run `none`, `self_attn`, and `linear_attn` ablations |
| `--allowed_num_objects` | `None` | Comma-separated bins to restrict evaluation (e.g. `3,4,5`) |
| `--seed` | `0` | Random seed for stratified sampling |
| `--max_new_tokens` | `16` | Maximum tokens to generate per example |
| `--task_json` | `task.json` | Path to the task examples file |
| `--output_json` | auto | Override output path (only valid for a single configuration) |

---

## Saved Results

Pre-computed results for OLMo 7B models (chat mode) are in `output/`. Overall accuracy by
subtype:

| Subtype | OLMo-Hybrid-7B | OLMo-3-7B (pure) |
|---------|---------------|------------------|
| `what_color` | 0.988 | 0.988 |
| `yes_no_color` | 0.994 | 0.931 |
| `neither_color` | 0.353 | 0.180 |
| `spatial` | 0.478 | 0.310 |
| `arithmetic` | 0.306 | 0.389 |

`arithmetic` is the one subtype where the pure transformer leads. Qwen3-8B and Qwen3.5-9B
achieve near-perfect accuracy (1.0) on `what_color` and `yes_no_color`.

---

## Visualization

```bash
python make_comparison_plot.py
```

Generates SVG charts in `output/` comparing Pure Transformer vs. Hybrid accuracy by
`num_objects` across subtypes. Pre-rendered copies are in `Figs/In-Context-Retrieval/`.

---

## Output JSON Structure

```json
{
  "model_name": "allenai/Olmo-Hybrid-Instruct-SFT-7B",
  "subtype": "what_color",
  "prompt_mode": "chat",
  "ablate_group": "none",
  "ablation_modules": [],
  "num_test_set": 250,
  "n_per_num_objects_bin": 50,
  "available_num_objects_bins": [2, 3, 4, 5, 6],
  "overall": {"n": 250, "accuracy": 0.812},
  "by_num_objects": {
    "2": {"n": 50, "accuracy": 0.94},
    "6": {"n": 50, "accuracy": 0.62}
  },
  "by_subtype": {"what_color": {"n": 250, "accuracy": 0.812}},
  "rows": [...]
}
```
