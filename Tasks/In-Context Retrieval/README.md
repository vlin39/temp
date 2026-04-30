# In-Context Retrieval

Two benchmarks that probe how well models retrieve and reason over information stored in context:

- **Colored Objects** — multi-step reasoning with object attributes at varying distractors counts
- **NIAH** — Needle-in-a-Haystack long-context retrieval at controlled depths and context lengths

---

## Colored Objects

### Overview

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

Task examples are loaded from `colored_objects/task.json`. Results are stratified by `num_objects`
and written to `colored_objects/output/`.

### Single Evaluation

```bash
cd "Tasks/In-Context Retrieval/colored_objects"

python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color \
    --prompt_mode chat \
    --num_test_set 250 \
    --seed 0
```

### Ablation Sweep

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

### Key Arguments

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

### Visualization

```bash
python make_comparison_plot.py
```

Generates SVG charts in `output/` comparing Pure Transformer vs. Hybrid accuracy by `num_objects`
across subtypes. Copies are also stored in `Figs/In-Context-Retrieval/`.

### Output JSON Structure

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

---

## NIAH (Needle-in-a-Haystack)

### Overview

A single "needle" fact is embedded at a controlled depth inside a long context built from Paul
Graham essays. The model is asked to retrieve it. This benchmark stresses long-context memory at
context lengths from 1K to 32K (or 65K) tokens and depths from 0 % (start) to 100 % (end).

> Built on the upstream [`needlehaystack`](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
> package. The NIAH task uses API calls to hosted models; local ablation (patching attention
> modules) is not applied here.

### Setup

```bash
pip install -r "Tasks/In-Context Retrieval/NIAH/requirements.txt"

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

### Commands

```bash
cd "Tasks/In-Context Retrieval/NIAH"

# Quick single-point test (OpenAI)
python needlehaystack/run.py \
    --provider OpenAI \
    --model_name gpt-4o \
    --context_lengths_min 1000 \
    --context_lengths_max 32000 \
    --context_lengths_num_intervals 10 \
    --document_depth_percents_num_intervals 10

# Anthropic
python needlehaystack/run.py \
    --provider Anthropic \
    --model_name claude-sonnet-4-6

# Cohere
python needlehaystack/run.py \
    --provider Cohere \
    --model_name command-r-plus
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | `OpenAI` | API provider: `OpenAI`, `Anthropic`, `Cohere` |
| `--model_name` | `gpt-4o` | Model ID for the chosen provider |
| `--context_lengths_min` | `1000` | Minimum context length in tokens |
| `--context_lengths_max` | `128000` | Maximum context length in tokens |
| `--context_lengths_num_intervals` | `35` | Number of context-length steps |
| `--document_depth_percents_min` | `0` | Minimum needle depth (percent of context) |
| `--document_depth_percents_max` | `100` | Maximum needle depth |
| `--document_depth_percents_num_intervals` | `35` | Number of depth steps |
| `--num_concurrent_requests` | `1` | Parallel API requests |
| `--save_results` | `True` | Write results to JSON |
| `--save_contexts` | `True` | Save the generated context strings |

### Visualization

Open `viz/CreateVizFromLLMTesting.ipynb` in Jupyter. The notebook reads saved JSON results and
renders the retrieval-score heatmaps. Pre-rendered heatmaps are stored in
`Figs/In-Context-Retrieval/fig_niah_*.jpeg`.
