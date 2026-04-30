# State Tracking

This folder contains the **state-tracking** side of the project: experiments that test whether a language model can maintain and update a latent world state as objects are moved among boxes over time. In the broader project, this task family is used to study whether hybrid models organize stateful computation differently from pure transformers.

The benchmark asks a model to read a short scenario describing:
- the initial contents of several boxes
- a sequence of operations such as **put**, **remove**, and **move**
- a final query of the form: **“What does Box k contain?”**

The model must infer the final contents of the queried box after all updates.

## Folder layout

```text
State-Tracking/
├── README.md
└── entity-tracking-lms/
    ├── README.md
    ├── configs/
    │   └── boxes_pipeline.yaml
    ├── output/
    ├── run_evaluation.py
    ├── scripts/
    │   ├── data_generation/
    │   └── run_boxes_eval_pipeline.py
    ├── src/
    │   ├── boxes_lm_eval.py
    │   ├── dataset_generation/generate_boxes_data.py
    │   ├── evaluation/compute_metrics.py
    │   └── visualize_helpers.py
    └── visualize_results.ipynb
```

## What is included

This task folder is built around the `entity-tracking-lms/` benchmark code and includes:

- dataset generation for synthetic box-world scenarios
- evaluation code for Hugging Face causal LMs
- metrics for exact match and set match
- a YAML-based sweep pipeline
- many saved JSON outputs for OLMo and Qwen model families

## Task design

### World model
The synthetic world consists of:
- a fixed number of boxes
- a pool of objects
- optional limits on the number of items per box
- a sequence of sampled operations

The generator in `src/dataset_generation/generate_boxes_data.py` defines the main operations:
- `put`
- `remove`
- `move`

and stores scenarios in a zero-shot-friendly format.

### Query format
The evaluation code treats each example as a completion problem ending with a masked query like:

```text
Box 3 contains
```

The model must produce only the box contents, for example:
- `the apple`
- `the apple and the key`
- `nothing`

### Metrics
The task uses two primary metrics:

- **Exact accuracy**  
  The predicted phrase must match the gold completion exactly.

- **Set accuracy**  
  The predicted set of items must match the gold set, ignoring ordering / formatting differences such as item order.

This distinction matters because some generations are semantically correct even if wording is slightly reordered.

## Important files

### `run_evaluation.py`
Main entry point for evaluating a Hugging Face model on a JSONL split.

Key options include:
- `--model_name`
- `--split`
- `--prompt_mode` (`continuation` or `chat`)
- `--num_test_set_per_bin`
- `--dataset_dir` or `--data_file`

The script stratifies evaluation by the number of operations touching the queried box.

### `src/boxes_lm_eval.py`
Core evaluation logic:
- builds raw or chat prompts
- generates completions
- parses model outputs
- computes exact / set match
- aggregates accuracy by `numops`

### `src/dataset_generation/generate_boxes_data.py`
Synthetic data generator for the benchmark.  
This is the source of truth for:
- how initial world states are sampled
- how operations are sampled and applied
- how box descriptions are phrased

### `scripts/run_boxes_eval_pipeline.py`
Convenience pipeline that:
1. generates missing datasets
2. expands YAML sweeps
3. runs evaluations across all requested configurations

This is the easiest way to reproduce the box-count sweeps already present in `output/`.

## Running the task

### Evaluate a model on a dataset
```bash
cd "Tasks/State-Tracking/entity-tracking-lms"

python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --split test   --prompt_mode chat   --num_test_set_per_bin 300   --seed 0
```

### Evaluate a custom dataset directory
```bash
python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --dataset_dir data/boxes5_exp2_max3_nops12_zero_shot   --prompt_mode chat
```

### Run the generation + evaluation sweep pipeline
```bash
python scripts/run_boxes_eval_pipeline.py --config configs/boxes_pipeline.yaml
```

The included example config sweeps:
- `num_boxes` over `[2, 3, 4, 5, 6]`
- selected Qwen models
- shared prompt/eval settings

## Saved results and what they show

The `output/` directory already contains 34 JSON result files, including:
- `def_ds/` — default OLMo baseline comparison
- `boxes2_exp2_max3_nops12_zero_shot/`
- `boxes3_exp2_max3_nops12_zero_shot/`
- `boxes4_exp2_max3_nops12_zero_shot/`
- `boxes5_exp2_max3_nops12_zero_shot/`
- `boxes6_exp2_max3_nops12_zero_shot/`

### Main empirical pattern in the OLMo pair
The saved `def_ds/` chat runs reproduce one of the clearest project findings:

- For **0 operations**, the pure transformer is much stronger  
  (`set_acc ≈ 0.887` vs `0.407` for the hybrid)

- But once actual state updates are required, the **hybrid is stronger from 1 through 5 operations**  
  Example set accuracy by number of operations:
  - 1 op: hybrid `≈ 0.343` vs pure `≈ 0.190`
  - 2 ops: hybrid `≈ 0.360` vs pure `≈ 0.157`
  - 3 ops: hybrid `≈ 0.327` vs pure `≈ 0.087`
  - 4 ops: hybrid `≈ 0.237` vs pure `≈ 0.067`
  - 5 ops: hybrid `≈ 0.263` vs pure `≈ 0.087`

This is exactly the sort of tradeoff that makes the benchmark useful for a “division of labor” study: one model can look much better on trivial retrieval while another is more robust once iterative state updates accumulate.

### Scaling across number of boxes
Across the saved `boxes2`–`boxes6` datasets, the hybrid OLMo model consistently outperforms the pure OLMo transformer in set accuracy:

| Number of boxes | Pure OLMo set acc | Hybrid OLMo set acc |
|---|---:|---:|
| 2 | 0.356 | 0.488 |
| 3 | 0.304 | 0.404 |
| 4 | 0.280 | 0.363 |
| 5 | 0.272 | 0.357 |
| 6 | 0.268 | 0.333 |

The gap narrows slightly as the task gets harder, but it remains in the hybrid’s favor throughout the saved sweep.

### Other included model families
The saved outputs also include Qwen-based runs. Among the committed results, the strongest state-tracking numbers come from the larger Qwen-family models, especially `Qwen/Qwen3.5-27B`, which stays well above the OLMo models across the box-count sweep.

## Relation to the project figures

The provided figures summarize this task family in two especially useful ways:

- a **number-of-operations** comparison, highlighting that the hybrid is weaker at 0-op lookup but better once updates accumulate
- a **number-of-boxes** comparison across OLMo and Qwen families, showing how performance changes as the latent state becomes larger

Those plots are a good visual companion to the JSON files in `output/`.

## Notes on ablations

Unlike `Tasks/In-Context Retrieval/colored_objects/`, this task does **not** currently expose ablation flags directly from the CLI. However, the repo’s shared ablation utilities still exist in:

```text
Models/model_util.py
```

So if you want to run paper-style layer or attention-group interventions on state tracking, the most natural extension point is:
- load the model through the shared utilities
- apply `ablate_groups(...)` or `ablate_single_attn_module(...)`
- then call the existing evaluation loop in `src/boxes_lm_eval.py`

## Practical advice

If you are trying to understand or extend this task, the best order is:

1. Read `entity-tracking-lms/run_evaluation.py`
2. Read `src/boxes_lm_eval.py`
3. Read `src/dataset_generation/generate_boxes_data.py`
4. Inspect one of the saved JSON files in `output/`
5. Use `scripts/run_boxes_eval_pipeline.py` for new sweeps

That will give you the data format, prompting logic, metrics, and reproduction path in the smallest number of files.
