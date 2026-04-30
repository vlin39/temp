# Entity Tracking

Models are evaluated on their ability to track the contents of a set of boxes through a sequence
of *put*, *remove*, and *move* operations. Given a narrative like

> "Box A contains a pen. Move the pen from Box A to Box B. Box B contains a ball.
> Box B contains:"

the model must complete the statement with the correct current contents of the box.

Two accuracy metrics are reported per run, both stratified by `numops` (0–5 operations):

- **`exact_accuracy`** — the predicted list matches the gold list exactly (order-sensitive)
- **`set_accuracy`** — the predicted set of items matches the gold set (order-insensitive)

Adapted from Kim & Schuster, ACL 2023.

---

## Dataset

The dataset is provided as a password-protected ZIP file to prevent leakage into the training data
of future language models. Please do not include the uncompressed files in any repositories if you
use the data.

- Download: [`data/boxes-dataset-v1.zip`](data/boxes-dataset-v1.zip)
- Password: `iamnotaLM`

---

## Model Outputs

The predictions from the original model runs are provided as a password-protected ZIP file to
prevent leakage into the training data of future language models. Please do not include the
uncompressed files in any repositories if you use the predictions.

- Download: [`model-outputs/model-outputs.zip`](model-outputs/model-outputs.zip)
- Password: `iamnotaLM`

---

## Dataset Generation

```bash
cd Tasks/State-Tracking/entity-tracking-lms

python src/dataset_generation/generate_boxes_data.py \
    --num_boxes 5 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 3 \
    --num_operations 12 \
    --num_samples 1000 \
    --output_dir data/boxes5_nso_exp2_max3_zero_shot
```

Generated datasets follow the naming convention
`boxes{N}_exp{expected}_max{max}_{num_ops}_zero_shot`, where `N` is the number of boxes. The
script writes `train-t5.jsonl`, `dev-t5.jsonl`, and `test-t5.jsonl` into `--output_dir`.

---

## Running a Single Evaluation

```bash
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --data_subdir boxes5_nso_exp2_max3_zero_shot \
    --prompt_mode chat \
    --num_test_set_per_bin 300 \
    --seed 0
```

Point directly at an existing dataset directory:

```bash
python run_evaluation.py \
    --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --dataset_dir data/boxes6_nso_exp2_max3_zero_shot \
    --prompt_mode chat
```

Results are written to `output/<dataset_slug>/<model>_<prompt_mode>.json`.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | *(required)* | HuggingFace model ID |
| `--data_subdir` | `boxes5_nso_exp2_max3_zero_shot` | Subfolder under `data/`; ignored when `--dataset_dir` is set |
| `--dataset_dir` | `None` | Direct path to a directory containing `{split}-t5.jsonl` files |
| `--data_file` | `None` | Override path to a single JSONL file (ignores split, subdir, and dir) |
| `--split` | `test` | Which split to load: `train`, `dev`, or `test` |
| `--prompt_mode` | `continuation` | `continuation` (raw prefix) or `chat` (apply_chat_template) |
| `--num_test_set_per_bin` | `300` | Examples sampled per `numops` bin |
| `--seed` | `0` | Random seed for stratified sampling |
| `--max_new_tokens` | `64` | Maximum tokens to generate per example |
| `--output_json` | auto | Override output path |

---

## Pipeline Sweep

`scripts/run_boxes_eval_pipeline.py` runs a full Cartesian sweep over box counts and models
defined in a YAML config:

```bash
python scripts/run_boxes_eval_pipeline.py --config configs/boxes_pipeline.yaml
```

The pipeline generates a dataset for each combination of `data` fields, then evaluates each
dataset against every combination of `model` fields. Existing dataset directories are skipped
(no redundant generation).

```yaml
# configs/boxes_pipeline.yaml
data:
  num_boxes: [2, 3, 4, 5, 6]      # list → sweep over all values
  expected_num_items_per_box: 2
  max_items_per_box: 3
  num_operations: 12
  num_samples: 1000

model:
  model_name: ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"]   # list → sweep over all values
  prompt_mode: chat
  split: test
  seed: 0
  num_test_set_per_bin: 300
  max_new_tokens: 64
```

Any field accepting a YAML list triggers a sweep; scalar fields are held fixed. The outer product
is taken within `data` fields and within `model` fields, then every data configuration is run
against every model configuration.

---

## Output JSON Structure

```json
{
  "model_name": "allenai/Olmo-Hybrid-Instruct-SFT-7B",
  "data_file": "data/boxes5_nso_exp2_max3_zero_shot/test-t5.jsonl",
  "dataset_slug": "boxes5_nso_exp2_max3_zero_shot",
  "split": "test",
  "num_test_set_per_bin": 300,
  "seed": 0,
  "prompt_mode": "chat",
  "pool_sizes_numops_0_to_3": [412, 380, 395, 313],
  "overall": {
    "n": 1200,
    "exact_accuracy": 0.47,
    "set_accuracy": 0.61
  },
  "by_numops": {
    "0": {"n": 300, "exact_accuracy": 0.82, "set_accuracy": 0.89},
    "3": {"n": 300, "exact_accuracy": 0.28, "set_accuracy": 0.41}
  },
  "rows": [...]
}
```

`exact_accuracy` requires the model output to match the gold content list in order; `set_accuracy`
only requires the same set of items, ignoring order. The gap between the two metrics reflects how
often models predict the correct items but in a wrong sequence.

---

## Visualization

Open `visualize_results.ipynb` in Jupyter. The notebook loads `output/` JSON files and renders:

- Set accuracy by `numops` (bar chart, grouped by model)
- Set accuracy vs. `n_boxes` (2–6) with Any-ops / Put+Remove-only breakdown

Pre-rendered figures are in `Figs/State-Tracking/`.

---

## Citation

```bibtex
@inproceedings{kim-schuster-2023-entity,
    title = "Entity Tracking in Language Models",
    author = "Kim, Najoung  and
      Schuster, Sebastian",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.213",
    pages = "3835--3855"
}
```
