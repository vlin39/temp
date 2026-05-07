# Dyck Languages (BBH)

Models are evaluated on their ability to **complete** a Dyck-4 word: given an
incomplete sequence of brackets, the model must produce the closing brackets
that balance the prefix. Example:

> Input: `"[ ( { < ( ) > }"` → Target: `"} ]"`

Recognising a Dyck-n language for n>1 is provably equivalent to operating a
typed stack: each opening bracket pushes a frame, each closing bracket pops
and verifies the type. Pure SSMs cannot represent arbitrary stacks of
unbounded depth ([Merrill & Sabharwal 2024](https://arxiv.org/abs/2404.08819)),
which makes this task a sharp probe for where stack-tracking computation lives
in hybrid models.

This complements `web-of-lies` (boolean chain propagation) and
`entity-tracking-lms` (set-update tracking) by probing a different state
structure — a depth-indexed *stack* rather than a single boolean or entity set.

Source: BIG-Bench Hard `dyck_languages` (Suzgun et al. 2022) via `lukaemon/bbh`.

---

## Task Format

- **Input:** a partial Dyck-4 word with the last few closing brackets stripped
  (preceded by an instruction: *"Complete the rest of the sequence, making
  sure that the parentheses are closed properly."*)
- **Target:** the sequence of closing brackets needed to make the prefix valid
- **Bracket types:** four — `()`, `[]`, `{}`, `<>`
- **Metrics:**
  - **`exact_accuracy`**: fraction of examples where the predicted bracket
    sequence matches the gold token-for-token
  - **`token_recall`**: average length of the longest matching prefix divided
    by gold target length (partial credit; rewards getting the first few
    closes right even when later ones are wrong)
- **Difficulty axes** stored per example:
  - `target_len` — number of closing brackets in the gold target
  - `max_open_depth` — peak nesting depth in the input prefix

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
| `--num_test_set` | `200` | Total examples sampled (random) |
| `--max_new_tokens` | `16` | Generation budget; targets are short sequences |
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
  "overall": {
    "n": 200,
    "exact_accuracy": 0.42,
    "token_recall":   0.71
  },
  "by_target_len": {
    "1": {"n": 30, "exact_accuracy": 0.83, "token_recall": 0.83},
    "2": {"n": 80, "exact_accuracy": 0.51, "token_recall": 0.78},
    "3": {"n": 60, "exact_accuracy": 0.30, "token_recall": 0.66},
    "4": {"n": 30, "exact_accuracy": 0.13, "token_recall": 0.55}
  },
  "by_depth":   {"2": {...}, "3": {...}, "4": {...}, "5": {...}},
  "rows": [...]
}
```

Output path: `output/<model_slug>_<prompt_mode>[_<ablate_group>].json`.

---

## Visualization

After running evaluations, render all comparison figures:

```bash
python make_comparison_plot.py --prompt_mode chat
```

Skip figures whose data hasn't been collected:

```bash
python make_comparison_plot.py --prompt_mode chat --skip_layerwise --skip_patching
```

| Output file | Contents |
|---|---|
| `output/dyck_languages_overall.svg` | Exact-match + token recall, per model (no ablation) |
| `output/dyck_languages_ablation_comparison.svg` | Exact-match accuracy under each ablation group |
| `output/dyck_languages_accuracy_by_depth.svg` | Accuracy vs max nesting depth (dyck-specific stack-depth axis) |
| `output/dyck_languages_layerwise_ablation.svg` | Per-layer ablation, 2×2 subplot grid |
| `output/dyck_languages_patching_effect.svg` | Activation patching effect by layer |
| `output/dyck_languages_summary.md` | Markdown summary table |

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
# Step 1: corrupt one bracket in the prefix to require a different closing
python make_patching_pairs.py --num_pairs 200 --seed 0

# Step 2: sweep all decoder layers per model
python run_patching.py \
    --model_name allenai/Olmo-3-7B-Instruct-SFT \
    --prompt_mode chat \
    --pairs_file output/patching_pairs_200.json
```

Each clean/corrupt pair differs by exactly one bracket character in the input
prefix, chosen so that the *required closing sequence* changes. Patching
measures how much the model recovers the correct first-closing-bracket logit
when the clean residual is injected at each layer L.

The metric reported per layer is `mean_logit_diff = patched[clean_first_tok] −
base_corrupt[clean_first_tok]`. Peaks identify which layers carry the
stack-tracking computation.

---

## Notes & Caveats

- BBH `dyck_languages` has only ~250 examples; 200-example samples are nearly
  the full set. Multiple seeds will overlap heavily.
- `target_len` distribution in BBH is concentrated around 2–4 closing
  brackets. Higher difficulty bins are smaller.
- The `parse_model_completion` parser tolerates whitespace and a leading
  *"Answer:"* / *"The answer is"* prefix; it stops at the first non-bracket
  non-space character to avoid contamination from chatty completions.

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

@inproceedings{merrill2024illusion,
  title={The Illusion of State in State-Space Models},
  author={Merrill, William and Sabharwal, Ashish},
  booktitle={ICML},
  year={2024}
}
```
