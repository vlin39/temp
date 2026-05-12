# In-Context Retrieval

This folder contains the **in-context retrieval** side of the project: experiments that test whether a model can recover, preserve, and reason over information supplied inside the prompt. In the broader project, this task family is used to study where retrieval-related computation is localized in hybrid language models and whether that localization differs from pure transformers.

There are **two complementary retrieval benchmarks** here:

1. `colored_objects/`  
   A compact synthetic benchmark for short-context retrieval and reasoning.

2. `NIAH/`  
   A long-context “Needle In A Haystack” benchmark for pressure-testing retrieval across context length and insertion depth.

## Folder layout

```text
In-Context Retrieval/
├── README.md
├── NIAH/
│   ├── README.md
│   ├── needlehaystack/
│   ├── tests/
│   └── viz/
└── colored_objects/
    ├── task.json
    ├── run_evaluation.py
    ├── src/colored_objects_eval.py
    ├── make_comparison_plot.py
    ├── colored_objects_reasoning.ipynb
    └── output/
```

## 1) `colored_objects/`: short-context retrieval and reasoning

### What the task is
`colored_objects/task.json` contains 2,000 examples from a structured synthetic benchmark named `reasoning_about_colored_objects`. Each example describes a small scene containing colored objects and asks a simple question whose answer must be recovered from that scene.

The task includes five subtypes:
- `what_color`
- `yes_no_color`
- `neither_color`
- `spatial`
- `arithmetic`

This benchmark is useful because it spans a clean difficulty spectrum:
- direct lookup (`what_color`)
- verification (`yes_no_color`)
- filtering / counting (`neither_color`)
- positional retrieval (`spatial`)
- lightweight symbolic manipulation (`arithmetic`)

### Important files
- `task.json` — the benchmark examples
- `run_evaluation.py` — main evaluation entry point
- `src/colored_objects_eval.py` — prompt construction, parsing, accuracy computation
- `make_comparison_plot.py` — builds SVG summaries from saved JSON outputs
- `output/` — included result JSONs and comparison figures

### Prompting modes
The evaluation script supports:
- `continuation`
- `chat`
- `chat_continual`

These let you compare raw continuation-style prompting against chat-template prompting.

### Built-in ablation support
This is the retrieval task with the **best direct support for the paper’s ablation story**.

`run_evaluation.py` exposes:
- `--ablate_group none`
- `--ablate_group self_attn`
- `--ablate_group linear_attn`

and also:
- `--sweep_ablate_groups`

Internally, those flags call the shared utilities in `Models/model_util.py`, so this folder is the easiest place to reproduce **module-group functional ablations**.

### Example commands

Run one subtype:
```bash
cd "Tasks/In-Context Retrieval/colored_objects"

python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --subtype what_color   --prompt_mode chat   --num_test_set 250   --seed 0
```

Sweep all subtypes:
```bash
python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --sweep_subtypes   --prompt_mode chat   --num_test_set 300   --seed 0
```

Sweep prompt modes and ablations together:
```bash
python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --sweep_subtypes   --sweep_prompt_modes   --sweep_ablate_groups   --num_test_set 300   --seed 0
```

Restrict evaluation to particular object-count bins:
```bash
python run_evaluation.py   --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B   --subtype spatial   --prompt_mode chat   --allowed_num_objects 3,4,5   --num_test_set 300
```

### Included results
The `output/` folder already contains saved runs for:
- `allenai/Olmo-3-7B-Instruct-SFT`
- `allenai/Olmo-Hybrid-Instruct-SFT-7B`
- selected Qwen models on the easier color subtasks

A few notable takeaways from the saved files and the project figures:
- **What-color lookup is near ceiling** for both OLMo variants.
- The **hybrid model is stronger on harder retrieval-and-reasoning variants**, especially:
  - `yes_no_color`
  - `neither_color`
  - `spatial`
- The **pure transformer is slightly stronger on the arithmetic subtype** in the saved OLMo runs.

In the saved OLMo 7B chat outputs:
- `what_color`: both models are ~0.988 accuracy
- `yes_no_color`: hybrid ~0.994 vs pure ~0.931
- `neither_color`: hybrid ~0.353 vs pure ~0.180
- `spatial`: hybrid ~0.478 vs pure ~0.310
- `arithmetic`: pure ~0.389 vs hybrid ~0.306

These trends also match the figure panels in the provided project images:
- the colored-object plots show near-perfect performance on simple color lookup
- larger gaps appear on counting/filtering and spatial variants
- arithmetic remains harder and more mixed

## 2) `NIAH/`: long-context retrieval pressure testing

### What it is
`NIAH/` contains a vendored / adapted version of the **Needle In A Haystack** benchmark, which tests whether a model can retrieve a planted fact (“needle”) from a long context (“haystack”) as:
- total context length increases
- needle depth varies

The included benchmark code supports:
- model-provider abstractions
- evaluators
- single-needle and multi-needle insertion
- visualization notebooks
- a text corpus (`needlehaystack/PaulGrahamEssays/`) for generating haystacks

### Important files
- `NIAH/README.md` — upstream-style benchmark documentation
- `needlehaystack/run.py` — package entry point
- `needlehaystack/llm_needle_haystack_tester.py` — single-needle testing
- `needlehaystack/llm_multi_needle_haystack_tester.py` — multi-needle testing
- `viz/` — visualization notebooks and CSVs

### Important caveat
This subfolder is primarily a **benchmark package**, not a polished project-specific experiment wrapper for the OLMo/Qwen Hugging Face models in the rest of the repo. In practice:
- it is useful for long-context retrieval infrastructure
- it provides the benchmark framing and tooling
- but the custom project results appear to be summarized more in the provided figures than in committed machine-readable result files inside this folder

### What the project figures suggest
The provided project figures include several retrieval heatmaps associated with NIAH-style pressure testing. Those figures indicate:
- strong long-context retrieval for some Qwen models
- more visible degradation regions for at least some OLMo-family runs
- a useful contrast between pure-transformer and hybrid behavior as context length and needle depth increase

So, in the context of this repository, `NIAH/` should be read as the **long-context complement** to `colored_objects/`:
- `colored_objects/` probes short, controlled retrieval/reasoning
- `NIAH/` probes retrieval robustness under long-context stress

## Recommended workflow for this task family

### If you want quick reproduction
Start with `colored_objects/`, because it has:
- included data
- included outputs
- direct Hugging Face model loading
- direct ablation hooks

### If you want long-context stress tests
Then move to `NIAH/` for:
- haystack generation
- multi-depth / multi-length retrieval evaluation
- visualization notebooks

### If you want paper-style sensitivity experiments
Start in `colored_objects/` and extend it from:
- group ablations (`self_attn` vs `linear_attn`)
to
- layerwise ablations via `Models/model_util.py::ablate_single_attn_module(...)`

## Minimal dependencies

For the custom retrieval scripts, the lightweight environment in:
```text
Env/colored_objects_environment.yml
```
is the best starting point.

For `NIAH/`, check the local `requirements.txt` and upstream README in that directory, since it follows its own package structure and assumptions.
