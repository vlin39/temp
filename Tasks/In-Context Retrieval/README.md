# In-Context Retrieval

Two benchmarks that probe how well models retrieve and reason over information stored in context:

- **[Colored Objects](colored_objects/README.md)** — multi-step reasoning with object attributes
  at varying distractor counts (2–6 objects, five subtypes). Supports local ablation via
  `--ablate_group`. Committed results for OLMo and Qwen models are in `colored_objects/output/`.

- **[NIAH](NIAH/README.md)** — Needle-in-a-Haystack long-context retrieval at controlled depths
  and context lengths (1K–65K tokens). Uses the upstream `needlehaystack` API-based package;
  local ablation is not applied. Project results are rendered as heatmaps in
  `Figs/In-Context-Retrieval/fig_niah_*.jpeg`.

Pre-rendered comparison figures for both tasks are in `Figs/In-Context-Retrieval/`.
