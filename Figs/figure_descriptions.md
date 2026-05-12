# Figure Descriptions

All project figures are stored in this directory, organized into task subfolders. File names
follow the convention `fig_<task>_<description>.jpeg`. The source script or notebook used to
generate each figure is noted below.

---

## In-Context Retrieval

Figures for two benchmarks: **Colored Objects** (multi-step reasoning with object attributes) and
**NIAH** (Needle-in-a-Haystack long-context retrieval).

### Colored Objects

Source: `Tasks/In-Context Retrieval/colored_objects/make_comparison_plot.py`

**`fig_colored_objects_core_subtypes_accuracy_by_num_objects.jpeg`**
Three-panel bar chart. Each panel shows accuracy vs. number of objects (2–6) for one subtype:
*What Color*, *Yes/No Color*, and *Neither Color*. Bars are grouped by model — Pure Transformer
(green) vs. Hybrid (pink). Shows how retrieval difficulty scales with the number of distractors
across the three core subtypes.

**`fig_colored_objects_arithmetic_accuracy_by_num_objects.jpeg`**
Single-panel bar chart. Accuracy vs. number of objects (2–14) for the *Arithmetic* subtype, which
requires counting or summing object attributes rather than direct retrieval. The wider object-count
range (up to 14) stresses models beyond the core 2–6 range.

### NIAH (Needle-in-a-Haystack)

Source: `Tasks/In-Context Retrieval/NIAH/viz/CreateVizFromLLMTesting.ipynb`

Each figure is a retrieval-score heatmap. X-axis: context window size in tokens (1K–32K or 65K).
Y-axis: depth percent (0 % = needle at the start, 100 % = needle at the end). Color: retrieval
score 1–10 (green = 10 / correct, pink = 0 / failed). A fully green heatmap indicates robust
retrieval at all depths and context lengths.

| Figure | Model | Context | Notes |
|--------|-------|---------|-------|
| `fig_niah_olmo_hybrid_sft_7b_32k.jpeg` | OLMo-Hybrid-Instruct-SFT-7B | 32K | Moderate retrieval; degrades at mid-context depths |
| `fig_niah_olmo_hybrid_sft_7b_32k_v2.jpeg` | OLMo-Hybrid-Instruct-7B-SFT | 32K | Second run / configuration variant |
| `fig_niah_olmo_hybrid_dpo_7b_32k.jpeg` | OLMo-Hybrid-Instruct-7B-DPO | 32K | Noticeably stronger than the SFT variant |
| `fig_niah_olmo3_instruct_7b_32k.jpeg` | OLMo-3-Instruct-7B (pure) | 32K | More degradation than the hybrid counterpart |
| `fig_niah_olmo_instruct_7b_32k.jpeg` | OLMo-Instruct-7B (pure) | 32K | Widespread pink at longer contexts |
| `fig_niah_olmo_instruct_dpo_7b_32k.jpeg` | OLMo-Instruct-7B-DPO (pure) | 32K | Pink-dominant beyond ~8K tokens |
| `fig_niah_olmo_instruct_sft_7b_32k.jpeg` | OLMo-Instruct-7B-SFT (pure) | 32K | Best-performing of the pure-transformer variants |
| `fig_niah_olmo_instruct_sft_65k.jpeg` | OLMo-Instruct-7B-SFT (pure) | 65K | Extended-context window test; shows failure modes beyond 32K |
| `fig_niah_qwen3_5_9b_32k.jpeg` | Qwen3.5-9B (hybrid) | 32K | Near-perfect retrieval (nearly all green) |
| `fig_niah_qwen3_8b_32k.jpeg` | Qwen3-8B (hybrid) | 32K | Near-perfect retrieval (nearly all green) |

---

## State Tracking

Source: `Tasks/State-Tracking/entity-tracking-lms/visualize_results.ipynb`

**`fig_state_tracking_hybrid_vs_pure_transformer_set_accuracy_by_numops.jpeg`**
Grouped bar chart comparing set accuracy by `numops` (0–5 operations) for a single model pair
(OLMo-3-7B vs. OLMo-Hybrid-7B) on the 5-box configuration. Each bar group corresponds to one
`numops` level; bars are color-coded by model type (pure transformer vs. hybrid). Illustrates
whether one architecture degrades faster as the operation count grows.

**`fig_state_tracking_model_comparison_summary.jpeg`**
6-panel summary figure. Top row: set accuracy by `numops` for three model pairs — (1)
OLMo-3-7B / OLMo-Hybrid-7B, (2) Qwen3-8B / Qwen3.5-9B, (3) Qwen3-32B / Qwen3.5-27B — at
`n_boxes = 6`. Bottom row: overall set accuracy vs. `n_boxes` (2–6) for the same pairs, further
split into *Any operations* vs. *Put+Remove-only* subsets.

---

## Clinical Sequence Modeling

Source: `Tasks/Clinical_Sequence_Modeling/MIMIC-IV/evaluate_across_time.py`

**`fig_clinical_architecture_diagram.jpeg`**
System architecture diagram (manuscript figure). Left panel: Missingness-Aware Contrastive
pre-training across 5 clinical modalities (CXR images, radiology notes, discharge summaries,
vital-sign time series, demographics). Right panel: Belief Update Transformer (BUT) — the LLM
backbone receives a chronological stream of modality embeddings interleaved with learnable
`[PREDICT]` tokens and outputs a mortality probability at each prediction point.

**`fig_clinical_olmo_hybrid_7b_auroc_uncertainty_vs_seq_len.jpeg`**
Dual-panel plot for **OLMo-Hybrid-Instruct-SFT-7B** (early evaluation run). Top panel: AUROC
(blue line) and mean predictive entropy (red dashed) vs. the number of clinical events observed.
Bottom panel: histogram of sample counts per event-count bucket. As the sequence lengthens, AUROC
rises while entropy decreases — the model becomes more accurate and more confident.

**`fig_clinical_olmo3_7b_auroc_uncertainty_vs_seq_len.jpeg`**
Same dual-panel layout as above, but for **OLMo-3-7B-Instruct-SFT** (pure transformer baseline,
early run). Serves as a direct comparison to the hybrid figure above.

**`fig_clinical_olmo3_7b_auroc_vs_seq_len.jpeg`**
Single-metric AUROC-vs-sequence-length plot for **OLMo-3-7B-Instruct-SFT**, evaluated on a larger
validation set. Cleaner signal than the early-run dual-panel figure.

**`fig_clinical_olmo_hybrid_7b_auroc_vs_seq_len.jpeg`**
Single-metric AUROC-vs-sequence-length plot for **OLMo-Hybrid-Instruct-SFT-7B**, larger
evaluation set. Paired with the OLMo-3 figure above for direct comparison.

**`fig_clinical_qwen3_8b_auroc_vs_seq_len.jpeg`**
AUROC vs. sequence length for **Qwen3-8B** (hybrid backbone). Shows the anytime-prediction
trajectory on MIMIC-IV mortality.

**`fig_clinical_qwen3_5_9b_auroc_vs_seq_len.jpeg`**
AUROC vs. sequence length for **Qwen3.5-9B** (hybrid backbone).

**`fig_clinical_qwen3_32b_auroc_vs_seq_len.jpeg`**
AUROC vs. sequence length for **Qwen3-32B** (hybrid backbone). Larger model in the Qwen3 family.

**`fig_clinical_qwen3_5_27b_auroc_vs_seq_len.jpeg`**
AUROC vs. sequence length for **Qwen3.5-27B** (hybrid backbone). All four Qwen figures show AUROC
rising steeply toward ~1.0 as sequence length increases.

---

## Misc

**`fig_removal_audit_meditron_7b_ehr_ablation.jpeg`**
Reference figure from related work (not generated by scripts in this repository). Shows a removal
audit on `epfl-llm/meditron-7b` fine-tuned on clinical sequences: left column = scratch-trained
model AUROC trajectory before and after ablating the Time-Series (EHR) modality; right column =
contrastively pre-trained model under the same ablation. Includes N×N internal attention heatmaps
before and after removal. Included for reference and comparison purposes.
