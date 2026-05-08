# Web of Lies Results Summary

## Overall accuracy by ablation group

| Model | none | self_attn | linear_attn |
|-------|------|-----------|-------------|
| OLMo-3 (pure) | 0.500 | 0.000 | 0.475 |
| Qwen3-8B (pure) | — | 0.000 | 0.515 |
| OLMo-Hybrid | 0.490 | 0.000 | 0.000 |
| Qwen3.5-9B | — | 0.000 | 0.000 |

## Accuracy by answer (no ablation)

| Model | yes | no | overall |
|-------|-----|----|---------|
| OLMo-3 (pure) | 0.980 | 0.020 | 0.500 |
| Qwen3-8B (pure) | — | — | — |
| OLMo-Hybrid | 0.640 | 0.340 | 0.490 |
| Qwen3.5-9B | — | — | — |

## Layerwise ablation — top-3 most-impactful layers per model

Layers ranked by accuracy drop relative to the unablated baseline.

### OLMo-3 (pure)
baseline accuracy: 0.500

| layer_idx | layer_type | accuracy | drop |
|-----------|------------|----------|------|
| 7 | self_attn | 0.465 | 0.035 |
| 5 | self_attn | 0.470 | 0.030 |
| 12 | self_attn | 0.470 | 0.030 |

### Qwen3-8B (pure)

| layer_idx | layer_type | accuracy | drop |
|-----------|------------|----------|------|
| 32 | self_attn | 0.475 | — |
| 15 | self_attn | 0.495 | — |
| 28 | self_attn | 0.495 | — |

### OLMo-Hybrid
baseline accuracy: 0.490

| layer_idx | layer_type | accuracy | drop |
|-----------|------------|----------|------|
| 0 | linear_attn | 0.095 | 0.395 |
| 9 | linear_attn | 0.455 | 0.035 |
| 22 | linear_attn | 0.460 | 0.030 |

### Qwen3.5-9B

| layer_idx | layer_type | accuracy | drop |
|-----------|------------|----------|------|
| 0 | linear_attn | 0.000 | — |
| 3 | self_attn | 0.000 | — |
| 7 | self_attn | 0.155 | — |

## Activation patching — top-3 layers by recovery

Higher mean normalized logit diff → that layer's clean activations carry more answer-relevant information.

### OLMo-3 (pure)

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 28 | self_attn | 0.947 | 0.480 |
| 23 | self_attn | 0.947 | 0.485 |
| 30 | self_attn | 0.942 | 0.485 |

### Qwen3-8B (pure)

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 35 | self_attn | 0.980 | 0.475 |
| 21 | self_attn | 0.976 | 0.480 |
| 29 | self_attn | 0.976 | 0.480 |

### OLMo-Hybrid

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 27 | self_attn | 0.992 | 0.500 |
| 31 | self_attn | 0.990 | 0.500 |
| 30 | linear_attn | 0.987 | 0.500 |

### Qwen3.5-9B

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 28 | linear_attn | 0.896 | 0.550 |
| 23 | self_attn | 0.888 | 0.555 |
| 19 | self_attn | 0.886 | 0.540 |

