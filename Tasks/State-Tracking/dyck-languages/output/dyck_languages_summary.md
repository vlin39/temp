# Dyck Languages Results Summary

## Exact-match accuracy by ablation group

| Model | none | self_attn | linear_attn |
|-------|------|-----------|-------------|
| OLMo-3 (pure) | 0.000 | 0.000 | 0.000 |
| Qwen3-8B (pure) | 0.000 | 0.000 | 0.000 |
| OLMo-Hybrid | 0.000 | 0.000 | 0.000 |
| Qwen3.5-9B | 0.000 | 0.000 | 0.000 |

## Overall: exact / token recall (no ablation)

| Model | exact | recall |
|-------|-------|--------|
| OLMo-3 (pure) | 0.000 | 0.000 |
| Qwen3-8B (pure) | 0.000 | 0.000 |
| OLMo-Hybrid | 0.000 | 0.000 |
| Qwen3.5-9B | 0.000 | 0.000 |

## Layerwise ablation — top-3 most-impactful layers per model

### OLMo-3 (pure)
baseline exact accuracy: 0.000

| layer_idx | layer_type | exact | drop |
|-----------|------------|-------|------|
| 0 | self_attn | 0.000 | 0.000 |
| 1 | self_attn | 0.000 | 0.000 |
| 2 | self_attn | 0.000 | 0.000 |

### Qwen3-8B (pure)
baseline exact accuracy: 0.000

| layer_idx | layer_type | exact | drop |
|-----------|------------|-------|------|
| 0 | self_attn | 0.000 | 0.000 |
| 1 | self_attn | 0.000 | 0.000 |
| 2 | self_attn | 0.000 | 0.000 |

### OLMo-Hybrid
baseline exact accuracy: 0.000

| layer_idx | layer_type | exact | drop |
|-----------|------------|-------|------|
| 0 | linear_attn | 0.000 | 0.000 |
| 1 | linear_attn | 0.000 | 0.000 |
| 2 | linear_attn | 0.000 | 0.000 |

### Qwen3.5-9B
baseline exact accuracy: 0.000

| layer_idx | layer_type | exact | drop |
|-----------|------------|-------|------|
| 0 | linear_attn | 0.000 | 0.000 |
| 1 | linear_attn | 0.000 | 0.000 |
| 2 | linear_attn | 0.000 | 0.000 |

## Activation patching — top-3 layers by recovery

### OLMo-3 (pure)

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 31 | self_attn | 0.568 | 0.000 |
| 29 | self_attn | 0.564 | 0.000 |
| 30 | self_attn | 0.559 | 0.000 |

### Qwen3-8B (pure)

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 35 | self_attn | 0.568 | 0.000 |
| 34 | self_attn | 0.546 | 0.000 |
| 33 | self_attn | 0.194 | 0.000 |

### OLMo-Hybrid

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 31 | self_attn | 0.540 | 0.000 |
| 30 | linear_attn | 0.524 | 0.000 |
| 28 | linear_attn | 0.524 | 0.000 |

### Qwen3.5-9B

| layer_idx | layer_type | norm. logit diff | patched accuracy |
|-----------|------------|------------------|------------------|
| 31 | self_attn | 0.584 | 0.000 |
| 30 | linear_attn | 0.522 | 0.000 |
| 29 | linear_attn | 0.519 | 0.000 |

