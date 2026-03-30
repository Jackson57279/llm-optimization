# Architecture: Synaptic Pruning

## System Overview

Synaptic Pruning is a neural network compression framework combining:
1. **Activity Tracking**: EMA-based per-weight activity monitoring
2. **Tiered Quantization**: FP16 → 4-bit → 1-bit based on activity
3. **Recovery Mechanism**: HyperNetwork regenerates "pruned" weights

## Component Diagram

```
Input
  ↓
SynapticLayer
  ├─ EMAActivity (tracks weight usage)
  ├─ TieredQuantizer (multi-precision storage)
  ├─ HyperNetwork (recovery for cold weights)
  └─ EffectiveWeight (combines all tiers)
  ↓
Output
```

## Data Flow

### Forward Pass
1. Compute activity-based tier masks
2. Retrieve/quantize weights per tier:
   - Hot: FP16 master weights
   - Warm: 4-bit quantized
   - Cold: 1-bit or recovered from latent
3. Construct effective weight matrix
4. Linear transformation: output = input @ effective_weight + bias

### Backward Pass
1. Gradients flow via Straight-Through Estimator (STE)
2. Update FP16 master weights (all tiers)
3. Update activity scores based on gradient magnitude
4. Recovery network receives gradients if used

## Key Design Decisions

- **Master Weights**: Always maintain FP16 for stable training
- **Tier Assignment**: Dynamic based on activity, not fixed
- **Recovery**: Latent codes are learnable parameters
- **State Management**: Efficient serialization per tier
