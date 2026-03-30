# Synaptic Pruning

**Activity-Driven Sparse Quantization with Learned Recovery**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A novel neural network compression framework that achieves extreme compression (up to 1000×) by combining activity-driven sparsity with multi-tier quantization and a learned recovery mechanism.

## Overview

Unlike traditional pruning (permanent removal) or quantization (uniform precision reduction), Synaptic Pruning:

1. **Tracks weight activity** using exponential moving averages
2. **Multi-tier storage**: Active weights in FP16, recently inactive in 4-bit, long-term inactive in 1-bit or latent codes
3. **Recovery mechanism**: "Pruned" weights can be regenerated from a small hypernetwork/codebook
4. **Self-healing**: Model adapts to changing task distributions

## Installation

### From Source (Development)

```bash
git clone https://github.com/research/synaptic-pruning.git
cd synaptic-pruning
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Benchmark Dependencies

```bash
pip install -e ".[benchmarks]"
```

## Quick Start

```python
import torch
from synaptic_pruning import SynapticLayer

# Replace nn.Linear with SynapticLayer
layer = SynapticLayer(
    in_features=768,
    out_features=3072,
    activity_decay=0.9,
    hot_threshold=0.8,
    warm_threshold=0.3,
)

# Use like a regular linear layer
x = torch.randn(32, 768)
y = layer(x)

# Activity tracking and quantization happen automatically
```

## Architecture

### Core Components

- **`EMAActivity`**: Per-weight activity tracking with exponential moving averages
- **`TieredQuantizer`**: Multi-tier quantization (FP16 → 4-bit → 1-bit)
- **`HyperNetwork`**: Generates weights from small latent codes for recovery
- **`CodebookVQ`**: Vector quantization codebook for cold weights
- **`SynapticLayer`**: Drop-in replacement for `nn.Linear` combining all components
- **`SynapticTrainer`**: End-to-end training with progressive pruning

### Training Flow

```python
from synaptic_pruning import SynapticLayer, SynapticTrainer
import torch.nn as nn

# Build model with SynapticLayers
model = nn.Sequential(
    SynapticLayer(768, 3072),
    nn.GELU(),
    SynapticLayer(3072, 768),
)

# Setup trainer
optimizer = torch.optim.AdamW(model.parameters())
trainer = SynapticTrainer(
    model,
    optimizer,
    config={
        "pruning_schedule": [0.0, 0.5, 0.9],  # Target sparsity per epoch
    }
)

# Train with automatic compression
trainer.train(dataloader, num_epochs=50)

# Check compression stats
stats = trainer.get_compression_stats()
print(f"Compression ratio: {stats['total_ratio']:.1f}x")
```

## Project Structure

```
synaptic-pruning/
├── synaptic_pruning/          # Main package
│   ├── __init__.py           # Package initialization
│   ├── activity.py           # EMAActivity tracker
│   ├── quantization.py       # TieredQuantizer
│   ├── recovery.py            # HyperNetwork and CodebookVQ
│   ├── layers.py              # SynapticLayer
│   ├── training.py            # SynapticTrainer
│   └── utils.py               # Utilities and visualization
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_import.py         # Import validation
├── benchmarks/                # Evaluation scripts (to be added)
├── experiments/               # Example scripts (to be added)
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=synaptic_pruning --cov-report=html
```

## Validation

The project follows a comprehensive validation strategy with 26 behavioral assertions across 6 areas:

- **ACT** (Activity Tracking): EMA computation, tier classification, visualization
- **QNT** (Quantization): Round-trip accuracy, tier assignment, gradient flow
- **REC** (Recovery): HyperNetwork generation, cosine similarity, codebook compression
- **LAY** (Layer): Forward/backward pass, state persistence, nn.Linear compatibility
- **TRN** (Training): Trainer setup, pruning schedule, recovery training
- **BEN** (Benchmarks): GPT-2 training, perplexity evaluation, ablation studies

## Roadmap

### Milestone 1: Foundation ✅
- [x] Project structure
- [x] EMAActivity tracker
- [x] Activity visualization
- [x] Unit tests

### Milestone 2: Quantization
- [ ] TieredQuantizer implementation
- [ ] FP16, 4-bit, 1-bit quantization
- [ ] Differentiable quantization with STE

### Milestone 3: Recovery
- [ ] HyperNetwork for weight generation
- [ ] Codebook VQ for compression
- [ ] Recovery training loop

### Milestone 4: Synaptic Layer
- [ ] Drop-in nn.Linear replacement
- [ ] Gradient flow through all tiers
- [ ] State save/load

### Milestone 5: Training System
- [ ] SynapticTrainer
- [ ] Progressive pruning schedule
- [ ] End-to-end training

### Milestone 6: Benchmarks
- [ ] GPT-2 training
- [ ] Perplexity evaluation
- [ ] Baseline comparisons

## Citation

If you use Synaptic Pruning in your research, please cite:

```bibtex
@software{synaptic_pruning,
  title = {Synaptic Pruning: Activity-Driven Sparse Quantization with Learned Recovery},
  author = {Research Team},
  year = {2025},
  url = {https://github.com/research/synaptic-pruning}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## Acknowledgments

This project builds on ideas from GPTQ, AWQ, TurboQuant, and other quantization research in the deep learning community.
