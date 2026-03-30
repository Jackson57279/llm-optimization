# User Testing Guide

## Validation Surface

This is a Python research library - validation is through:
1. **pytest**: Unit and integration tests
2. **Benchmark scripts**: End-to-end training and evaluation
3. **Example scripts**: Tutorial-style demonstrations

## Required Tools

- pytest for automated testing
- Python 3.14+ interpreter
- No browser/CLI/TUI testing needed

## Resource Cost Classification

**pytest (unit tests)**: 
- Max concurrent: 5
- Each test: ~100MB RAM, <1s runtime
- Headroom: Can run full suite in parallel

**benchmark scripts**:
- Max concurrent: 1 (training uses all resources)
- GPT-2 training: ~2GB RAM, 5-10 minutes for 100 steps
- Headroom: Run sequentially only

## Testing Commands

```bash
# Unit tests
pytest tests/ -v

# Specific component
pytest tests/test_activity.py -v

# Benchmark
python benchmarks/train_gpt2_small.py

# Example
python examples/end_to_end_pipeline.py
```
