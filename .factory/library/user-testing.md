# Flow Validator Guidance: pytest

## Isolation Rules

For pytest-based validation of the Synaptic Pruning library:

### Resource Boundaries
- Each flow validator runs pytest on specific test files
- Tests are isolated by Python process - no shared state between test runs
- CPU/memory resources are consumed during test execution

### Safe Concurrency
- Multiple pytest processes can run concurrently
- Each subagent should run a focused subset of tests
- No shared database or file locks to worry about

### Off-Limits
- Do not modify source code during testing
- Do not write to production data directories
- Only write to designated evidence/output directories

## Execution Pattern

1. Activate the skill (already done by validator)
2. Run specific test file with pytest
3. Collect test output and evidence
4. Write results to evidence directory

## Commands

```bash
# Run specific test file
cd /home/dih/llm-optimization/synaptic-pruning
pytest tests/test_quantization.py -v --tb=short 2>&1

# Run specific test class
pytest tests/test_quantization.py::Test4BitQuantization -v --tb=short

# Run specific test method  
pytest tests/test_quantization.py::Test4BitQuantization::test_4bit_round_trip_error -v --tb=short
```

## Evidence Collection

- Test output (stdout/stderr)
- Exit code (0 = success, non-zero = failure)
- Any generated files

## Assertion Mapping

- VAL-QNT-001: 4-bit round-trip error -> test_4bit_round_trip_error
- VAL-QNT-002: 1-bit binary representation -> test_1bit_values_are_binary, test_dequantize_1bit_produces_scaled_values
- VAL-QNT-003: Tier assignment -> test_tier_assignment_hot_is_fp16, test_tier_assignment_warm_is_4bit, test_tier_assignment_cold_is_1bit
- VAL-QNT-004: Gradient flow -> test_4bit_gradient_flow, test_1bit_gradient_flow, test_tiered_quantization_gradient_flow
