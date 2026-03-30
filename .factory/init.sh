#!/bin/bash
# Initialization script for Synaptic Pruning mission
# Must be idempotent - safe to run multiple times

set -e

echo "Initializing Synaptic Pruning environment..."

# Check Python version
python3 --version

# Install package in editable mode if not already installed
if ! python3 -c "import synaptic_pruning" 2>/dev/null; then
    echo "Installing synaptic_pruning package..."
    pip install -e . --quiet
fi

# Verify pytest is available
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest --quiet
fi

echo "Initialization complete!"
