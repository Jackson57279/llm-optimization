#!/usr/bin/env python3
"""Quick import test for package validation.

This script verifies that the synaptic_pruning package can be imported
correctly after installation.
"""

import sys


def test_import():
    """Test that the package imports successfully."""
    print("Testing synaptic_pruning import...")

    try:
        import synaptic_pruning

        print("✓ synaptic_pruning imported successfully")
        print(f"  Version: {synaptic_pruning.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import synaptic_pruning: {e}")
        return False


def test_module_imports():
    """Test that all submodules can be imported."""
    print("\nTesting submodule imports...")

    modules = [
        "synaptic_pruning.activity",
        "synaptic_pruning.quantization",
        "synaptic_pruning.recovery",
        "synaptic_pruning.layers",
        "synaptic_pruning.training",
        "synaptic_pruning.utils",
    ]

    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False

    return all_ok


def main():
    """Run all import tests."""
    print("=" * 60)
    print("Synaptic Pruning Import Test")
    print("=" * 60)

    success = True
    success &= test_import()
    success &= test_module_imports()

    print("\n" + "=" * 60)
    if success:
        print("✓ All import tests passed!")
        return 0
    else:
        print("✗ Some import tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
