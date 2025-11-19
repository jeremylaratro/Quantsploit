#!/usr/bin/env python3
"""
Debug script to see what's happening with module discovery
"""

import sys
import os
import importlib.util
import inspect

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantsploit.core.module import BaseModule

def test_import_module(module_file):
    """Test importing a single module"""
    print(f"\n[*] Testing: {module_file}")

    try:
        # Import the module
        spec = importlib.util.spec_from_file_location(
            f"test_module",
            module_file
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find BaseModule subclasses
        found = False
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseModule) and
                obj is not BaseModule and
                not inspect.isabstract(obj)):

                print(f"  ✓ Found class: {name}")
                print(f"    - Has 'name' attr: {hasattr(obj, 'name')}")
                print(f"    - Has 'category' attr: {hasattr(obj, 'category')}")
                print(f"    - Has 'description' attr: {hasattr(obj, 'description')}")
                found = True

        if not found:
            print(f"  ✗ No BaseModule subclasses found")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 60)
    print("Debug Module Discovery")
    print("=" * 60)

    # Test a few specific modules
    test_modules = [
        "quantsploit/modules/strategies/ml_swing_trading.py",
        "quantsploit/modules/strategies/pairs_trading.py",
        "quantsploit/modules/strategies/sma_crossover.py",
    ]

    for module_file in test_modules:
        if os.path.exists(module_file):
            test_import_module(module_file)
        else:
            print(f"\n[!] File not found: {module_file}")

if __name__ == "__main__":
    main()
