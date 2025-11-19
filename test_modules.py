#!/usr/bin/env python3
"""
Test script to verify all modules can be discovered
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantsploit.core.framework import Framework

def main():
    print("=" * 60)
    print("Testing Quantsploit Module Discovery")
    print("=" * 60)

    # Initialize framework
    framework = Framework()

    # Discover modules
    print("\n[*] Discovering modules...")
    framework.discover_modules()

    # List all modules
    print(f"\n[+] Found {len(framework.modules)} modules:\n")

    # Group by category
    by_category = {}
    for path, metadata in framework.modules.items():
        category = metadata.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(metadata)

    # Display by category
    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()}:")
        print("-" * 60)
        for metadata in sorted(by_category[category], key=lambda m: m.name):
            print(f"  ✓ {metadata.name:30s} - {metadata.description}")

    print("\n" + "=" * 60)
    print(f"Total: {len(framework.modules)} modules loaded successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()
