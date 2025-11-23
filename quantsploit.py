#!/usr/bin/env python3
"""
Quantsploit - Quantitative Trading Framework
Simple launcher script
"""

import sys
import os

# Add the current directory to Python path so we can import quantsploit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        ('yfinance', 'yfinance'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('tabulate', 'tabulate'),
        ('prompt_toolkit', 'prompt_toolkit'),
        ('rich', 'rich'),
        ('yaml', 'pyyaml'),
    ]

    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("=" * 60)
        print("ERROR: Missing Required Dependencies")
        print("=" * 60)
        print("\nThe following packages are required but not installed:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install all dependencies, run:")
        print("  ./install_deps.sh")
        print("\nOr install manually:")
        print(f"  pip3 install --user {' '.join(missing)}")
        print("=" * 60)
        sys.exit(1)


# Check dependencies first
check_dependencies()

# Now import and run the main application
from quantsploit.core.framework import Framework
from quantsploit.ui.console import Console


def main():
    """Main entry point"""
    try:
        # Initialize framework
        framework = Framework()

        # Discover modules
        framework.discover_modules()

        # Start console
        console = Console(framework)
        console.start()

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[-] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
