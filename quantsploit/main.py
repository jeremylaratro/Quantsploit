"""
Main entry point for Quantsploit
"""

import sys
import os
from .core.framework import Framework
from .ui.console import Console


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
