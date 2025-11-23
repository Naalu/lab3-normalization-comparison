"""
Main entry point for Lab 3: Normalization Techniques

This script provides a command-line interface to run experiments.

Usage:
    python main.py              # Run all experiments
    python main.py --verify     # Run verification tests only

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from experiments.run import run_all


def main():
    """
    Parse command-line arguments and execute requested mode.
    """
    parser = argparse.ArgumentParser(
        description="Lab 3: Normalization Techniques - Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run all experiments
    python main.py --verify           # Verification only (future feature)
    
After running, check:
    - results/metrics/*.json         # Numerical results
    - plots/*.png                    # Visualization plots
        """,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification tests only (compare custom vs TF implementations)",
    )

    args = parser.parse_args()

    if args.verify:
        print("Verification-only mode not yet implemented.")
        print(
            "Run 'python main.py' to execute all experiments (includes verification)."
        )
        sys.exit(1)
    else:
        # Run all experiments
        run_all()


if __name__ == "__main__":
    main()
