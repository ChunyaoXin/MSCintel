# -*- coding: utf-8 -*-
"""
Unified model development pipeline for MSCintel.

Usage:
    # Run the full pipeline
    python SMOD_star_train.py --config config.yaml

"""

from pathlib import Path
import argparse
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent


def run_command(command):
    """Run a command under the project root and stop if it fails."""
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the MSCintel model development pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--skip-check", action="store_true", help="Skip feature output validation.")
    parser.add_argument("--skip-feature-generation", action="store_true", help="Skip feature generation and use existing feature files.")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Only generate feature files; skip feature selection and Stacking.")

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if args.skip_feature_generation and args.skip_feature_selection:
        parser.error("Cannot skip both feature generation and feature selection.")

    print("========== MSCintel model development started ==========")
    print(f"Project root: {REPO_ROOT}")
    print(f"Config file: {config_path}")

    # Step 1: Feature generation
    if not args.skip_feature_generation:
        print("\n========== Step 1: Feature generation ==========")

        feature_cmd = [
            sys.executable,
            str(REPO_ROOT / "feature_generation" / "FeatureGen.py"),
            "--train",
            "--config", str(config_path),
        ]

        if args.skip_check:
            feature_cmd.append("--skip-check")

        run_command(feature_cmd)
    else:
        print("\nSkipping feature generation.")

    # Step 2: Feature selection and Stacking model construction
    if not args.skip_feature_selection:
        print("\n========== Step 2: Feature selection and Stacking model construction ==========")

        selection_cmd = [
            sys.executable,
            str(REPO_ROOT / "feature_selection" / "pipeline.py"),
            "--config", str(config_path),
        ]

        run_command(selection_cmd)
    else:
        print("\nSkipping feature selection and Stacking.")

    print("\n========== MSCintel model development completed ==========")


if __name__ == "__main__":
    main()