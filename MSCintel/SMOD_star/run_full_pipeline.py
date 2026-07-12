# -*- coding: utf-8 -*-
"""
Run the full MSCintel pipeline:
1. Generate/standardize input FASTA and external prediction features.
2. Calculate sequence-derived and prediction-derived feature files.
3. Run feature selection.
4. Train and evaluate the final Stacking model.
"""

from pathlib import Path
import argparse
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent


def run_command(command):
    """Run one command under project root and stop if it fails."""
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full feature generation + feature selection + Stacking pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file. Default: config.yaml")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing external prediction results during feature calculation.")
    parser.add_argument("--skip-merge", action="store_true", help="Skip FASTA merging step in FeatureGen.py.")
    parser.add_argument("--skip-standardize", action="store_true", help="Skip external prediction standardization step in FeatureGen.py.")
    parser.add_argument("--skip-check", action="store_true", help="Skip feature output checking step in FeatureGen.py.")
    parser.add_argument("--skip-feature-generation", action="store_true", help="Skip FeatureGen.py and directly run feature selection from existing feature files.")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Only generate feature files; skip feature selection and Stacking.")

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print("========== Full MSCintel pipeline started ==========")
    print(f"Config file: {config_path}")

    # Step 1: generate feature files
    if not args.skip_feature_generation:
        feature_cmd = [sys.executable, str(REPO_ROOT / "feature_generation" / "FeatureGen.py"), "--config", str(config_path)]

        if args.allow_missing:
            feature_cmd.append("--allow-missing")

        if args.skip_merge:
            feature_cmd.append("--skip-merge")

        if args.skip_standardize:
            feature_cmd.append("--skip-standardize")

        if args.skip_check:
            feature_cmd.append("--skip-check")

        run_command(feature_cmd)
    else:
        print("\nSkipping feature generation step.")

    # Step 2: run feature selection and Stacking model
    if not args.skip_feature_selection:
        selection_cmd = [sys.executable, str(REPO_ROOT / "feature_selection" / "pipeline.py"), "--config", str(config_path)]
        run_command(selection_cmd)
    else:
        print("\nSkipping feature selection step.")

    print("\n========== Full MSCintel pipeline completed ==========")
    print("Final outputs should be saved according to config YAML paths.")


if __name__ == "__main__":
    main()