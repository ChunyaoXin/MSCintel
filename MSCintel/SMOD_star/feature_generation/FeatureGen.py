# -*- coding: utf-8 -*-
"""
Run the full feature calculation pipeline.
This script performs:
1. Merge raw positive/negative FASTA files into train/test FASTA files.
2. Standardize external prediction files.
3. Calculate all sequence-derived and prediction-derived features.
4. Check whether expected feature files are generated successfully.
"""

from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd
import yaml


DEFAULT_FEATURE_SPECS = {
    "AA_frequency": {"script": "calculate_AA_frequency.py", "output": "AA_frequency_features.csv"},
    "AA_pairs": {"script": "calculate_AA_pairs.py", "output": "AA_pairs_features.csv"},
    "AAindex": {"script": "calculate_AAindex.py", "output": "AAindex_features.csv"},
    "Conservation": {"script": "calculate_Conservation_features.py", "output": "Conservation_features.csv"},
    "disorder": {"script": "calculate_disorder_features.py", "output": "disorder_features.csv"},
    "linker": {"script": "calculate_linker_features.py", "output": "linker_features.csv"},
    "SS": {"script": "calculate_SS_features.py", "output": "SS_features.csv"},
    "SS_RSA": {"script": "calculate_SS_RSA_features.py", "output": "SS_RSA_features.csv"}}


def load_config(config_path):
    """Load YAML configuration file."""
    config_path = Path(config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config, config_path


def resolve_project_root(config, config_path):
    """Resolve project root from config['repo_root'] relative to config file location."""
    repo_root = Path(config.get("repo_root", "."))

    if repo_root.is_absolute():
        return repo_root.resolve()

    return (config_path.parent / repo_root).resolve()


def resolve_path(project_root, path_value):
    """Resolve a config path relative to project root."""
    path_value = Path(path_value)

    if path_value.is_absolute():
        return path_value

    return project_root / path_value


def get_feature_specs(config):
    feature_cfg = config.get("feature_calculation", {})
    custom_specs = feature_cfg.get("feature_specs")

    if custom_specs is None:
        return DEFAULT_FEATURE_SPECS

    feature_specs = {}

    for name, spec in custom_specs.items():
        if "script" not in spec or "output" not in spec:
            raise ValueError(f"feature_calculation.feature_specs.{name} must contain 'script' and 'output'.")
        feature_specs[name] = {"script": spec["script"], "output": spec["output"]}

    return feature_specs


def run_command(command, project_root):
    """Run one command and stop immediately if it fails."""
#    print("\nRunning:")
#    print(" ".join(str(x) for x in command))
    subprocess.run(command, cwd=project_root, check=True,)


def count_fasta_records(fasta_path):
    """Count FASTA records by counting header lines."""
    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    count = 0

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1

    return count


def get_dataset_fasta_paths(config, project_root):
    """Get train/test FASTA paths from config."""
    return {
        "train": resolve_path(project_root, config.get("train_fasta", "Data/train.fasta")),
        "test": resolve_path(project_root, config.get("test_fasta", "Data/test.fasta"))}


def get_feature_output_root(config, project_root):
    """Get feature output root directory from config."""
    return resolve_path(project_root, config.get("feature_output_dir", "outputs/features"))


def check_feature_outputs(project_root, config, datasets, feature_specs):
    """Check whether all expected feature files exist and have consistent row numbers."""
    feature_root = get_feature_output_root(config, project_root)
    fasta_paths = get_dataset_fasta_paths(config, project_root)

    for dataset in datasets:
        if dataset not in fasta_paths:
            raise ValueError(f"Unsupported dataset: {dataset}")

        expected_rows = count_fasta_records(fasta_paths[dataset])
        dataset_dir = feature_root / dataset

        print(f"\nChecking feature outputs for {dataset}...")
        print(f"Feature directory: {dataset_dir}")
        print(f"Expected rows: {expected_rows}")

        for feature_name, spec in feature_specs.items():
            file_name = spec["output"]
            path = dataset_dir / file_name

            if not path.is_file():
                raise FileNotFoundError(f"Missing feature file: {path}")
            df = pd.read_csv(path)

            if "AccessionName" not in df.columns:
                raise ValueError(f"{path} does not contain AccessionName column")

            if "label" not in df.columns:
                raise ValueError(f"{path} does not contain label column")

            if df["AccessionName"].duplicated().any():
                duplicated = df.loc[df["AccessionName"].duplicated(), "AccessionName"].tolist()
                raise ValueError(f"{path} contains duplicated AccessionName: {duplicated[:5]}")

            if len(df) != expected_rows:
                raise ValueError(f"{path} row number mismatch: got {len(df)}, expected {expected_rows}")

            if df.isna().any().any():
                raise ValueError(f"{path} contains NaN values")

            print(f"  OK: {feature_name}: {file_name}, shape={df.shape}")

    if "train" in datasets and "test" in datasets:
        print("\nChecking train/test column consistency...")

        for feature_name, spec in feature_specs.items():
            file_name = spec["output"]

            train_path = feature_root / "train" / file_name
            test_path = feature_root / "test" / file_name

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if list(train_df.columns) != list(test_df.columns):
                raise ValueError(f"{file_name}: train/test columns are inconsistent")

            print(f"  OK: {feature_name}: {file_name}")

    print("\nAll feature output checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Run the full MSCintel feature calculation pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml. Default: config.yaml.")
    parser.add_argument("--datasets", nargs="+", default=["train", "test"], choices=["train", "test"], help="Datasets to calculate features for. Default: train test.")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge_fasta.py.")
    parser.add_argument("--skip-standardize", action="store_true", help="Skip standardize_predictions.py.")
    parser.add_argument("--allow-missing", action="store_true", help="Pass --allow-missing to feature calculation scripts.",)
    parser.add_argument("--skip-check", action="store_true", help="Skip output validation after feature calculation.")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    project_root = resolve_project_root(config, config_path)
    script_dir = Path(__file__).resolve().parent
    feature_specs = get_feature_specs(config)

    print("Start feature calculation pipeline.")
    print(f"Project root: {project_root}")
    print(f"Config file: {config_path}")
    print(f"Feature output root: {get_feature_output_root(config, project_root)}")

    required_scripts = ["merge_fasta.py", "standardize_predictions.py", *[spec["script"] for spec in feature_specs.values()]]

    for script in required_scripts:
        script_path = script_dir / script

        if not script_path.is_file():
            raise FileNotFoundError(f"Required script not found: {script_path}")

    if not args.skip_merge:
        run_command([sys.executable, str(script_dir / "merge_fasta.py"), "--config", str(config_path)], project_root)

    if not args.skip_standardize:
        run_command([sys.executable, str(script_dir / "standardize_predictions.py"), "--config", str(config_path)], project_root)

    for dataset in args.datasets:
        print(f"\n========== Calculating features for dataset: {dataset} ==========")

        for feature_name, spec in feature_specs.items():
            script = spec["script"]

            command = [sys.executable, str(script_dir / script), "--dataset", dataset, "--config", str(config_path),]

            if args.allow_missing:
                command.append("--allow-missing")

#            print(f"\nFeature block: {feature_name}")
            run_command(command, project_root)

    if not args.skip_check:
        check_feature_outputs(project_root=project_root, config=config, datasets=args.datasets, feature_specs=feature_specs)

    print("\nCompleted!!")


if __name__ == "__main__":
    main()