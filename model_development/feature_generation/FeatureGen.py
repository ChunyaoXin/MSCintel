# -*- coding: utf-8 -*-
"""
Feature calculation pipeline for MSCintel.
Calculates sequence-derived features from protein FASTA files and precomputed external prediction results, and saves features into separate CSV files by feature category.

Usage:
    cd model_development/feature_generation/
    python FeatureGen.py \
        --train-fasta Data/train.fasta \
        --test-fasta Data/test.fasta \
        --prediction-dir predicted_features \
        --output-dir ../example
"""


from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd

from scripts.workflow_utils import read_labelled_fasta


DEFAULT_FEATURE_SPECS = {
    "AA_frequency": {"script": "calculate_AA_frequency.py", "output": "AA_frequency_features.csv"},
    "AA_pairs": {"script": "calculate_AA_pairs.py", "output": "AA_pairs_features.csv"},
    "AAindex": {"script": "calculate_AAindex.py", "output": "AAindex_features.csv"},
    "Conservation": {"script": "calculate_Conservation_features.py", "output": "Conservation_features.csv"},
    "disorder": {"script": "calculate_disorder_features.py", "output": "disorder_features.csv"},
    "linker": {"script": "calculate_linker_features.py", "output": "linker_features.csv"},
    "SS": {"script": "calculate_SS_features.py", "output": "SS_features.csv"},
    "SS_RSA": {"script": "calculate_SS_RSA_features.py", "output": "SS_RSA_features.csv"}}



def run_command(command, project_root):
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


def check_feature_outputs(feature_root, fasta_paths, datasets, feature_specs):
    """Check whether all expected feature files exist and have consistent row numbers."""

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
        
def create_temporary_manifest(dataset, fasta, pred_dir, output_dir):
    """Create a temporary prediction manifest from standardized files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    apod_path = pred_dir / "linker.apod"
    if not apod_path.is_file():
        raise FileNotFoundError(f"Missing linker file: {apod_path}")

    linker_accessions = set()
    with open(apod_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                linker_accessions.add(line[1:].strip())

    rows = []

    for record in read_labelled_fasta(fasta):
        accession = record.accession

        hhm = pred_dir / "hhm" / f"{accession}.hhm"
        disorder = pred_dir / "disorder" / f"{accession}.txt"
        ss = pred_dir / "SS" / accession / "seq.phipsi"

        available = {
            "hhm": hhm.is_file(),
            "disorder": disorder.is_file(),
            "ss": ss.is_file(),
            "linker": accession in linker_accessions
        }

        missing = [name for name, found in available.items() if not found]

        if missing:
            raise FileNotFoundError(
                f"Missing predictions for {accession}: {', '.join(missing)}"
            )

        rows.append({
            "accession": accession,
            "label": record.label,
            "hhm_raw_path": str(hhm.resolve()),
            "hhm_available": True,
            "disorder_raw_path": str(disorder.resolve()),
            "disorder_available": True,
            "ss_raw_path": str(ss.resolve()),
            "ss_available": True,
            "linker_available": True
        })

    manifest = output_dir / "_prediction_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    return manifest

    
def resolve(value):
    """Resolve paths relative to the current working directory."""
    return Path(value).expanduser().resolve()
            
def main():
    parser = argparse.ArgumentParser(description="Run the MSCintel feature calculation pipeline.")
    parser.add_argument("--train-fasta", default="Data/train.fasta")
    parser.add_argument("--test-fasta", default="Data/test.fasta")
    parser.add_argument("--prediction-dir", default="Data/standardized_predictions")
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--datasets", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    script_dir = project_root / "scripts"
    feature_specs = DEFAULT_FEATURE_SPECS
    
    fasta_paths = {
        "train": resolve(args.train_fasta),
        "test": resolve(args.test_fasta)
    }
    
    prediction_root = resolve(args.prediction_dir)
    feature_root = resolve(args.output_dir)
    feature_root.mkdir(parents=True, exist_ok=True)

    print("Start feature calculation pipeline.")
    print(f"Project root: {project_root}")
    print(f"Feature output root: {feature_root}")

    required_scripts = [spec["script"] for spec in feature_specs.values()]

    for script in required_scripts:
        script_path = script_dir / script

        if not script_path.is_file():
            raise FileNotFoundError(f"Required script not found: {script_path}")

    for dataset in args.datasets:
        print(f"\n========== Calculating features for dataset: {dataset} ==========")
    
        fasta = fasta_paths[dataset]
        prediction_dir = prediction_root / dataset
        dataset_output_dir = feature_root / dataset
    
        # Generate temporary prediction manifest
        manifest = create_temporary_manifest(dataset, fasta, prediction_dir, dataset_output_dir)
    
        for feature_name, spec in feature_specs.items():
            script = spec["script"]
    
            command = [
                sys.executable,
                str(script_dir / script),
                "--dataset", dataset,
                "--input-fasta", str(fasta),
                "--standardized-prediction-dir", str(prediction_dir),
                "--manifest-path", str(manifest),
                "--output-dir", str(dataset_output_dir),
            ]
    
            if args.allow_missing:
                command.append("--allow-missing")
    
            run_command(command, project_root)
    
    if not args.skip_check:
        check_feature_outputs(feature_root=feature_root, fasta_paths=fasta_paths, datasets=args.datasets, feature_specs=feature_specs)

    
    # Remove temporary manifests
    for dataset in args.datasets:
        manifest = feature_root / dataset / "_prediction_manifest.csv"
    
        if manifest.is_file():
            manifest.unlink()
    
    
    print("\nCompleted!!")


if __name__ == "__main__":
    main()