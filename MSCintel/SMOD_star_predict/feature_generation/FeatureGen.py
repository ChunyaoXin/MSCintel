# -*- coding: utf-8 -*-
"""
Run the MSCintel feature calculation pipeline.

Supported modes
---------------
1. train-test
   Old workflow for labelled train/test FASTA files.

2. single
   One FASTA file plus one raw prediction directory with HHM/IDR/Linker/SS.

3. programmatic prediction interface
   The function calculate_features_for_prediction() is designed to be called by
   run_prediction_from_features_v2.py. It calculates feature values into a
   temporary directory, loads them into memory, optionally keeps only the selected
   model features, and optionally saves the calculated feature CSV files.
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


FEATURE_SCRIPTS = [
    "calculate_AA_frequency.py",
    "calculate_AA_pairs.py",
    "calculate_AAindex.py",
    "calculate_Conservation_features.py",
    "calculate_disorder_features.py",
    "calculate_linker_features.py",
    "calculate_SS_features.py",
    "calculate_SS_RSA_features.py"]

EXPECTED_FEATURE_FILES = [
    "AA_frequency_features.csv",
    "AA_pairs_features.csv",
    "AAindex_features.csv",
    "Conservation_features.csv",
    "disorder_features.csv",
    "linker_features.csv",
    "SS_features.csv",
    "SS_RSA_features.csv"]

FEATURE_FILES = {
    "AA_frequency": "AA_frequency_features.csv",
    "AA_pairs": "AA_pairs_features.csv",
    "AAindex": "AAindex_features.csv",
    "Conservation": "Conservation_features.csv",
    "disorder": "disorder_features.csv",
    "linker": "linker_features.csv",
    "SS": "SS_features.csv",
    "SS_RSA": "SS_RSA_features.csv"}

# Do not include "ID" here, because ID can be a real CKSAAP feature: I-D.
SKIP_COLUMNS = {"AccessionName", "accession", "label", "Label"}


def expected_feature_files(prefix=""):
    if prefix:
        return [f"{prefix}_{name}" for name in EXPECTED_FEATURE_FILES]
    return EXPECTED_FEATURE_FILES


def add_prefix_to_filename(filename, prefix=""):
    if prefix is None or prefix == "":
        return filename
    return f"{prefix}_{filename}"


def run_command(command, project_root):
    """Run one command and stop immediately if it fails."""
#    print("\nRunning:")
#    print(" ".join(str(x) for x in command))
    subprocess.run(command, cwd=project_root, check=True)


def count_fasta_records(fasta_path):
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    count = 0
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def check_one_feature_output(dataset_name, fasta_path, output_dir, prefix="", require_label=False):
    """Check expected feature files for one dataset."""
    expected_rows = count_fasta_records(fasta_path)
    output_dir = Path(output_dir)

#    print(f"\nChecking feature outputs for {dataset_name}...")
    for file_name in expected_feature_files(prefix):
        path = output_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")

        df = pd.read_csv(path)
        if "AccessionName" not in df.columns:
            raise ValueError(f"{path} does not contain AccessionName column")
        if require_label and "label" not in df.columns:
            raise ValueError(f"{path} does not contain label column")
        if df["AccessionName"].duplicated().any():
            duplicated = df.loc[df["AccessionName"].duplicated(), "AccessionName"].tolist()
            raise ValueError(f"{path} contains duplicated AccessionName: {duplicated[:5]}")
        if len(df) != expected_rows:
            raise ValueError(f"{path} row number mismatch: got {len(df)}, expected {expected_rows}")
        if df.isna().any().any():
            raise ValueError(f"{path} contains NaN values")
#        print(f"  OK: {file_name}, shape={df.shape}")


def check_train_test_feature_outputs(project_root, datasets):
    """Check whether all expected train/test feature files exist and are consistent."""
    feature_root = project_root / "outputs" / "features"
    fasta_paths = {
        "train": project_root / "Data" / "train.fasta",
        "test": project_root / "Data" / "test.fasta",
    }

    for dataset in datasets:
        if dataset not in fasta_paths:
            continue
        check_one_feature_output(dataset, fasta_paths[dataset], feature_root / dataset, require_label=True)

    if "train" in datasets and "test" in datasets:
#        print("\nChecking train/test column consistency...")
        for file_name in EXPECTED_FEATURE_FILES:
            train_df = pd.read_csv(feature_root / "train" / file_name)
            test_df = pd.read_csv(feature_root / "test" / file_name)
            if list(train_df.columns) != list(test_df.columns):
                raise ValueError(f"{file_name}: train/test columns are inconsistent")
#            print(f"  OK: {file_name}")

#    print("\nAll feature output checks passed.")


def resolve_cli_path(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_required_scripts(script_dir, include_merge=False):
    required_scripts = ["standardize_predictions.py", *FEATURE_SCRIPTS]
    if include_merge:
        required_scripts.insert(0, "merge_fasta.py")
    for script in required_scripts:
        script_path = script_dir / script
        if not script_path.is_file():
            raise FileNotFoundError(f"Required script not found: {script_path}")


def find_feature_file(feature_dir, filename, prefix=""):
    """Find a feature file. Try the requested prefix first, then the unprefixed name."""
    feature_dir = Path(feature_dir)
    candidates = [feature_dir / add_prefix_to_filename(filename, prefix)]
    if prefix:
        candidates.append(feature_dir / filename)

    for path in candidates:
        if path.is_file():
            return path

    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Feature file is missing. Tried:\n{candidate_text}")


def load_feature_frames(feature_dir, prefix=""):
    """Load the eight calculated feature CSV files as DataFrames."""
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        raise NotADirectoryError(f"Feature directory does not exist: {feature_dir}")

    frames = {}
    for block_name, filename in FEATURE_FILES.items():
        path = find_feature_file(feature_dir, filename, prefix=prefix)
        df = pd.read_csv(path)
        if "AccessionName" not in df.columns:
            raise ValueError(f"`AccessionName` column is missing in {path}")
        frames[block_name] = df
    return frames


def merge_feature_frames(feature_frames, selected_features=None):
    accession = None
    true_label = None
    feature_blocks = []

    for block_name, df in feature_frames.items():
        current_accession = df["AccessionName"].astype(str).tolist()
        if accession is None:
            accession = current_accession
        elif current_accession != accession:
            raise ValueError(
                f"AccessionName order is inconsistent in feature block `{block_name}`. "
                "Please make sure all feature files are generated from the same FASTA in the same order."
            )

        if "label" in df.columns:
            current_label = df["label"].tolist()
            if true_label is None:
                true_label = current_label
            elif current_label != true_label:
                raise ValueError(f"`label` column is inconsistent in feature block `{block_name}`")

        feature_cols = [col for col in df.columns if col not in SKIP_COLUMNS]
        if not feature_cols:
            raise ValueError(f"No feature columns found in feature block `{block_name}`")
        feature_blocks.append(df[feature_cols])

    matrix = pd.concat(feature_blocks, axis=1)

    duplicated = matrix.columns[matrix.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicated feature names found after merging feature files. "
            f"Examples: {duplicated[:20]}"
        )

    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        bad_cols = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(
            "NaN or non-numeric values were found in merged feature matrix. "
            f"Columns with problems include: {bad_cols[:30]}"
        )

    if selected_features is not None:
        selected_features = list(selected_features)
        missing_features = [feature for feature in selected_features if feature not in matrix.columns]
        if missing_features:
            raise ValueError(
                "Selected features are missing from calculated feature files: "
                f"{missing_features[:30]}"
            )
        matrix = matrix[selected_features]

    return matrix, pd.Series(accession, name="AccessionName"), true_label


def save_feature_frames(feature_frames, output_dir, prefix="", selected_features=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_set = set(selected_features) if selected_features is not None else None

    for block_name, filename in FEATURE_FILES.items():
        df = feature_frames[block_name]
        if selected_set is None:
            out_df = df.copy()
        else:
            selected_cols = [col for col in df.columns if col in selected_set]
            out_cols = ["AccessionName", *selected_cols]
            if "label" in df.columns:
                out_cols.append("label")
            out_df = df[out_cols].copy()

        output_path = output_dir / add_prefix_to_filename(filename, prefix)
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
#        print(f"Saved calculated feature file: {output_path}, shape={out_df.shape}")

def save_selected_feature_matrix(matrix, accession, true_label, output_dir, output_filename="feature_values.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.concat([accession.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1)
    if true_label is not None and len(true_label) == len(out_df):
        out_df.insert(1, "label", true_label)
    output_path = output_dir / output_filename
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

#    print(f"Saved selected-feature matrix to: {output_path}, shape={out_df.shape}")


def run_single_feature_scripts(*, project_root, script_dir, config_path, input_fasta, raw_prediction_dir,
    dataset_name, standardized_root, output_dir, prefix="", skip_standardize=False, allow_missing=False):
    """Run standardization and the eight feature calculation scripts for one FASTA."""
    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)
    standardized_root = Path(standardized_root)

    if not skip_standardize:
        if raw_prediction_dir is None:
            raise ValueError("raw_prediction_dir is required unless skip_standardize=True")
        command = [
            sys.executable,
            str(script_dir / "standardize_predictions.py"),
            "--config", str(config_path),
            "--mode", "single",
            "--input-fasta", str(input_fasta),
            "--raw-prediction-dir", str(raw_prediction_dir),
            "--dataset-name", dataset_name,
            "--standardized-prediction-dir", str(standardized_root)]
        run_command(command, project_root)

    standardized_dataset_dir = standardized_root / dataset_name
    if not standardized_dataset_dir.is_dir():
        raise NotADirectoryError(f"Standardized prediction dataset directory does not exist: {standardized_dataset_dir}")

    for script in FEATURE_SCRIPTS:
        command = [
            sys.executable,
            str(script_dir / script),
            "--dataset", "prediction",
            "--dataset-name", dataset_name,
            "--input-fasta", str(input_fasta),
            "--output-dir", str(output_dir),
            "--prefix", prefix,
            "--config", str(config_path),
            "--standardized-prediction-dir", str(standardized_dataset_dir),
        ]
        if allow_missing:
            command.append("--allow-missing")
        run_command(command, project_root)


def calculate_features_for_prediction(
    *,
    input_fasta,
    raw_prediction_dir=None,
    model_selected_features=None,
    selected_features=None,
    config_path="config.yaml",
    dataset_name="prediction",
    standardized_prediction_dir=None,
    feature_output_dir=None,
    prefix="",
    save_calculated_features=False,
    skip_standardize=False,
    allow_missing=False):
    """Calculate prediction features and return the matrix directly in memory."""
    if selected_features is None and model_selected_features is not None:
        selected_features = model_selected_features
    if selected_features is not None:
        selected_features = list(selected_features)

    project_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    ensure_required_scripts(script_dir, include_merge=False)

    config_path = resolve_cli_path(project_root, config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    input_fasta = resolve_cli_path(project_root, input_fasta)
    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")

    raw_prediction_dir = (resolve_cli_path(project_root, raw_prediction_dir) if raw_prediction_dir is not None else None)

    # Use temporary directories by default so the application-stage prediction
    # pipeline does not leave full 3019-feature CSV files on disk.
    with tempfile.TemporaryDirectory(prefix="mscintel_features_") as tmp_feature_dir, tempfile.TemporaryDirectory(prefix="mscintel_standardized_") as tmp_standardized_dir:
        working_feature_dir = Path(tmp_feature_dir)
        standardized_root = (resolve_cli_path(project_root, standardized_prediction_dir) if standardized_prediction_dir is not None else Path(tmp_standardized_dir))

#        print("Calculating feature values for prediction...")
        run_single_feature_scripts(
            project_root=project_root,
            script_dir=script_dir,
            config_path=config_path,
            input_fasta=input_fasta,
            raw_prediction_dir=raw_prediction_dir,
            dataset_name=dataset_name,
            standardized_root=standardized_root,
            output_dir=working_feature_dir,
            prefix=prefix,
            skip_standardize=skip_standardize,
            allow_missing=allow_missing,
        )

        check_one_feature_output(dataset_name, input_fasta, working_feature_dir, prefix=prefix, require_label=False,)

        feature_frames = load_feature_frames(working_feature_dir, prefix=prefix)
        matrix, accession, true_label = merge_feature_frames(feature_frames, selected_features=selected_features,)

        if save_calculated_features:
            if feature_output_dir is None:
                feature_output_dir = project_root / "outputs" / "features" / dataset_name
            else:
                feature_output_dir = resolve_cli_path(project_root, feature_output_dir)
        
            save_selected_feature_matrix(matrix=matrix, accession=accession, true_label=true_label, output_dir=feature_output_dir, output_filename=f"feature_values.csv")

        return matrix, accession, true_label


def main():
    parser = argparse.ArgumentParser(description="Run the MSCintel feature calculation pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml. Default: config.yaml")
    parser.add_argument("--mode", choices=["train-test", "single"], default="train-test")

    # Old train/test workflow arguments
    parser.add_argument("--datasets", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge_fasta.py in train-test mode.")

    # Single-dataset workflow arguments
    parser.add_argument("--input-fasta", default=None, help="Single input FASTA file.")
    parser.add_argument("--raw-prediction-dir", default=None, help="Raw prediction root containing HHM/ IDR/ Linker/ SS/.")
    parser.add_argument("--dataset-name", default="prediction", help="Name for the single dataset output folder.")
    parser.add_argument("--standardized-prediction-dir", default=None, help="Standardized prediction root. Default: config value.")
    parser.add_argument("--output-dir", default=None, help="Output directory for single-dataset features.")
    parser.add_argument("--prefix", default="", help="Output filename prefix in single mode. Use predict for predict_* files.")

    parser.add_argument("--skip-standardize", action="store_true", help="Skip standardize_predictions.py.")
    parser.add_argument("--allow-missing", action="store_true", help="Pass --allow-missing to feature calculation scripts.")
    parser.add_argument("--skip-check", action="store_true", help="Skip output validation after feature calculation.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    config_path = resolve_cli_path(project_root, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ensure_required_scripts(script_dir, include_merge=(args.mode == "train-test"))

#    print("Start feature calculation pipeline.")

    if args.mode == "single":
        if not args.input_fasta:
            raise ValueError("--input-fasta is required when --mode single")
        if not args.raw_prediction_dir and not args.skip_standardize:
            raise ValueError("--raw-prediction-dir is required when --mode single and standardization is not skipped")

        input_fasta = resolve_cli_path(project_root, args.input_fasta)
        output_dir = resolve_cli_path(project_root, args.output_dir) if args.output_dir else project_root / "outputs" / "features" / args.dataset_name

        standardized_root = (resolve_cli_path(project_root, args.standardized_prediction_dir) if args.standardized_prediction_dir else project_root / "predicted_features" / "standardized")

        run_single_feature_scripts(
            project_root=project_root,
            script_dir=script_dir,
            config_path=config_path,
            input_fasta=input_fasta,
            raw_prediction_dir=resolve_cli_path(project_root, args.raw_prediction_dir) if args.raw_prediction_dir else None,
            dataset_name=args.dataset_name,
            standardized_root=standardized_root,
            output_dir=output_dir,
            prefix=args.prefix,
            skip_standardize=args.skip_standardize,
            allow_missing=args.allow_missing)

        if not args.skip_check:
            check_one_feature_output(args.dataset_name, input_fasta, output_dir, prefix=args.prefix, require_label=False,)

    else:
        if not args.skip_merge:
            run_command([sys.executable, str(script_dir / "merge_fasta.py"), "--config", str(config_path)], project_root)

        if not args.skip_standardize:
            run_command([sys.executable, str(script_dir / "standardize_predictions.py"), "--config", str(config_path)], project_root)

        for dataset in args.datasets:
#            print(f"\n========== Calculating features for dataset: {dataset} ==========")
            for script in FEATURE_SCRIPTS:
                command = [sys.executable, str(script_dir / script), "--dataset", dataset, "--config", str(config_path)]
                if args.allow_missing:
                    command.append("--allow-missing")
                run_command(command, project_root)

        if not args.skip_check:
            check_train_test_feature_outputs(project_root, args.datasets)

    print("\nCompleted!!")


if __name__ == "__main__":
    main()
