# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd
import yaml

from workflow_utils import read_labelled_fasta, read_optional_label_fasta


DEFAULT_FEATURE_SPECS = {
    "AA_frequency": {"script": "calculate_AA_frequency.py", "output": "AA_frequency_features.csv"},
    "AA_pairs": {"script": "calculate_AA_pairs.py", "output": "AA_pairs_features.csv"},
    "AAindex": {"script": "calculate_AAindex.py", "output": "AAindex_features.csv"},
    "Conservation": {"script": "calculate_Conservation_features.py", "output": "Conservation_features.csv"},
    "disorder": {"script": "calculate_disorder_features.py", "output": "disorder_features.csv"},
    "linker": {"script": "calculate_linker_features.py", "output": "linker_features.csv"},
    "SS": {"script": "calculate_SS_features.py", "output": "SS_features.csv"},
    "SS_RSA": {"script": "calculate_SS_RSA_features.py", "output": "SS_RSA_features.csv"},
}


def load_config(config_path):
    """Load YAML configuration file."""
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty config file: {config_path}")

    return config, config_path


def resolve_project_root(config, config_path):
    """Resolve the project root relative to the config file."""
    repo_root = Path(config.get("repo_root", "."))

    if repo_root.is_absolute():
        return repo_root.resolve()

    return (config_path.parent / repo_root).resolve()


def resolve_path(project_root, path_value):
    """Resolve a path relative to the project root."""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def get_feature_specs(config):
    """Get feature calculation scripts and expected output filenames."""
    feature_cfg = config.get("feature_calculation", {})
    custom_specs = feature_cfg.get("feature_specs")

    if custom_specs is None:
        return DEFAULT_FEATURE_SPECS

    feature_specs = {}

    for name, spec in custom_specs.items():
        if "script" not in spec or "output" not in spec:
            raise ValueError(f"feature_calculation.feature_specs.{name} " "must contain 'script' and 'output'.")

        feature_specs[name] = {"script": spec["script"], "output": spec["output"],}

    return feature_specs


def run_command(command, project_root):
    """Run a command and stop if it fails."""
    subprocess.run(command, cwd=project_root, check=True)


def get_run_datasets(config, project_root, args):
    """Resolve input and output paths for the selected mode."""

    mode = "train" if args.train else "predict"

    if mode not in config:
        raise ValueError(f"Missing '{mode}' section in config.yaml")

    feature_cfg = config[mode]["feature_generation"]

    def get_path(key, override=None):
        value = override if override is not None else feature_cfg.get(key)

        if value is None:
            raise ValueError(f"Missing configuration parameter: {mode}.feature_generation.{key}")

        return resolve_path(project_root, value)

    if mode == "train":
        prediction_root = get_path("predicted_features_dir", args.predicted_features_dir,)

        datasets = {
            "train": {
                "fasta": get_path("train_fasta", args.train_fasta),
                "prediction_dir": prediction_root / "train",
                "output_dir": get_path("train_features_dir", args.train_features_dir,),
                "require_label": True,
            },
            "test": {
                "fasta": get_path("test_fasta", args.test_fasta),
                "prediction_dir": prediction_root / "test",
                "output_dir": get_path("test_features_dir", args.test_features_dir,),
                "require_label": True,
            },
        }

    else:
        datasets = {
            "prediction": {
                "fasta": get_path("input_fasta", args.input_fasta),
                "prediction_dir": get_path("feature_data_dir", args.feature_data_dir),
                "output_dir": get_path("output_dir", args.output_dir),
                "require_label": False
            }
        }

    return mode, datasets


def read_input_records(fasta, require_label):
    """Read FASTA records and validate accession names and labels."""

    reader = read_labelled_fasta if require_label else read_optional_label_fasta
    records = list(reader(fasta))

    if not records:
        raise ValueError(f"No FASTA records found: {fasta}")

    seen = set()

    for record in records:
        accession = record.accession

        if accession in seen:
            raise ValueError(f"Duplicated FASTA accession: {accession}")

        seen.add(accession)

        if require_label and record.label not in (0, 1):
            raise ValueError(f"Invalid or missing label for {accession}")

    return records


def check_input_files(fasta, prediction_dir, require_label):
    """Validate FASTA and external prediction directory."""

    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta}")

    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    records = read_input_records(fasta, require_label)

    print(f"  OK: FASTA, {len(records)} sequences")
    print(f"  OK: Prediction directory: {prediction_dir}")

    return records


def create_temporary_manifest(records, pred_dir):
    """Create a manifest mapping FASTA accessions to prediction files."""

    apod_path = pred_dir / "linker.apod"

    if not apod_path.is_file():
        raise FileNotFoundError(f"Missing linker file: {apod_path}")

    linker_accessions = set()

    with open(apod_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                accession = line[1:].strip()

                if accession in linker_accessions:
                    raise ValueError(f"Duplicated linker accession: {accession}")

                linker_accessions.add(accession)

    rows = []

    for record in records:
        accession = record.accession

        hhm = pred_dir / "hhm" / f"{accession}.hhm"
        disorder = pred_dir / "disorder" / f"{accession}.txt"
        ss = pred_dir / "SS" / accession / "seq.phipsi"

        available = {
            "hhm": hhm.is_file(),
            "disorder": disorder.is_file(),
            "ss": ss.is_file(),
            "linker": accession in linker_accessions,
        }

        missing = [name for name, found in available.items() if not found]

        if missing:
            raise FileNotFoundError(f"Missing predictions for {accession}: {', '.join(missing)}")

        rows.append({
            "accession": accession,
            "label": record.label,
            "hhm_raw_path": str(hhm.resolve()),
            "hhm_available": True,
            "disorder_raw_path": str(disorder.resolve()),
            "disorder_available": True,
            "ss_raw_path": str(ss.resolve()),
            "ss_available": True,
            "linker_available": True,
        })

    manifest = pred_dir / "prediction_manifest.csv"

    if manifest.exists():
        raise FileExistsError(f"Manifest already exists: {manifest}. Remove or back up the existing file before running.")

    pd.DataFrame(rows).to_csv(manifest, index=False)

    return manifest


def check_feature_outputs(records, output_dir, feature_specs, require_label):
    """Validate calculated features against FASTA records."""

    fasta_records = {record.accession: record.label for record in records}

    expected_accessions = set(fasta_records)
    expected_rows = len(fasta_records)

    print(f"\nChecking feature outputs: {output_dir}")
    print(f"Expected rows: {expected_rows}")

    for feature_name, spec in feature_specs.items():
        path = output_dir / spec["output"]

        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")

        df = pd.read_csv(path)

        if "AccessionName" not in df.columns:
            raise ValueError(f"{path} does not contain AccessionName column")

        if require_label and "label" not in df.columns:
            raise ValueError(f"{path} does not contain label column")

        if df["AccessionName"].isna().any():
            raise ValueError(f"{path} contains missing accession names")

        if df["AccessionName"].duplicated().any():
            raise ValueError(f"{path} contains duplicated AccessionName")

        if len(df) != expected_rows:
            raise ValueError(f"{path} row number mismatch: got {len(df)}, expected {expected_rows}")

        actual_accessions = set(df["AccessionName"])

        if actual_accessions != expected_accessions:
            raise ValueError(f"{path}: AccessionName does not match input FASTA")

        # Validate calculated feature values.
        feature_columns = [col for col in df.columns if col not in ("AccessionName", "label")]

        if not feature_columns:
            raise ValueError(f"{path} contains no calculated feature columns")

        if df[feature_columns].isna().any().any():
            raise ValueError(f"{path} contains NaN feature values")

        # Validate optional labels.
        if "label" in df.columns:
            actual_labels = pd.to_numeric(df["label"], errors="coerce")

            expected_labels = pd.to_numeric(df["AccessionName"].map(fasta_records), errors="coerce")

            actual_values = actual_labels.to_numpy()
            expected_values = expected_labels.to_numpy()

            import numpy as np

            if not np.array_equal(actual_values, expected_values, equal_nan=True,):
                raise ValueError(f"{path}: Labels do not match input FASTA")

        elif any(label is not None for label in fasta_records.values()):
            raise ValueError(f"{path}: FASTA contains labels, but the feature CSV has no label column")

        print(f"  OK: {feature_name}: {spec['output']}, shape={df.shape}")

    print("\nAll feature output checks passed.")


def check_train_test_columns(datasets, feature_specs):
    """Validate train/test feature-column consistency."""

    train_dir = datasets["train"]["output_dir"]
    test_dir = datasets["test"]["output_dir"]

    print("\nChecking train/test column consistency...")

    for feature_name, spec in feature_specs.items():
        train_path = train_dir / spec["output"]
        test_path = test_dir / spec["output"]

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError(f"{feature_name}: train/test columns are inconsistent")

        print(f"  OK: {feature_name}")


def main():
    parser = argparse.ArgumentParser(description="Run the MSCintel feature calculation pipeline.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Calculate labelled train/test features.")
    mode_group.add_argument("--predict", action="store_true", help="Calculate features for prediction data.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    # Optional command-line path overrides.
    parser.add_argument("--train-fasta", default=None)
    parser.add_argument("--test-fasta", default=None)
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--predicted-features-dir", default=None)
    parser.add_argument("--train-features-dir", default=None)
    parser.add_argument("--test-features-dir", default=None)
    parser.add_argument("--feature-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-check", action="store_true", help="Skip feature-output validation.")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    project_root = resolve_project_root(config, config_path)
    script_dir = Path(__file__).resolve().parent

    feature_specs = get_feature_specs(config)

    mode, datasets = get_run_datasets(config, project_root, args)

    print("Start feature calculation pipeline.")
    print(f"Mode: {mode}")
    print(f"Project root: {project_root}")
    print(f"Config file: {config_path}")

    required_scripts = [spec["script"] for spec in feature_specs.values()]

    for script in required_scripts:
        script_path = script_dir / script

        if not script_path.is_file():
            raise FileNotFoundError(f"Required script not found: {script_path}")

    for dataset, paths in datasets.items():
        print(f"\n========== Calculating features: {dataset} ==========")

        fasta = paths["fasta"]
        prediction_dir = paths["prediction_dir"]
        output_dir = paths["output_dir"]
        require_label = paths["require_label"]

        records = check_input_files(fasta, prediction_dir, require_label,)

        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = None

        try:
            manifest = create_temporary_manifest(records, prediction_dir,)

            for feature_name, spec in feature_specs.items():
                command = [
                    sys.executable,
                    str(script_dir / spec["script"]),
                    "--config", str(config_path),
                    "--dataset", dataset,
                    "--input-fasta", str(fasta),
                    "--standardized-prediction-dir", str(prediction_dir),
                    "--manifest-path", str(manifest),
                    "--output-dir", str(output_dir),
                ]

                print(f"Calculating: {feature_name}")
                run_command(command, project_root)

            if not args.skip_check:
                check_feature_outputs(records, output_dir, feature_specs, require_label,)

        finally:
            # Only remove the manifest created in this run.
            if manifest is not None and manifest.is_file():
                manifest.unlink()

    if mode == "train" and not args.skip_check:
        check_train_test_columns(datasets, feature_specs)

    print("\nCompleted!!")


if __name__ == "__main__":
    main()