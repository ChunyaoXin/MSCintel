
# -*- coding: utf-8 -*-
"""
Predict mechanosensitive ion channels using the trained SMOD_star model.

Usage:
    # Predict using paths from config.yaml
    python SMOD_star_predict.py --config config.yaml

    # Predict with custom input and output paths
    python SMOD_star_predict.py.py \
        --config config.yaml \
        --input-fasta example/input.fasta \
        --feature-data-dir example/predicted_features \
        --output-dir outputs/predict/features \
        --output outputs/predict/predictions.csv

    # Save calculated feature CSV files
    python SMOD_star_predict.py --config config.yaml --save-calculated-features true
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_FILES = {
    "AA_frequency": "AA_frequency_features.csv",
    "AA_pairs": "AA_pairs_features.csv",
    "AAindex": "AAindex_features.csv",
    "Conservation": "Conservation_features.csv",
    "disorder": "disorder_features.csv",
    "linker": "linker_features.csv",
    "SS": "SS_features.csv",
    "SS_RSA": "SS_RSA_features.csv",
}

SKIP_COLUMNS = {"AccessionName", "accession", "label", "Label"}


def str2bool(value):
    """Convert a string or boolean into a boolean."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Expected TRUE or FALSE.")


def resolve_path(project_root, value):
    """Resolve paths relative to the configured project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(config_path):
    """Load the nested YAML configuration."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty config file: {path}")

    project_root = resolve_path(path.parent, config.get("repo_root", "."))
    return config, path, project_root


def get_model_path(project_root, config, model_value):
    """Resolve a model filename against the trained Stacking model directory."""
    path = Path(model_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    if path.parent != Path("."):
        return resolve_path(project_root, path)

    model_dir = config["train"]["feature_selection"]["stacking_output_dir"]
    return resolve_path(project_root, Path(model_dir) / "model" / path)


def load_selected_features(selected_features_path=None, model_package=None):
    """Load the selected feature names in their training order."""
    if isinstance(model_package, dict):
        for key in ("feature_list", "Feature_list", "selected_features", "feature_names", "required_order", "features"):
            value = model_package.get(key)
            if value is not None and len(value) > 0:
                return list(value)

    if selected_features_path is None:
        raise ValueError("Selected feature list is missing from the model package. Provide --selected-features.")

    path = Path(selected_features_path)
    if not path.is_file():
        raise FileNotFoundError(f"Selected feature file not found: {path}")

    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        for key in ("selected_features", "feature_names", "required_order", "features"):
            if key in obj and len(obj[key]) > 0:
                return list(obj[key])
        raise ValueError(f"No selected feature list found in: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for column in ("feature", "Feature", "selected_features", "feature_name"):
            if column in df.columns:
                return df[column].dropna().astype(str).tolist()
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(str).tolist()
        raise ValueError(f"Cannot identify the selected feature column in: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Selected feature file is empty: {path}")
    return [item.strip() for item in content.split(",") if item.strip()] if "\n" not in content else [line.strip() for line in content.splitlines() if line.strip()]


def load_feature_matrix(feature_dir):
    """Combine eight feature CSV files into one aligned feature matrix."""
    feature_dir = Path(feature_dir)

    if not feature_dir.is_dir():
        raise NotADirectoryError(f"Feature directory not found: {feature_dir}")

    accession = None
    true_label = None
    feature_blocks = []
    label_presence = []

    for feature_name, filename in FEATURE_FILES.items():
        path = feature_dir / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")

        df = pd.read_csv(path)

        if "AccessionName" not in df.columns:
            raise ValueError(f"Missing AccessionName column in: {path}")

        if df["AccessionName"].isna().any() or df["AccessionName"].duplicated().any():
            raise ValueError(f"Invalid or duplicated accessions in: {path}")

        current_accession = df["AccessionName"].astype(str).tolist()

        if accession is None:
            accession = current_accession
        elif current_accession != accession:
            raise ValueError(f"Inconsistent accession order in: {path}")

        has_label = "label" in df.columns
        label_presence.append(has_label)

        if has_label:
            current_label = pd.to_numeric(df["label"], errors="coerce").tolist()
            if true_label is None:
                true_label = current_label
            elif not np.array_equal(np.asarray(current_label), np.asarray(true_label), equal_nan=True):
                raise ValueError(f"Inconsistent labels in: {path}")

        feature_columns = [col for col in df.columns if col not in SKIP_COLUMNS]

        if not feature_columns:
            raise ValueError(f"No feature columns found in: {path}")

        feature_blocks.append(df[feature_columns])

    if any(label_presence) and not all(label_presence):
        raise ValueError("Inconsistent label column presence across feature CSV files.")

    matrix = pd.concat(feature_blocks, axis=1)

    duplicated = matrix.columns[matrix.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicated feature names: {duplicated[:20]}")

    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.isna().any().any():
        bad_columns = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"NaN or non-numeric feature values: {bad_columns[:20]}")

    return matrix, pd.Series(accession, name="AccessionName"), true_label


def get_model_and_threshold(model_path, threshold_arg=None):
    """Load a trained estimator or a custom Stacking model package."""
    package = joblib.load(model_path)

    if isinstance(package, dict):
        threshold = float(threshold_arg) if threshold_arg is not None else float(package.get("threshold", 0.5))
        direct_model = package.get("model", package.get("estimator"))

        if direct_model is not None:
            return direct_model, threshold, package

        if package.get("base_models") is not None and package.get("meta_model") is not None:
            return package, threshold, package

        raise ValueError(f"Unsupported model package structure: {list(package.keys())}")

    threshold = float(threshold_arg) if threshold_arg is not None else 0.5
    return package, threshold, None


def positive_class_probability(model, matrix, model_name="model"):
    """Get probabilities for class 1 from a fitted classifier."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"{model_name} does not support predict_proba().")

    probability = np.asarray(model.predict_proba(matrix))

    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError(f"{model_name} must return a two-class probability matrix.")

    classes = getattr(model, "classes_", None)
    positive_index = list(classes).index(1) if classes is not None and 1 in classes else 1
    return probability[:, positive_index]


def predict_probability(model, matrix):
    """Predict using a single model or the saved custom Stacking package."""
    if not isinstance(model, dict):
        return positive_class_probability(model, matrix)

    base_models = model.get("base_models")
    meta_model = model.get("meta_model")

    if base_models is None or meta_model is None:
        raise ValueError("Stacking package must contain base_models and meta_model.")

    if isinstance(base_models, dict):
        model_items = list(base_models.items())
    elif isinstance(base_models, (list, tuple)):
        model_items = [(f"base_model_{i + 1}", item) for i, item in enumerate(base_models)]
    else:
        raise TypeError("base_models must be a dictionary, list, or tuple.")

    if not model_items:
        raise ValueError("base_models is empty.")

    meta_features = np.column_stack([
        positive_class_probability(base_model, matrix, model_name=str(name))
        for name, base_model in model_items
    ])

    expected_features = getattr(meta_model, "n_features_in_", None)
    if expected_features is not None and meta_features.shape[1] != expected_features:
        raise ValueError(f"Meta-model expects {expected_features} features, got {meta_features.shape[1]}.")

    return positive_class_probability(meta_model, meta_features, model_name="meta_model")


def validate_selected_features(matrix, selected_features):
    """Select the exact features required by the trained model."""
    if not selected_features:
        raise ValueError("Selected feature list is empty.")

    if len(selected_features) != len(set(selected_features)):
        raise ValueError("Selected feature list contains duplicated names.")

    missing = [name for name in selected_features if name not in matrix.columns]

    if missing:
        raise ValueError(f"Selected features are missing: {missing[:30]}")

    return matrix.loc[:, selected_features]


def run_feature_generation(config_path, input_fasta, feature_data_dir, output_dir):
    """Call the unified FeatureGen.py prediction workflow."""
    script = PROJECT_ROOT / "feature_generation" / "FeatureGen.py"

    if not script.is_file():
        raise FileNotFoundError(f"Feature generation script not found: {script}")

    command = [
        sys.executable, str(script),
        "--predict", "--config", str(config_path),
        "--input-fasta", str(input_fasta),
        "--feature-data-dir", str(feature_data_dir),
        "--output-dir", str(output_dir),
    ]

    print("\nRunning feature generation...", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Predict mechanosensitive ion channels using the trained SMOD_star model.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--input-fasta", default=None, help="FASTA containing sequences to predict.")
    parser.add_argument("--feature-data-dir", default=None, help="Directory containing prepared HHM, disorder, SS, and linker predictions.")
    parser.add_argument("--output-dir", default=None, help="Directory for calculated feature CSV files.")
    parser.add_argument("--feature-dir", default=None, help="Use existing feature CSV files instead of running feature generation.")
    parser.add_argument("--model", default=None, help="Trained model filename or path.")
    parser.add_argument("--selected-features", default=None, help="Optional file containing selected feature names.")
    parser.add_argument("--output", default=None, help="Output CSV for prediction results.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the classification threshold.")
    parser.add_argument("--save-calculated-features", type=str2bool, default=None, help="Save calculated feature CSV files: true or false.")
    args = parser.parse_args()

    config, config_path, project_root = load_config(args.config)
    fg_cfg = config["predict"]["feature_generation"]
    pred_cfg = config["predict"]["prediction"]

    input_fasta = resolve_path(project_root, args.input_fasta or fg_cfg["input_fasta"])
    feature_data_dir = resolve_path(project_root, args.feature_data_dir or fg_cfg["feature_data_dir"])
    feature_output_dir = resolve_path(project_root, args.output_dir or fg_cfg["output_dir"])
    prediction_output = resolve_path(project_root, args.output or pred_cfg["output"])
    model_path = get_model_path(project_root, config, args.model or pred_cfg["model"])
    selected_features_path = resolve_path(project_root, args.selected_features) if args.selected_features else None
    save_features = str2bool(args.save_calculated_features if args.save_calculated_features is not None else pred_cfg.get("save_calculated_features", True))

    print("========== SMOD_star prediction started ==========")
    print(f"Config file: {config_path}")
    print(f"Model file: {model_path}")
    print(f"Prediction output: {prediction_output}")

    # Step 1: Load the trained model.
    print("\n[1/5] Loading trained model...")

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model, threshold, package = get_model_and_threshold(model_path, args.threshold)
    selected_features = load_selected_features(selected_features_path, package)

    print(f"Selected features: {len(selected_features)}")
    print(f"Classification threshold: {threshold}")

    # Step 2: Load existing features or generate them from FASTA.
    if args.feature_dir:
        print("\n[2/5] Loading existing feature CSV files...")
        feature_dir = resolve_path(project_root, args.feature_dir)
        matrix, accession, true_label = load_feature_matrix(feature_dir)
    elif save_features:
        print("\n[2/5] Calculating and saving feature CSV files...")
        run_feature_generation(config_path, input_fasta, feature_data_dir, feature_output_dir)
        matrix, accession, true_label = load_feature_matrix(feature_output_dir)
    else:
        print("\n[2/5] Calculating features in a temporary directory...")
        with tempfile.TemporaryDirectory(prefix="mscintel_features_") as temp_dir:
            temp_feature_dir = Path(temp_dir)
            run_feature_generation(config_path, input_fasta, feature_data_dir, temp_feature_dir)
            matrix, accession, true_label = load_feature_matrix(temp_feature_dir)

    # Step 3: Select the features used by the trained model.
    print("\n[3/5] Selecting model features...")
    matrix = validate_selected_features(matrix, selected_features)

    # Step 4: Predict.
    print("\n[4/5] Predicting probabilities...")
    probability = np.asarray(predict_probability(model, matrix))
    predicted_label = (probability >= threshold).astype(int)

    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Model returned invalid prediction probabilities.")

    result = pd.DataFrame({
        "AccessionName": accession,
        "prediction_probability": probability,
        "predicted_label": predicted_label,
    })

    if true_label is not None and len(true_label) == len(result):
        result.insert(1, "true_label", true_label)

    # Step 5: Save predictions.
    print("\n[5/5] Saving prediction results...")
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(prediction_output, index=False)

    print(f"Saved predictions to: {prediction_output}")
    if args.feature_dir:
        print("Used existing calculated features.")
    elif save_features:
        print(f"Saved calculated features to: {feature_output_dir}")
    else:
        print("Temporary feature CSV files were automatically removed.")

    print("\n========== SMOD_star prediction completed ==========")


if __name__ == "__main__":
    main()
