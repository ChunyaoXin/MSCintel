"""Run the MSCintel prediction pipeline.
This version can run the full application-stage pipeline:
    FASTA + raw external prediction results + trained Stacking model
    -> calculate feature values in memory
    -> keep only the selected model features
    -> predict positive-class probability for each protein
By default, calculated intermediate feature CSV files are NOT saved.
Set save_calculated_features: true in config.yaml or use
--save-calculated-features TRUE if you want to keep the calculated feature CSV files.
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_generation.FeatureGen import calculate_features_for_prediction


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

# Do not include "ID" here, because ID can be a real CKSAAP feature: I-D.
SKIP_COLUMNS = {"AccessionName", "accession", "label", "Label"}


def str2bool(value):
    """Parse TRUE/FALSE style command-line or YAML values."""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y", "t"}:
        return True

    if value in {"false", "0", "no", "n", "f"}:
        return False

    raise argparse.ArgumentTypeError("Expected TRUE or FALSE.")


def resolve_config_path(config_path):
    config_path = Path(config_path).expanduser()

    if config_path.is_absolute():
        resolved = config_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file is missing: {resolved}")
        return resolved

    cwd_candidate = (Path.cwd() / config_path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    script_candidate = (PROJECT_ROOT / config_path).resolve()
    if script_candidate.is_file():
        return script_candidate

    raise FileNotFoundError(
        "Config file is missing. Tried:\n"
        f"  {cwd_candidate}\n"
        f"  {script_candidate}"
    )


def load_yaml_config(config_path):
    config_path = resolve_config_path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        config = {}
    return config, config_path


def resolve_path_from_base(path_value, base_dir):
    if path_value is None:
        return None
    path_value = Path(str(path_value)).expanduser()
    if path_value.is_absolute():
        return path_value.resolve()
    return (base_dir / path_value).resolve()


def choose_value(cli_value, pred_cfg, pred_key, config=None, config_key=None, default=None):
    if cli_value is not None:
        return cli_value, "cli"
    if pred_key in pred_cfg and pred_cfg[pred_key] is not None:
        return pred_cfg[pred_key], "config"
    if config is not None and config_key is not None:
        if config_key in config and config[config_key] is not None:
            return config[config_key], "config"
    return default, "default"


def resolve_chosen_path(value, source, config_dir):
    if value is None:
        return None
    base_dir = config_dir if source == "config" else Path.cwd()
    return resolve_path_from_base(value, base_dir)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_prefix_to_filename(filename, prefix):
    if prefix is None or prefix == "":
        return filename
    return f"{prefix}_{filename}"


def find_feature_file(feature_dir, filename, prefix=""):
    feature_dir = Path(feature_dir)
    candidates = [feature_dir / add_prefix_to_filename(filename, prefix)]

    if prefix:
        candidates.append(feature_dir / filename)
    for path in candidates:
        if path.is_file():
            return path
    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Feature file is missing. Tried:\n{candidate_text}")


def load_selected_features(selected_features_path=None, model_package=None):
    if isinstance(model_package, dict):
        for key in (
            "feature_list",
            "Feature_list",
            "selected_features",
            "feature_names",
            "required_order",
            "features"):
            value = model_package.get(key)
            if value:
                return list(value)
    if selected_features_path is None:
        raise FileNotFoundError(
            "Selected feature list was not found in the model package. "
            "For your stacking_merge_model.py output, the joblib package should contain `feature_list`. "
            "If you use another model file, provide --selected-features."
        )
    path = Path(selected_features_path)
    if not path.is_file():
        raise FileNotFoundError(f"Selected feature file is missing: {path}")
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        for key in ("selected_features", "feature_names", "required_order", "features"):
            if key in obj and obj[key]:
                return list(obj[key])
        raise ValueError(f"No selected feature list found in JSON file: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for col in ("feature", "Feature", "selected_features", "feature_name"):
            if col in df.columns:
                return df[col].dropna().astype(str).tolist()
        if df.shape[1] == 1:
            return df.iloc[:, 0].dropna().astype(str).tolist()
        raise ValueError(
            f"Cannot infer selected feature column from CSV: {path}. "
            "Expected a column named feature/Feature/selected_features/feature_name.")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Selected feature file is empty: {path}")
    if "\n" not in text and "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_feature_matrix(feature_dir, prefix=""):
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        raise NotADirectoryError(f"Feature directory does not exist: {feature_dir}")
    accession = None
    true_label = None
    feature_blocks = []
    for block_name, filename in FEATURE_FILES.items():
        path = find_feature_file(feature_dir, filename, prefix=prefix)
        df = pd.read_csv(path)
        if "AccessionName" not in df.columns:
            raise ValueError(f"`AccessionName` column is missing in {path}")
        current_accession = df["AccessionName"].astype(str).tolist()
        if accession is None:
            accession = current_accession
        elif current_accession != accession:
            raise ValueError(
                f"AccessionName order is inconsistent in {path}. "
                "Please make sure all feature files are generated from the same FASTA in the same order.")
        if "label" in df.columns:
            current_label = df["label"].tolist()
            if true_label is None:
                true_label = current_label
            elif current_label != true_label:
                raise ValueError(f"`label` column is inconsistent in {path}")
        feature_cols = [col for col in df.columns if col not in SKIP_COLUMNS]
        if not feature_cols:
            raise ValueError(f"No feature columns found in {path}")
        feature_blocks.append(df[feature_cols])
    matrix = pd.concat(feature_blocks, axis=1)
    duplicated = matrix.columns[matrix.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicated feature names found after merging feature files. "
            f"Examples: {duplicated[:20]}")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        bad_cols = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(
            "NaN or non-numeric values were found in merged feature matrix. "
            f"Columns with problems include: {bad_cols[:30]}")
    return matrix, pd.Series(accession, name="AccessionName"), true_label


def get_model_and_threshold(model_path, threshold_arg=None):
    """Load either a single estimator or a packaged custom Stacking model."""
    package = joblib.load(model_path)

    if isinstance(package, dict):
        threshold = (
            float(threshold_arg)
            if threshold_arg is not None
            else float(package.get("threshold", 0.5))
        )

        # Conventional package containing one directly usable estimator.
        direct_model = package.get("model", package.get("estimator"))
        if direct_model is not None:
            return direct_model, threshold, package

        # Custom Stacking package saved by the training script.
        if package.get("meta_model") is not None and package.get("base_models") is not None:
            return package, threshold, package

        available_keys = ", ".join(sorted(map(str, package.keys())))
        raise ValueError(
            "Unsupported model-package structure. Expected `model`/`estimator`, "
            "or both `base_models` and `meta_model`. "
            f"Available keys: {available_keys}"
        )

    threshold = 0.5 if threshold_arg is None else float(threshold_arg)
    return package, threshold, None


def _positive_class_probability(model, matrix, model_name="model"):
    """Return positive-class probabilities from one fitted classifier."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model_name} does not have predict_proba(). "
            "All base models and the meta-model must be probabilistic classifiers."
        )

    proba = np.asarray(model.predict_proba(matrix))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(
            f"{model_name}.predict_proba() did not return a two-class probability matrix."
        )
    return proba[:, 1]


def predict_probability(model, matrix):
    """Predict with either one estimator or a custom two-level Stacking package."""
    if isinstance(model, dict):
        base_models = model.get("base_models")
        meta_model = model.get("meta_model")

        if base_models is None or meta_model is None:
            raise ValueError(
                "The Stacking package must contain both `base_models` and `meta_model`."
            )

        if isinstance(base_models, dict):
            model_items = list(base_models.items())
        elif isinstance(base_models, (list, tuple)):
            model_items = [(f"base_model_{i + 1}", item) for i, item in enumerate(base_models)]
        else:
            raise TypeError("`base_models` must be a dict, list, or tuple.")

        if not model_items:
            raise ValueError("`base_models` is empty.")

        meta_features = np.column_stack([
            _positive_class_probability(base_model, matrix, model_name=str(name))
            for name, base_model in model_items
        ])

        expected_meta_features = getattr(meta_model, "n_features_in_", None)
        if expected_meta_features is not None and meta_features.shape[1] != expected_meta_features:
            raise ValueError(
                "The number of generated base-model probabilities does not match the "
                f"meta-model input: generated {meta_features.shape[1]}, "
                f"expected {expected_meta_features}."
            )

        return _positive_class_probability(meta_model, meta_features, model_name="meta_model")

    return _positive_class_probability(model, matrix)


def validate_selected_features(matrix, selected_features):
    missing_features = [feature for feature in selected_features if feature not in matrix.columns]
    if missing_features:
        raise ValueError(f"Selected features are missing from calculated feature values: {missing_features[:30]}")
    return matrix[selected_features]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml used by feature calculation scripts.",)
    parser.add_argument("--input-fasta", default=None, help="FASTA file to predict. If provided, features will be calculated first.",)
    parser.add_argument("--raw-prediction-dir", default=None, help="Raw external prediction root containing HHM/ IDR/ Linker/ SS/.",)
    parser.add_argument("--dataset-name", default=None, help="Dataset name used for temporary standardization.",)
    parser.add_argument( "--standardized-prediction-dir", default=None, help="Optional standardized prediction root. Usually not needed.",)
    parser.add_argument("--save-calculated-features", type=str2bool, default=None, help="Whether to save calculated feature CSV files. Use TRUE or FALSE.",)
    parser.add_argument("--feature-output-dir", default=None, help="Directory to save calculated feature CSV files when save_calculated_features is TRUE.",)
    parser.add_argument("--skip-standardize", action="store_true", default=None, help="Use an existing standardized prediction directory instead of raw prediction files.",)
    parser.add_argument("--allow-missing", action="store_true", default=None, help="Allow incomplete external prediction coverage when feature scripts support it.",)
    parser.add_argument("--feature-dir", default=None, help="Directory containing the eight calculated feature CSV files. Used only when input_fasta is not provided.",)
    parser.add_argument("--model", default=None, help="Saved Stacking model joblib file.",)
    parser.add_argument("--selected-features", default=None, help="Optional selected feature list. Usually not needed because stacking_merge_model.py saves `feature_list` inside the joblib package.",)
    parser.add_argument("--output", default=None, help="Output CSV file for prediction probabilities.",)
    parser.add_argument("--prefix", default=None, help='Feature file prefix. Use "" for AA_frequency_features.csv; use "predict" for predict_AA_frequency_features.csv.',)
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold. Default: threshold saved in model package, otherwise 0.5.",)
    args = parser.parse_args()

    config, config_path = load_yaml_config(args.config)
    config_dir = config_path.parent
    pred_cfg = config.get("prediction", {})

    input_fasta_value, input_fasta_source = choose_value(cli_value=args.input_fasta, pred_cfg=pred_cfg,
        pred_key="input_fasta", config=config, config_key=None, default=None)

    raw_prediction_dir_value, raw_prediction_dir_source = choose_value(
        cli_value=args.raw_prediction_dir,
        pred_cfg=pred_cfg,
        pred_key="raw_prediction_dir",
        config=config,
        config_key="raw_predicted_features_dir",
        default=None)

    dataset_name_value, _ = choose_value(
        cli_value=args.dataset_name,
        pred_cfg=pred_cfg,
        pred_key="dataset_name",
        config=config,
        config_key=None,
        default="prediction")

    standardized_prediction_dir_value, standardized_prediction_dir_source = choose_value(
        cli_value=args.standardized_prediction_dir,
        pred_cfg=pred_cfg,
        pred_key="standardized_prediction_dir",
        config=config,
        config_key="standardized_predicted_features_dir",
        default=None)

    save_calculated_features_value, _ = choose_value(
        cli_value=args.save_calculated_features,
        pred_cfg=pred_cfg,
        pred_key="save_calculated_features",
        config=config,
        config_key=None,
        default=False)

    feature_output_dir_value, feature_output_dir_source = choose_value(
        cli_value=args.feature_output_dir,
        pred_cfg=pred_cfg,
        pred_key="feature_output_dir",
        config=config,
        config_key="feature_output_dir",
        default="outputs/features")

    skip_standardize_value, _ = choose_value(
        cli_value=args.skip_standardize,
        pred_cfg=pred_cfg,
        pred_key="skip_standardize",
        config=config,
        config_key=None,
        default=False)

    allow_missing_value, _ = choose_value(
        cli_value=args.allow_missing,
        pred_cfg=pred_cfg,
        pred_key="allow_missing",
        config=config,
        config_key=None,
        default=False)

    feature_dir_value, feature_dir_source = choose_value(
        cli_value=args.feature_dir,
        pred_cfg=pred_cfg,
        pred_key="feature_dir",
        config=config,
        config_key="feature_output_dir",
        default="outputs/features")

    model_value, model_source = choose_value(
        cli_value=args.model,
        pred_cfg=pred_cfg,
        pred_key="model",
        config=config,
        config_key=None,
        default="outputs/feature_selection/stacking_model/stacking_model.joblib")

    selected_features_value, selected_features_source = choose_value(
        cli_value=args.selected_features,
        pred_cfg=pred_cfg,
        pred_key="selected_features",
        config=config,
        config_key=None,
        default=None)

    output_value, output_source = choose_value(
        cli_value=args.output,
        pred_cfg=pred_cfg,
        pred_key="output",
        config=config,
        config_key=None,
        default="outputs/predictions/prediction_results.csv")

    prefix_value, _ = choose_value(
        cli_value=args.prefix,
        pred_cfg=pred_cfg,
        pred_key="prefix",
        config=config,
        config_key=None,
        default="")

    # Normalize boolean values from YAML or command line.
    save_calculated_features = str2bool(save_calculated_features_value)
    skip_standardize = str2bool(skip_standardize_value)
    allow_missing = str2bool(allow_missing_value)

    # Resolve paths.
    model_path = resolve_chosen_path(model_value, model_source, config_dir)
    output_path = resolve_chosen_path(output_value, output_source, config_dir)
    selected_features_path = (resolve_chosen_path(selected_features_value, selected_features_source, config_dir) if selected_features_value else None)
    input_fasta_path = (resolve_chosen_path(input_fasta_value, input_fasta_source, config_dir) if input_fasta_value else None)
    raw_prediction_dir_path = (resolve_chosen_path(raw_prediction_dir_value, raw_prediction_dir_source, config_dir) if raw_prediction_dir_value else None)
    standardized_prediction_dir_path = (resolve_chosen_path(standardized_prediction_dir_value, standardized_prediction_dir_source, config_dir) if standardized_prediction_dir_value else None)
    feature_output_dir_path = (resolve_chosen_path(feature_output_dir_value, feature_output_dir_source, config_dir) if feature_output_dir_value else None)
    feature_dir_path = (resolve_chosen_path(feature_dir_value, feature_dir_source, config_dir) if feature_dir_value else None)

    print("[1/5] Loading model and selected features...")
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model file is missing: {model_path}")
    model, threshold, package = get_model_and_threshold(model_path, threshold_arg=args.threshold,)
    selected_features = load_selected_features(selected_features_path, model_package=package,)
    print(f"Using {len(selected_features)} selected features.")
    print(f"Classification threshold: {threshold}")

    if input_fasta_path is not None:
        print("[2/5] Calculating feature values from FASTA and external predictions...")

        if raw_prediction_dir_path is None and not skip_standardize:
            raise ValueError(
                "raw_prediction_dir is required when input_fasta is provided unless skip_standardize is true. "
                "Set prediction.raw_prediction_dir in config.yaml or pass --raw-prediction-dir.")

        matrix, accession, true_label = calculate_features_for_prediction(
            input_fasta=input_fasta_path,
            raw_prediction_dir=raw_prediction_dir_path,
            selected_features=selected_features,
            config_path=config_path,
            dataset_name=dataset_name_value,
            standardized_prediction_dir=standardized_prediction_dir_path,
            feature_output_dir=feature_output_dir_path,
            prefix=prefix_value,
            save_calculated_features=save_calculated_features,
            skip_standardize=skip_standardize,
            allow_missing=allow_missing)

    else:
        print("[2/5] Loading already calculated feature CSV files...")
        matrix, accession, true_label = load_feature_matrix(feature_dir_path, prefix=prefix_value)
        
        print("[3/5] Selecting final model features...")
        matrix = validate_selected_features(matrix, selected_features)
    if input_fasta_path is not None:
        print("[3/5] Feature selection was applied during feature calculation.")

    print("[4/5] Predicting probabilities...")
    probability = predict_probability(model, matrix)
    predicted_label = (probability >= threshold).astype(int)
    result = pd.DataFrame({"AccessionName": accession, "prediction_probability": probability, "predicted_label": predicted_label})
    if true_label is not None and len(true_label) == len(result):
        result.insert(1, "true_label", true_label)

    print("[5/5] Saving prediction results...")
    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False)
    print(f"Saved prediction results to: {output_path}")
    if input_fasta_path is not None and save_calculated_features:
        print(f"Saved calculated feature files to: {feature_output_dir_path}")
    elif input_fasta_path is not None:
        print("Intermediate calculated feature CSV files were not saved.")

    print("Completed!!")


if __name__ == "__main__":
    main()