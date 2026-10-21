# -*- coding: utf-8 -*-

import time
from pathlib import Path
import pandas as pd
import numpy as np
from function import perform_iterations_and_analyze
from feature_file_utils import discover_csv_files

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_datasets(train_features_file_dir):
    train_features_file_dir = Path(train_features_file_dir)

    csv_files = discover_csv_files(train_features_file_dir)

    datasets = []

    for csv_file in csv_files:
        stem = csv_file.stem

        datasets.append({
            "name": stem,
            "correlative_file": f"correlative_{stem}.csv",
            "frequency_file": csv_file.name,
            "var_prefix": f"significant_{stem}",
        })

    return datasets


def extract_correlative_features(config, correlative_results=None, correlative_base_dir=None):
    """Extract correlated feature names from either in-memory results or saved correlative feature CSV files."""
    dataset_name = config["name"]

    if correlative_results is not None and dataset_name in correlative_results:
        data = correlative_results[dataset_name]

        if isinstance(data, pd.DataFrame):
            if data.empty:
                return []

            if "Variable" in data.columns:
                return data["Variable"].dropna().astype(str).tolist()

            return data.iloc[:, 0].dropna().astype(str).tolist()

        if isinstance(data, (list, tuple, set)):
            return list(data)

        raise TypeError(f"Unsupported correlative result type for {dataset_name}: {type(data)}")

    if correlative_base_dir is None:
        raise ValueError("Either correlative_results or correlative_base_dir must be provided.")

    correlative_path = Path(correlative_base_dir) / config["correlative_file"]

    if not correlative_path.exists():
        print(f"No correlative feature file found for {dataset_name}, skip this dataset.")
        return []

    return pd.read_csv(correlative_path).iloc[:, 0].dropna().astype(str).tolist()


def get_significant_features(train_features_file_dir, datasets, outer_iterations, inter_iterations, repeat_times, alpha, file_path, pcc, correlative_results=None, correlative_base_dir=None):
    """
    Run significance-based feature selection for all feature categories.
    """
    train_features_file_dir = Path(train_features_file_dir)
    file_path = Path(file_path)

    results = {}
    all_sig_features_list = []

    for config in datasets:
        frequency_path = train_features_file_dir / config["frequency_file"]

        if not frequency_path.exists():
            raise FileNotFoundError(f"Training feature file not found: {frequency_path}")

        correlative_features = extract_correlative_features(config=config, correlative_results=correlative_results, correlative_base_dir=correlative_base_dir)
        if len(correlative_features) == 0:
            print(f"No correlative features for {config['name']}, skip significance selection.")
            results[config["var_prefix"]] = []
            continue
        frequency_df = pd.read_csv(frequency_path)
        
        if frequency_df.columns[0] != "AccessionName":
            raise ValueError(f"The first column of {frequency_path} must be 'AccessionName'.")
        
        if frequency_df.columns[-1] != "label":
            raise ValueError(f"The last column of {frequency_path} must be 'label'.")
        
        label_col = "label"
        
        label = pd.to_numeric(frequency_df[label_col], errors="coerce")
        
        if not set(label.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"The last column `{label_col}` must be a binary 0/1 label.")
        
        positive_df = frequency_df.loc[label == 1].iloc[:, 1:-1]
        negative_df = frequency_df.loc[label == 0].iloc[:, 1:-1]
        
        start_time = time.time()
        significant_features = perform_iterations_and_analyze(positive_df, negative_df, correlative_features, outer_iterations, inter_iterations, repeat_times, alpha, file_path, pcc)
        end_time = time.time()

        print(f"{config['name']} runtime: {end_time - start_time} seconds")

        results[config["var_prefix"]] = significant_features

        for feat in significant_features:
            all_sig_features_list.append({"Feature": feat, "Source": config["name"]})

        print(f"{config['name']} significant features: {significant_features}")
        print(f"{config['name']} number of significant features: {len(significant_features)}")

    all_sig_features_df = pd.DataFrame(all_sig_features_list, columns=["Feature", "Source"])

    return results, all_sig_features_df


def save_significant_features(significant_results_dir, results, all_sig_features_df,):
    """Save significance-based feature selection results."""
    significant_results_dir = Path(significant_results_dir)
    significant_base_dir = significant_results_dir / "significant_features"
    all_sig_features_path = significant_results_dir / "all_significant_features.csv"

    significant_base_dir.mkdir(parents=True, exist_ok=True)

    for feature_type, features in results.items():
        output_file = significant_base_dir / f"{feature_type}.csv"
        pd.DataFrame(features).to_csv(output_file, index=False, header=False, encoding="utf-8-sig")
        print(f"Saved: {output_file}")

    if not all_sig_features_df.empty:
        all_sig_features_df.to_csv(all_sig_features_path, index=False, encoding="utf-8-sig")
        print(f"All significant features saved to: {all_sig_features_path.resolve()}")
    else:
        print("No significant features were identified.")


def run_significant_selection(
    train_features_file_dir=None,
    correlative_results=None,
    correlative_base_dir=None,
    significant_results_dir=None,
    outer_iterations=10,
    inter_iterations=100,
    repeat_times=0.7,
    alpha=0.05,
    pcc=0.7,
    file_path=None,
    save_outputs=False,
    random_state=42,
):
    """Run significance-based feature selection."""
    if train_features_file_dir is None:
        train_features_file_dir = REPO_ROOT / "outputs" / "features" / "train"
    else:
        train_features_file_dir = Path(train_features_file_dir)

    if correlative_base_dir is None:
        correlative_base_dir = REPO_ROOT / "outputs" / "feature_selection" / "correlative_results" / "correlative_features"
        
    else:
        correlative_base_dir = Path(correlative_base_dir)

    if significant_results_dir is None:
        significant_results_dir = REPO_ROOT / "outputs" / "feature_selection" / "significant_results"

    else:
        significant_results_dir = Path(significant_results_dir)

    if file_path is None:
        file_path = REPO_ROOT / "feature_selection" / "feature_value.txt"
    else:
        file_path = Path(file_path)

    datasets = build_datasets(train_features_file_dir)
    
    if random_state is not None:
        np.random.seed(random_state)

    results, all_sig_features_df = get_significant_features(train_features_file_dir=train_features_file_dir, datasets=datasets, outer_iterations=outer_iterations, inter_iterations=inter_iterations,
        repeat_times=repeat_times, alpha=alpha, file_path=file_path, pcc=pcc, correlative_results=correlative_results, correlative_base_dir=correlative_base_dir)

    if save_outputs:
        save_significant_features(significant_results_dir=significant_results_dir, results=results, all_sig_features_df=all_sig_features_df,)

    return results, all_sig_features_df


if __name__ == "__main__":
    run_significant_selection(
        train_features_file_dir=REPO_ROOT / "outputs" / "features" / "train",
        correlative_base_dir=REPO_ROOT / "outputs" / "feature_selection" / "correlative_results" / "correlative_features",
        significant_results_dir=REPO_ROOT / "outputs" / "feature_selection" / "significant_results",
        outer_iterations=10,
        inter_iterations=100,
        repeat_times=0.7,
        alpha=0.05,
        pcc=0.7,
        file_path=REPO_ROOT / "feature_selection" / "feature_value.txt",
        save_outputs=True,
        random_state=42
    )