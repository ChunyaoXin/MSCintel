# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from feature_file_utils import discover_csv_files


REPO_ROOT = Path(__file__).resolve().parents[1]


def calculate_pbc_from_csv(input_csv, p_value_threshold, output_csv=None, output_sig_csv=None, save_outputs=False):
    """Calculate point-biserial correlation for all valid features."""
    df = pd.read_csv(input_csv)

    if df.columns[0] != "AccessionName":
        raise ValueError(f"The first column of {input_csv} must be 'AccessionName'.")

    if df.columns[-1] != "label":
        raise ValueError(f"The last column of {input_csv} must be 'label'.")

    label_col = df.columns[-1]

    if not set(df[label_col].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"The last column `{label_col}` must be a binary 0/1 label.")

    pbc_results = []

    skip_cols = {"AccessionName", label_col}

    for col in df.columns:
        if col in skip_cols:
            continue

        feature_col = pd.to_numeric(df[col], errors="coerce")
        cleaned_col = feature_col.replace([np.inf, -np.inf], np.nan).dropna()

        if cleaned_col.empty:
            continue

        if cleaned_col.nunique() <= 1:
            continue

        aligned_label = df.loc[cleaned_col.index, label_col]

        if aligned_label.nunique() <= 1:
            continue

        try:
            correlation, p_value = pointbiserialr(cleaned_col, aligned_label)

            pbc_results.append({
                "Variable": col,
                "Point-Biserial Correlation": correlation,
                "P-Value": p_value,
                "Abs Correlation": abs(correlation),
            })

        except Exception as e:
            print(f"Feature {col} failed during PBC calculation: {str(e)}")

    result_df = pd.DataFrame(pbc_results)

    if result_df.empty:
        print(f"No valid features were available for PBC calculation in {input_csv}.")
        return result_df, result_df.copy()

    result_df = result_df.sort_values(["P-Value", "Abs Correlation"], ascending=[True, False])

    significant_df = result_df[result_df["P-Value"] < p_value_threshold].copy()

    if save_outputs:
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"All PBC results saved to: {output_csv.resolve()}")

        if output_sig_csv is not None:
            output_sig_csv = Path(output_sig_csv)
            output_sig_csv.parent.mkdir(parents=True, exist_ok=True)

            if significant_df.shape[0] > 0:
                significant_df.to_csv(output_sig_csv, index=False, encoding="utf-8-sig")
                print(f"Significant correlated features saved to: {output_sig_csv.resolve()}")
            else:
                print(f"No significant features were found with p < {p_value_threshold}.")

    return result_df, significant_df


def process_multiple_datasets(datasets, base_path, output_dir, correlative_features_dir, all_correlative_features_path, p_value_threshold, save_outputs=False):
    """Process multiple feature files for point-biserial correlation analysis."""
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    correlative_features_dir = Path(correlative_features_dir)
    all_correlative_features_path = Path(all_correlative_features_path)

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        correlative_features_dir.mkdir(parents=True, exist_ok=True)
        all_correlative_features_path.parent.mkdir(parents=True, exist_ok=True)

    all_pbc_results = {}
    correlative_results = {}
    all_significant_results = []

    for dataset in datasets:
        input_csv = base_path / f"{dataset['param']}.csv"
        output_csv = output_dir / f"correlative_{dataset['name']}.csv"
        output_sig_csv = correlative_features_dir / f"correlative_{dataset['name']}.csv"

        print(f"Processing dataset: {dataset['name']}")

        try:
            result_df, significant_df = calculate_pbc_from_csv(
                input_csv=input_csv,
                p_value_threshold=p_value_threshold,
                output_csv=output_csv,
                output_sig_csv=output_sig_csv,
                save_outputs=save_outputs,
            )

            all_pbc_results[dataset["name"]] = result_df
            correlative_results[dataset["name"]] = significant_df

            if not significant_df.empty:
                significant_df = significant_df.copy()
                significant_df["Source"] = dataset["name"]
                all_significant_results.append(significant_df)

        except Exception as e:
            print(f"Failed to process {dataset['name']}: {str(e)}\n")

    if all_significant_results:
        all_correlative_features_df = pd.concat(all_significant_results, ignore_index=True,)
        all_correlative_features_df = all_correlative_features_df.sort_values("P-Value")

        if save_outputs:
            all_correlative_features_df.to_csv(all_correlative_features_path, index=False, encoding="utf-8-sig")
            print(f"All significant correlated features saved to: {all_correlative_features_path.resolve()}")

    else:
        all_correlative_features_df = pd.DataFrame()
        print("No significant correlated features were identified.")

    return all_pbc_results, correlative_results, all_correlative_features_df

def discover_feature_datasets(feature_dir):
    feature_dir = Path(feature_dir)

    csv_files = discover_csv_files(feature_dir)

    datasets = []

    for csv_file in csv_files:
        datasets.append({
            "name": csv_file.stem,
            "param": csv_file.stem,
        })

    return datasets

def run_correlative_selection(train_features_file_dir=None, correlative_results_dir=None, p_value_threshold=0.05, save_outputs=False):
    """Run correlation-based feature filtering. This function is designed for pipeline usage."""

    if train_features_file_dir is None:
        train_features_file_dir = REPO_ROOT / "outputs" / "features" / "train"
    else:
        train_features_file_dir = Path(train_features_file_dir)

    datasets = discover_feature_datasets(train_features_file_dir)

    if correlative_results_dir is None:
        correlative_results_dir = REPO_ROOT / "outputs" / "feature_selection" / "correlative_results"
    else:
        correlative_results_dir = Path(correlative_results_dir)

    all_features_pbc_dir = correlative_results_dir / "all_features_pbc"
    correlative_features_dir = correlative_results_dir / "correlative_features"
    all_correlative_features_path = correlative_results_dir / "all_correlative_features.csv"

    return process_multiple_datasets(
        datasets=datasets,
        base_path=train_features_file_dir,
        output_dir=all_features_pbc_dir,
        correlative_features_dir=correlative_features_dir,
        all_correlative_features_path=all_correlative_features_path,
        p_value_threshold=p_value_threshold,
        save_outputs=save_outputs,
    )


if __name__ == "__main__":
    run_correlative_selection(
        train_features_file_dir=REPO_ROOT / "outputs" / "features" / "train",
        correlative_results_dir=REPO_ROOT / "outputs" / "feature_selection" / "correlative_results",
        p_value_threshold=0.05,
        save_outputs=True)