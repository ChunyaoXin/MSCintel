# -*- coding: utf-8 -*-

import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dataframe(data):
    """
    Load a DataFrame from either an existing DataFrame object or a CSV file path.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    data_path = Path(data)

    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    return pd.read_csv(data_path)


def get_cor_sig_features(all_correlative_features_pbc, all_significant_features, top_ratio=0.3):

    df1 = load_dataframe(all_correlative_features_pbc)
    df2 = load_dataframe(all_significant_features)

    if df1.empty or df2.empty:
        print("Input feature file is empty.")
        return pd.DataFrame(), pd.DataFrame()

    if "Abs Correlation" not in df1.columns:
        raise ValueError("Column 'Abs Correlation' is required in the correlative feature file.")

    # Use the first column as feature names to keep the function format-flexible.
    features1 = df1.iloc[:, 0].dropna().astype(str).tolist()
    features2 = df2.iloc[:, 0].dropna().astype(str).tolist()

    common_features = sorted(set(features1) & set(features2))

    if not common_features:
        print("No overlapping features were found between correlative and significant results.")
        return pd.DataFrame(columns=df1.columns), pd.DataFrame(columns=df1.columns)

    filtered_df = df1[df1.iloc[:, 0].astype(str).isin(common_features)].copy()
    sorted_df = filtered_df.sort_values(by="Abs Correlation", ascending=False)

    top_n = max(1, math.ceil(len(sorted_df) * top_ratio))
    top_features_df = sorted_df.head(top_n)

    return sorted_df, top_features_df


def save_cor_sig_features(wrapper_features_dir, sorted_df, top_features_df):
    """
    Save intersected correlation-significance feature results.
    """
    wrapper_features_dir = Path(wrapper_features_dir)
    wrapper_features_dir.mkdir(parents=True, exist_ok=True)
    correlative_and_significant_features_all = wrapper_features_dir / "correlative_significant_features_all.csv"
    correlative_and_significant_features_30 = wrapper_features_dir / "correlative_significant_features_30.csv"
    sorted_df.to_csv(correlative_and_significant_features_all, index=False, encoding="utf-8-sig")
    top_features_df.to_csv(correlative_and_significant_features_30, index=False, encoding="utf-8-sig")

    print(f"All intersected features saved to: {correlative_and_significant_features_all.resolve()}")
    print(f"Top features saved to: {correlative_and_significant_features_30.resolve()}")


def run_correlative_significant_selection(
    all_correlative_features_pbc=None,
    all_significant_features=None,
    wrapper_features_dir=None,
    feature_selection_dir=None,
    top_ratio=0.3,
    save_outputs=False,
):
    """Run intersection between correlation-selected and significance-selected features."""
    
    if feature_selection_dir is None:
        feature_selection_dir = REPO_ROOT / "outputs" / "feature_selection"
    else:
        feature_selection_dir = Path(feature_selection_dir)

    if all_correlative_features_pbc is None:
        all_correlative_features_pbc = feature_selection_dir / "correlative_results" / "all_correlative_features.csv"

    if all_significant_features is None:
        all_significant_features = feature_selection_dir / "significant_results" / "all_significant_features.csv"

    if wrapper_features_dir is None:
        wrapper_features_dir = feature_selection_dir / "wrapper_features"
    else:
        wrapper_features_dir = Path(wrapper_features_dir)

    sorted_df, top_features_df = get_cor_sig_features(all_correlative_features_pbc=all_correlative_features_pbc, all_significant_features=all_significant_features, top_ratio=top_ratio)

    if save_outputs:
        save_cor_sig_features(wrapper_features_dir=wrapper_features_dir, sorted_df=sorted_df, top_features_df=top_features_df,)

    return sorted_df, top_features_df


if __name__ == "__main__":
    run_correlative_significant_selection(
        feature_selection_dir=REPO_ROOT / "outputs" / "feature_selection",
        wrapper_features_dir=(REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features"),
        top_ratio=0.3,
        save_outputs=True)

    print("Complete!!!!")