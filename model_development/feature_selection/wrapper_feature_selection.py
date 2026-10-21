# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from feature_file_utils import discover_csv_files

from function import perform_feature_selection_and_auc_calculation, normalize, get_wrapper_models

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_feature_source_to_list(feature_source):
    if isinstance(feature_source, pd.DataFrame):
        values = feature_source.iloc[:, 0].dropna().astype(str).tolist()
    elif isinstance(feature_source, (list, tuple, set)):
        values = list(feature_source)
    else:
        data = pd.read_csv(feature_source)
        values = data.iloc[:, 0].dropna().astype(str).tolist()

    values = list(dict.fromkeys(values))
    return values

def load_features_dict(file_dict):
    all_feature_frames = []
    reference_accession = None
    reference_label = None

    for feature_name, path in file_dict.items():
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        df = pd.read_csv(path)

        if df.columns[0] != "AccessionName":
            raise ValueError(f"The first column of {path} must be 'AccessionName'.")

        if df.columns[-1] != "label":
            raise ValueError(f"The last column of {path} must be 'label'.")

        accession = df.iloc[:, 0].astype(str).reset_index(drop=True)
        label = pd.to_numeric(df.iloc[:, -1], errors="raise").astype(int).reset_index(drop=True)

        if not set(label.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"The label column in {path} must be binary 0/1.")

        if reference_accession is None:
            reference_accession = accession
            reference_label = label
        else:
            if not accession.equals(reference_accession):
                raise ValueError(f"AccessionName order mismatch in {path}. Please make sure all feature files were generated from the same training FASTA order.")

            if not label.equals(reference_label):
                raise ValueError(f"Label mismatch in {path}. Please check whether all feature files were generated from the same labelled training FASTA.")

        feature_df = df.iloc[:, 1:-1]
        all_feature_frames.append(feature_df)

    combined_df = pd.concat(all_feature_frames, axis=1)

    duplicated_columns = combined_df.columns[combined_df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"Duplicated feature columns found after merging feature files: {duplicated_columns[:20]}")

    return combined_df, reference_label


def print_model_results(
    model_name,
    selected_features_list,
    selection_auc_list,
    refit_auc_list,
    refit_aps_list
):
    print(f"\nSelected features for {model_name} model:")

    for name, features in selected_features_list:
        print(f"{name}: {features}")
        print(f"Number of features: {len(features)}")

    print(f"\nSFFS selection-process Average AUC for {model_name} model:")
    for name, auc in selection_auc_list:
        print(f"{name}: {auc:.4f}")

    print(f"\nRe-evaluated 5-fold Average AUC for {model_name} model:")
    for name, auc in refit_auc_list:
        print(f"{name}: {auc:.4f}")

    print(f"\nRe-evaluated 5-fold Average AUPRC for {model_name} model:")
    for name, aps in refit_aps_list:
        print(f"{name}: {aps:.4f}")


def save_final_features(wrapper_features_df, output_file_path):

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper_features_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")

    print(f"Wrapper feature-selection results saved to: {output_file_path}")

def build_wrapper_features_df(selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list):
    if not (
        len(selected_features_list)
        == len(selection_auc_list)
        == len(refit_auc_list)
        == len(refit_aps_list)
    ):
        raise ValueError(
            f"List length mismatch: "
            f"features={len(selected_features_list)}, "
            f"selection_auc={len(selection_auc_list)}, "
            f"refit_auc={len(refit_auc_list)}, "
            f"refit_auprc={len(refit_aps_list)}"
        )

    rows = []

    for (
        (model_name, features),
        (_, selection_auc),
        (_, refit_auc),
        (_, refit_aps)
    ) in zip(
        selected_features_list,
        selection_auc_list,
        refit_auc_list,
        refit_aps_list
    ):
        rows.append({
            "Model": model_name,
            "Features List": features,
            "Number of Features": len(features),
            "Selection_CV_AUC": selection_auc,
            "Refit_CV_AUC": refit_auc,
            "Refit_CV_AUPRC": refit_aps
        })

    return pd.DataFrame(rows)
    
def discover_feature_files(feature_dir):
    feature_dir = Path(feature_dir)

    csv_files = discover_csv_files(feature_dir)

    return {csv_file.stem: csv_file for csv_file in csv_files}

def run_wrapper_feature_selection(
    train_features_file_dir=None,
    correlative_significant_features=None,
    correlative_significant_features_path=None,
    wrapper_features_dir=None,
    output_file_path=None,
    cv_splits=5,
    random_state=42,
    save_outputs=False
):
    train_features_file_dir = REPO_ROOT / "outputs" / "features" / "train" if train_features_file_dir is None else Path(train_features_file_dir)
    wrapper_features_dir = REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features" if wrapper_features_dir is None else Path(wrapper_features_dir)
    correlative_significant_features_path = wrapper_features_dir / "correlative_significant_features_30.csv" if correlative_significant_features_path is None else Path(correlative_significant_features_path)
    output_file_path = wrapper_features_dir / "wrapper_features.csv" if output_file_path is None else Path(output_file_path)

    features_file_dict = discover_feature_files(train_features_file_dir)

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    with tqdm(total=3, desc="Data prepare", file=sys.stdout, mininterval=1) as pbar:
        combined_df, train_y = load_features_dict(features_file_dict)
        min_class_count = train_y.value_counts().min()
        if cv_splits > min_class_count:
            raise ValueError(f"cv_splits={cv_splits} is larger than the smallest class size ({min_class_count}).")
        pbar.update(1)

        significant_features = read_feature_source_to_list(correlative_significant_features if correlative_significant_features is not None else correlative_significant_features_path)
        print("Number of significant features:", len(significant_features))
        if len(significant_features) == 0:
            raise ValueError(
                    "No features were selected in the previous stage. "
                    "Wrapper feature selection cannot proceed. "
                    "Please check the input dataset and feature selection thresholds.")
        pbar.update(1)

        missing_features = [feature for feature in significant_features if feature not in combined_df.columns]
        if missing_features:
            raise ValueError("Some selected features are not present in the merged training feature matrix: " f"{missing_features[:30]}")

        train_x = combined_df[significant_features]
        train_x, _, _ = normalize(train_x)
        pbar.update(1)

    models = get_wrapper_models()
    feature_selection_steps = len(models) * 2 + 2

    with tqdm(total=feature_selection_steps, desc="features select", file=sys.stdout, mininterval=1) as pbar:
        start_time = time.time()

        selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list = perform_feature_selection_and_auc_calculation(train_x, train_y, significant_features, cv=skf, models=models, progress_callback=lambda: pbar.update(1))
        
        print_model_results("ALL", selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list)
        pbar.update(1)
        
        wrapper_features_df = build_wrapper_features_df(selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list)

        if save_outputs:
            save_final_features(wrapper_features_df=wrapper_features_df, output_file_path=output_file_path)

        pbar.update(1)

        end_time = time.time()
        print(f"\nfeatures select running time: {end_time - start_time:.2f} seconds")

    return wrapper_features_df, selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list

if __name__ == "__main__":
    run_wrapper_feature_selection(
        train_features_file_dir=REPO_ROOT / "outputs" / "features" / "train",
        correlative_significant_features_path=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features" / "correlative_significant_features_30.csv",
        wrapper_features_dir=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features",
        output_file_path=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features" / "wrapper_features_new_pytorch_3.8.csv",
        cv_splits=5,
        random_state=42,
        save_outputs=True,
    )
