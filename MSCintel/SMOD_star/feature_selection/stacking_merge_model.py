# -*- coding: utf-8 -*-
"""
Feature-set comparison and Stacking model.

This script evaluates:

1. Individual base models using their model-specific wrapper-selected features.

2. Three feature-set integration strategies:
       SMOD_∪ : features selected by at least one base model
       SMOD_star  : features selected by at least two base models
       SMOD_∩ : features selected by all four base models

For each strategy:
       selected features -> LR / SVM / KNN / RF
       -> four OOF probabilities -> meta-LR
"""

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from feature_file_utils import discover_csv_files


REPO_ROOT = Path(__file__).resolve().parents[1]


# ================================================================
# Feature-loading utilities
# ================================================================

def load_final_features_dict(file_path):
    return pd.read_csv(file_path)


def parse_feature_list(features):
    if isinstance(features, str):
        return ast.literal_eval(features)
    if isinstance(features, (list, tuple, set)):
        return list(features)
    raise TypeError(f"Unsupported feature-list type: {type(features)}")


def load_features_and_labels(file_dict, dataset_name="dataset"):
    all_data = []
    reference_accession = None
    reference_label = None

    for feature_name, path in file_dict.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{dataset_name} feature file not found: {path}")

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
                raise ValueError(f"AccessionName order mismatch in {path}. Please make sure all {dataset_name} feature files were generated from the same FASTA order.")
            if not label.equals(reference_label):
                raise ValueError(f"Label mismatch in {path}. Please make sure all {dataset_name} feature files were generated from the same labelled FASTA.")

        all_data.append(df.iloc[:, 1:-1])

    combined_df = pd.concat(all_data, axis=1)

    duplicated_columns = combined_df.columns[combined_df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"Duplicated feature columns found in {dataset_name} feature matrix: {duplicated_columns[:20]}")

    return combined_df, reference_label.to_numpy(), reference_accession.to_numpy()


def discover_feature_files(train_dir, test_dir):
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train_csv_files = discover_csv_files(train_dir)
    test_csv_files = discover_csv_files(test_dir)

    train_files = {p.name: p for p in train_csv_files}
    test_files = {p.name: p for p in test_csv_files}

    missing_in_test = sorted(set(train_files) - set(test_files))
    missing_in_train = sorted(set(test_files) - set(train_files))

    if missing_in_test:
        raise ValueError(f"Feature files missing in test folder: {missing_in_test}")
    if missing_in_train:
        raise ValueError(f"Feature files missing in train folder: {missing_in_train}")

    train_feature_dict = {}
    test_feature_dict = {}

    for train_path in train_csv_files:
        feature_name = train_path.stem
        filename = train_path.name
        train_feature_dict[feature_name] = train_path
        test_feature_dict[feature_name] = test_files[filename]

    return train_feature_dict, test_feature_dict


# ================================================================
# Feature-set selection utilities
# ================================================================

def get_cor_sig_dict(cor_sig_features_source):
    if isinstance(cor_sig_features_source, pd.DataFrame):
        df = cor_sig_features_source.copy()
    else:
        df = pd.read_csv(cor_sig_features_source)

    if "Variable" not in df.columns:
        raise ValueError("Input correlative-significant features do not contain 'Variable' column.")
    if "Abs Correlation" not in df.columns:
        raise ValueError("Input correlative-significant features do not contain 'Abs Correlation' column.")

    df["Abs Correlation"] = pd.to_numeric(df["Abs Correlation"], errors="coerce")
    df = df.dropna(subset=["Variable", "Abs Correlation"])

    return dict(zip(df["Variable"].astype(str), df["Abs Correlation"]))


def get_model_specific_features(final_df, wrapper_model_name):
    model_row = final_df[final_df["Model"] == wrapper_model_name]
    if model_row.empty:
        raise ValueError(f"Model {wrapper_model_name} not found in wrapper feature file.")
    return parse_feature_list(model_row["Features List"].iloc[0])


def collect_feature_counts(final_df, selected_models):
    selected_feature_sets = []
    all_features = set()

    for model_name in selected_models:
        features = set(get_model_specific_features(final_df, model_name))
        selected_feature_sets.append(features)
        all_features.update(features)

    feature_counts = {feature: 0 for feature in all_features}
    for feature_set in selected_feature_sets:
        for feature in feature_set:
            feature_counts[feature] += 1

    return feature_counts


def select_features_by_min_count(final_df, selected_models, feature_relevance, min_count):
    """
    Select Stacking features by feature co-occurrence across wrapper models.

    min_count can be:
        len(selected_models) / 2  -> at least half of models; with 4 models, at least 2
        len(selected_models)      -> intersection
        1                         -> union
    """
    feature_counts = collect_feature_counts(final_df, selected_models)
    selected_features = [f for f, c in feature_counts.items() if c >= min_count]
    selected_features = sorted(selected_features, key=lambda x: (-feature_relevance.get(x, 0), x))
    return selected_features


def build_feature_matrix_from_feature_list(train_combined_df, test_combined_df, feature_list, dataset_description="selected features"):
    if not feature_list:
        raise ValueError(f"No features were selected for {dataset_description}.")

    missing_train_features = [f for f in feature_list if f not in train_combined_df.columns]
    missing_test_features = [f for f in feature_list if f not in test_combined_df.columns]

    if missing_train_features:
        raise ValueError(f"Missing features in training data for {dataset_description}: {missing_train_features[:30]}")
    if missing_test_features:
        raise ValueError(f"Missing features in test data for {dataset_description}: {missing_test_features[:30]}")

    train_x = train_combined_df[feature_list]
    test_x = test_combined_df[feature_list]
    return train_x, test_x


def get_feature_set_strategies(selected_models):
    n_models = len(selected_models)
    return [
        {"name": "SMOD_union", "min_count": 1, "description": "features selected by at least one wrapper model"},
        {"name": "SMOD_star", "min_count": n_models / 2, "description": "features selected by at least half of the wrapper models"},
        {"name": "SMOD_intersection", "min_count": n_models, "description": "features selected by all wrapper models"}
    ]


# ================================================================
# Model utilities
# ================================================================

def get_45dim_base_model_specs(random_state=42, class_weights_dict=None):
    return [
        {"wrapper_model": "Logistic Regression", "display_name": "Logistic Regression", "model": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=10000, random_state=random_state))])},
        {"wrapper_model": "SVM", "display_name": "SVM", "model": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=random_state))])},
        {"wrapper_model": "KNN", "display_name": "KNN", "model": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())])},
        {"wrapper_model": "Random Forest", "display_name": "Random Forest", "model": RandomForestClassifier(class_weight=class_weights_dict, random_state=random_state)},
    ]


def _subset_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _safe_model_key(model_name):
    return model_name.replace(" ", "_").replace("+", "plus").replace("-", "_").replace("/", "_")


def generate_oof_and_test_proba(model, X_train, y_train, X_test, cv, model_name, verbose=True):
    """
    Generate OOF probabilities on X_train and prediction probabilities on X_test.
    """
    y_train = np.asarray(y_train).astype(int)
    oof_proba = np.zeros(len(y_train), dtype=float)

    for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        fold_model = clone(model)

        X_fit = _subset_rows(X_train, fit_idx)
        X_val = _subset_rows(X_train, val_idx)
        y_fit = y_train[fit_idx]
        y_val = y_train[val_idx]

        fold_model.fit(X_fit, y_fit)
        val_proba = fold_model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

    full_model = clone(model)
    full_model.fit(X_train, y_train)
    test_proba = full_model.predict_proba(X_test)[:, 1]

    return oof_proba, test_proba, full_model


# ================================================================
# Individual base-model evaluation: model-specific feature subsets
# ================================================================

def evaluate_base_models_with_own_wrapper_features(final_df, train_combined_df, train_label, test_combined_df, test_label, base_model_specs, cv_splits=5, random_state=42):
    """
    Evaluate each base model using its own wrapper-selected feature set.

    This function is independent of union / at_least_two / intersection.
    """
    train_label = np.asarray(train_label).astype(int)
    test_label = np.asarray(test_label).astype(int)

    base_results = []
    curve_data_list = []
    base_fold_rows = []

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for spec in base_model_specs:
        wrapper_model = spec["wrapper_model"]
        display_name = spec["display_name"]
        model = spec["model"]

        features = get_model_specific_features(final_df, wrapper_model)
        train_x, test_x = build_feature_matrix_from_feature_list(
            train_combined_df=train_combined_df,
            test_combined_df=test_combined_df,
            feature_list=features,
            dataset_description=f"{display_name} model-specific wrapper features",
        )

        oof_proba = np.zeros(len(train_label), dtype=float)
        fold_rows = []

        for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(train_x, train_label), start=1):
            fold_model = clone(model)

            X_fit = _subset_rows(train_x, fit_idx)
            X_val = _subset_rows(train_x, val_idx)
            y_fit = train_label[fit_idx]
            y_val = train_label[val_idx]

            fold_model.fit(X_fit, y_fit)
            val_proba = fold_model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = val_proba

            fold_auc = float(roc_auc_score(y_val, val_proba))
            fold_auprc = float(average_precision_score(y_val, val_proba))

            fold_rows.append({
                "model": display_name,
                "fold": fold_idx,
                "n_features": len(features),
                "AUC": fold_auc,
                "AUPRC": fold_auprc,
            })

        fold_df = pd.DataFrame(fold_rows)
        mean_cv_auc = float(fold_df["AUC"].mean())
        mean_cv_auprc = float(fold_df["AUPRC"].mean())
        pooled_oof_auc = float(roc_auc_score(train_label, oof_proba))
        pooled_oof_auprc = float(average_precision_score(train_label, oof_proba))

        final_model = clone(model)
        final_model.fit(train_x, train_label)
        test_proba = final_model.predict_proba(test_x)[:, 1]

        test_auc = float(roc_auc_score(test_label, test_proba))
        test_auprc = float(average_precision_score(test_label, test_proba))

        fpr, tpr, _ = roc_curve(test_label, test_proba)
        precision, recall, _ = precision_recall_curve(test_label, test_proba)

        base_results.append({
            "model": display_name,
            "wrapper_model": wrapper_model,
            "feature_set_strategy": "model_specific",
            "feature_setting": "model-specific wrapper-selected features",
            "n_features": len(features),
            "Mean_CV_AUC": mean_cv_auc,
            "Mean_CV_AUPRC": mean_cv_auprc,
            "Pooled_OOF_AUC": pooled_oof_auc,
            "Pooled_OOF_AUPRC": pooled_oof_auprc,
            "AVG_AUC": mean_cv_auc,
            "AVG_AUPRC": mean_cv_auprc,
            "AUC": test_auc,
            "AUPRC": test_auprc,
        })

        curve_data_list.append({
            "model_name": f"{display_name}",
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "auc": test_auc,
            "auprc": test_auprc,
        })

        base_fold_rows.extend(fold_rows)

    return base_results, curve_data_list


# ================================================================
# 45-dim Stacking
# ================================================================

def fit_45dim_stacking_once(
    train_x_45,
    train_label,
    pred_x_45,
    base_models,
    cv_splits=5,
    random_state=42,
    verbose=True,
):
    """
    Fit one 45-dim feature-only Stacking model on a given training set and predict a given set.
    """
    train_label = np.asarray(train_label).astype(int)
    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    train_meta_features = {}
    pred_meta_features = {}
    fitted_base_models = {}

    for model_name, model in base_models:
        oof_proba, pred_proba, full_model = generate_oof_and_test_proba(
            model=model,
            X_train=train_x_45,
            y_train=train_label,
            X_test=pred_x_45,
            cv=inner_cv,
            model_name=model_name,
            verbose=verbose,
        )

        key = _safe_model_key(model_name)
        train_meta_features[f"{key}_OOF_proba"] = oof_proba
        pred_meta_features[f"{key}_OOF_proba"] = pred_proba
        fitted_base_models[model_name] = full_model

    train_meta_df = pd.DataFrame(train_meta_features)
    pred_meta_df = pd.DataFrame(pred_meta_features)
    pred_meta_df = pred_meta_df[train_meta_df.columns]

    meta_model = LogisticRegression(random_state=random_state)
    meta_model.fit(train_meta_df, train_label)

    pred_proba = meta_model.predict_proba(pred_meta_df)[:, 1]

    return {
        "meta_model": meta_model,
        "base_models": fitted_base_models,
        "pred_proba": pred_proba,
    }


def compute_45dim_stacking_outer_cv(
    train_x_45,
    train_label,
    base_models,
    feature_set_strategy,
    cv_splits=5,
    random_state=42,
):
    """
    Compute outer-CV performance for the full 45-dim Stacking procedure.
    """
    train_label = np.asarray(train_label).astype(int)
    outer_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    outer_oof_proba = np.zeros(len(train_label), dtype=float)
    outer_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_cv.split(train_x_45, train_label), start=1):

        outer_train_x_45 = _subset_rows(train_x_45, outer_train_idx)
        outer_val_x_45 = _subset_rows(train_x_45, outer_val_idx)
        outer_train_y = train_label[outer_train_idx]
        outer_val_y = train_label[outer_val_idx]

        fitted = fit_45dim_stacking_once(
            train_x_45=outer_train_x_45,
            train_label=outer_train_y,
            pred_x_45=outer_val_x_45,
            base_models=base_models,
            cv_splits=cv_splits,
            random_state=random_state,
            verbose=False,
        )

        val_proba = fitted["pred_proba"]
        outer_oof_proba[outer_val_idx] = val_proba

        fold_auc = float(roc_auc_score(outer_val_y, val_proba))
        fold_auprc = float(average_precision_score(outer_val_y, val_proba))

        outer_rows.append({
            "feature_set_strategy": feature_set_strategy,
            "outer_fold": outer_fold,
            "AUC": fold_auc,
            "AUPRC": fold_auprc,
        })

    outer_cv_df = pd.DataFrame(outer_rows)
    mean_outer_auc = float(outer_cv_df["AUC"].mean())
    mean_outer_auprc = float(outer_cv_df["AUPRC"].mean())
    pooled_outer_auc = float(roc_auc_score(train_label, outer_oof_proba))
    pooled_outer_auprc = float(average_precision_score(train_label, outer_oof_proba))

    return {
        "Average_AUC": mean_outer_auc,
        "Average_AUPRC": mean_outer_auprc,
        "Outer_OOF_AUC": pooled_outer_auc,
        "Outer_OOF_AUPRC": pooled_outer_auprc,
        "outer_cv_df": outer_cv_df,
        "outer_oof_proba": outer_oof_proba,
    }


def train_45dim_stacking_model(
    train_x_45,
    test_x_45,
    train_label,
    test_label,
    base_models,
    feature_set_strategy,
    feature_set_description,
    selected_features,
    min_count,
    cv_splits=5,
    random_state=42,
):
    """
    Train and evaluate one 45-dim Stacking model under one feature-set strategy.
    """
    train_label = np.asarray(train_label).astype(int)
    test_label = np.asarray(test_label).astype(int)

    outer_cv_result = compute_45dim_stacking_outer_cv(
        train_x_45=train_x_45,
        train_label=train_label,
        base_models=base_models,
        feature_set_strategy=feature_set_strategy,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    final_fit = fit_45dim_stacking_once(
        train_x_45=train_x_45,
        train_label=train_label,
        pred_x_45=test_x_45,
        base_models=base_models,
        cv_splits=cv_splits,
        random_state=random_state,
        verbose=True,
    )

    final_test_proba = final_fit["pred_proba"]

    test_auc = float(roc_auc_score(test_label, final_test_proba))
    test_auprc = float(average_precision_score(test_label, final_test_proba))

    fpr, tpr, _ = roc_curve(test_label, final_test_proba)
    precision, recall, _ = precision_recall_curve(test_label, final_test_proba)

    print(f"\nFinal Stacking performance [{feature_set_strategy}]:")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Outer-CV mean AUC: {outer_cv_result['Average_AUC']:.4f}")
    print(f"Outer-CV mean AUPRC: {outer_cv_result['Average_AUPRC']:.4f}")
    print(f"Independent test AUC: {test_auc:.4f}")
    print(f"Independent test AUPRC: {test_auprc:.4f}")

    return {
        "models": [name for name, _ in base_models],
        "num_models": len(base_models),
        "feature_set_strategy": feature_set_strategy,
        "feature_set_description": feature_set_description,
        "feature_min_count": min_count,
        "Feature_list": selected_features,
        "n_features": len(selected_features),
        "Average_AUC": outer_cv_result["Average_AUC"],
        "Average_AUPRC": outer_cv_result["Average_AUPRC"],
        "Outer_OOF_AUC": outer_cv_result["Outer_OOF_AUC"],
        "Outer_OOF_AUPRC": outer_cv_result["Outer_OOF_AUPRC"],
        "test_AUC": test_auc,
        "test_AUPRC": test_auprc,
        "roc_data": {"fpr": fpr, "tpr": tpr, "auc": test_auc},
        "prc_data": {"precision": precision, "recall": recall, "auc": test_auprc},
        "meta_model": final_fit["meta_model"],
        "base_models": final_fit["base_models"],
        "test_proba": final_test_proba,
        "outer_cv_df": outer_cv_result["outer_cv_df"],
        "outer_oof_proba": outer_cv_result["outer_oof_proba"],
    }


# ================================================================
# Reporting helpers
# ================================================================

def build_model_metrics_df(base_results, stacking_results):
    rows = []

    for result in base_results:
        rows.append({
            "model": result["model"],
            "n_features": result.get("n_features", np.nan),
            "Mean_CV_AUC": result["Mean_CV_AUC"],
            "Mean_CV_AUPRC": result["Mean_CV_AUPRC"],
            "Test_AUC": result["AUC"],
            "Test_AUPRC": result["AUPRC"]
        })

    for stacking_result in stacking_results:
        strategy = stacking_result["feature_set_strategy"]
        rows.append({
            "model": strategy,
            "n_features": stacking_result.get("n_features", np.nan),
            "Mean_CV_AUC": stacking_result["Average_AUC"],
            "Mean_CV_AUPRC": stacking_result["Average_AUPRC"],
            "Test_AUC": stacking_result["test_AUC"],
            "Test_AUPRC": stacking_result["test_AUPRC"],
        })

    columns = [
        "model",
        "n_features",
        "Mean_CV_AUC",
        "Mean_CV_AUPRC",
        "Test_AUC",
        "Test_AUPRC",
    ]

    return pd.DataFrame(rows)[columns]

def plot_combined_roc_curves(all_curve_data, output_dir, filename):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7, label="Random")
    for data in all_curve_data:
        plt.plot(data["fpr"], data["tpr"], lw=2, label=f"{data['model_name']} (AUC={data['auc']:.3f})")
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("ROC Curves of Base Models and Stacking Ensembles", fontsize=10)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3, linestyle="--")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_prc_curves(all_curve_data, output_dir, filename):
    plt.figure(figsize=(8, 6))
    for data in all_curve_data:
        plt.plot(data["recall"], data["precision"], lw=2, label=f"{data['model_name']} (AUPRC={data['auprc']:.3f})")
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.title("Precision-Recall Curves of Base Models and Stacking Ensembles", fontsize=10)
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3, linestyle="--")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ================================================================
# Main pipeline
# ================================================================

def run_stacking_model(
    train_features_file_dir=None,
    test_features_file_dir=None,
    wrapper_features_dir=None,
    wrapper_features_df=None,
    final_features_file_path=None,
    cor_sig_features=None,
    cor_sig_features_file_path=None,
    stacking_output_dir=None,
    model_metrics_file_path=None,
    cv_splits=5,
    random_state=42,
    save_outputs=True,
):
    """
    Run model-specific base-model evaluation and three Stacking feature-set strategies.
    """
    train_features_file_dir = Path(train_features_file_dir) if train_features_file_dir is not None else REPO_ROOT / "outputs" / "features" / "train"
    test_features_file_dir = Path(test_features_file_dir) if test_features_file_dir is not None else REPO_ROOT / "outputs" / "features" / "test"
    wrapper_features_dir = Path(wrapper_features_dir) if wrapper_features_dir is not None else REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features"
    final_features_file_path = Path(final_features_file_path) if final_features_file_path is not None else wrapper_features_dir / "wrapper_features.csv"
    cor_sig_features_file_path = Path(cor_sig_features_file_path) if cor_sig_features_file_path is not None else wrapper_features_dir / "correlative_significant_features_all.csv"
    stacking_output_dir = Path(stacking_output_dir) if stacking_output_dir is not None else REPO_ROOT / "outputs" / "feature_selection" / "stacking_model"
    model_metrics_file_path = Path(model_metrics_file_path) if model_metrics_file_path is not None else stacking_output_dir / "model_metrics.csv"

    train_features_file_dict, test_features_file_dict = discover_feature_files(
        train_features_file_dir,
        test_features_file_dir,
    )

    final_df = wrapper_features_df.copy() if wrapper_features_df is not None else load_final_features_dict(final_features_file_path)
    feature_relevance = get_cor_sig_dict(cor_sig_features) if cor_sig_features is not None else get_cor_sig_dict(cor_sig_features_file_path)

    train_combined_df, train_label, train_accession = load_features_and_labels(train_features_file_dict, dataset_name="train")
    test_combined_df, test_label, test_accession = load_features_and_labels(test_features_file_dict, dataset_name="test")

    classes = np.unique(train_label)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_label)
    class_weights_dict = dict(zip(classes, class_weights))

    base_model_specs = get_45dim_base_model_specs(random_state=random_state, class_weights_dict=class_weights_dict)
    selected_models = [spec["wrapper_model"] for spec in base_model_specs]
    stacking_base_models = [(spec["display_name"], spec["model"]) for spec in base_model_specs]

    if save_outputs:
        stacking_output_dir.mkdir(parents=True, exist_ok=True)

    base_results, base_curve_data = evaluate_base_models_with_own_wrapper_features(
        final_df=final_df,
        train_combined_df=train_combined_df,
        train_label=train_label,
        test_combined_df=test_combined_df,
        test_label=test_label,
        base_model_specs=base_model_specs,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    stacking_results = []
    stacking_curve_data = []

    for strategy in get_feature_set_strategies(selected_models):
        strategy_name = strategy["name"]
        min_count = strategy["min_count"]
        description = strategy["description"]

        selected_features = select_features_by_min_count(
            final_df=final_df,
            selected_models=selected_models,
            feature_relevance=feature_relevance,
            min_count=min_count,
        )


        train_x_45, test_x_45 = build_feature_matrix_from_feature_list(
            train_combined_df=train_combined_df,
            test_combined_df=test_combined_df,
            feature_list=selected_features,
            dataset_description=f"Stacking feature set [{strategy_name}]",
        )

        stacking_result = train_45dim_stacking_model(
            train_x_45=train_x_45,
            test_x_45=test_x_45,
            train_label=train_label,
            test_label=test_label,
            base_models=stacking_base_models,
            feature_set_strategy=strategy_name,
            feature_set_description=description,
            selected_features=selected_features,
            min_count=min_count,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        stacking_results.append(stacking_result)

        stacking_curve_data.append({
            "model_name": strategy_name,
            "fpr": stacking_result["roc_data"]["fpr"],
            "tpr": stacking_result["roc_data"]["tpr"],
            "precision": stacking_result["prc_data"]["precision"],
            "recall": stacking_result["prc_data"]["recall"],
            "auc": stacking_result["test_AUC"],
            "auprc": stacking_result["test_AUPRC"],
        })

    updated_model_metrics_df = build_model_metrics_df(base_results=base_results, stacking_results=stacking_results)
    all_curve_data = base_curve_data + stacking_curve_data

    if save_outputs:
        model_metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        updated_model_metrics_df.to_csv(model_metrics_file_path, index=False, encoding="utf-8-sig")
        print(f"Updated model metrics saved to: {model_metrics_file_path}")

        # Save the three Stacking models in one shared model directory.
        model_dir = stacking_output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        for stacking_result in stacking_results:
            strategy = stacking_result["feature_set_strategy"]
            model_file = model_dir / f"{strategy}_model.joblib"

            joblib.dump(
                {
                    "meta_model": stacking_result["meta_model"],
                    "base_models": stacking_result["base_models"],
                    "feature_list": stacking_result["Feature_list"],
                    "feature_set_strategy": strategy,
                    "feature_set_description": stacking_result["feature_set_description"],
                    "feature_min_count": stacking_result["feature_min_count"],
                    "stacking_protocol": "manual outer-CV + inner-OOF",
                    "threshold": 0.5,
                },
                model_file,
            )

            print(f"Stacking model saved to: {model_file.resolve()}")

        plot_combined_roc_curves(all_curve_data, stacking_output_dir, filename="roc_curves.png")
        plot_combined_prc_curves(all_curve_data, stacking_output_dir, filename="prc_curves.png")

    return stacking_results, base_results, updated_model_metrics_df, all_curve_data


if __name__ == "__main__":
    run_stacking_model(
        train_features_file_dir=REPO_ROOT / "outputs" / "features" / "train",
        test_features_file_dir=REPO_ROOT / "outputs" / "features" / "test",

        wrapper_features_dir=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features",
        final_features_file_path=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features" / "wrapper_features.csv",
        cor_sig_features_file_path=REPO_ROOT / "outputs" / "feature_selection" / "wrapper_features" / "correlative_significant_features_all.csv",

        stacking_output_dir=REPO_ROOT / "outputs" / "feature_selection" / "stacking_model",
        model_metrics_file_path=REPO_ROOT / "outputs" / "feature_selection" / "model_metrics.csv",

        cv_splits=5,
        random_state=42,
        save_outputs=True,
    )
