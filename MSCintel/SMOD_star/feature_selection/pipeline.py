# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import yaml

from merge_select_correlative_features import run_correlative_selection
from merge_select_significant_features import run_significant_selection
from correlative_significant_features import run_correlative_significant_selection
from wrapper_feature_selection import run_wrapper_feature_selection
from stacking_merge_model import run_stacking_model


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path):
    """
    Load YAML configuration file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(path_value):
    """
    Convert a relative path in config.yaml to an absolute path under REPO_ROOT.
    """
    path_value = Path(path_value)

    if path_value.is_absolute():
        return path_value

    return REPO_ROOT / path_value


def run_pipeline(config_path=None):
    """
    Run the full feature-selection and Stacking-model pipeline.

    Pipeline order:
    1. merge_select_correlative_features.py
    2. merge_select_significant_features.py
    3. correlative_significant_features.py
    4. wrapper_feature_selection.py
    5. stacking_merge_model.py
    """
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    config = load_config(config_path)
    fs_cfg = config["feature_selection"]
    save_cfg = fs_cfg.get("save_outputs", {})

    train_features_dir = resolve_path(fs_cfg["train_features_dir"])
    test_features_dir = resolve_path(fs_cfg["test_features_dir"])
    feature_value_file = resolve_path(fs_cfg["feature_value_file"])

    correlative_results_dir = resolve_path(fs_cfg["correlative_results_dir"])
    significant_results_dir = resolve_path(fs_cfg["significant_results_dir"])
    wrapper_features_dir = resolve_path(fs_cfg["wrapper_features_dir"])
    stacking_output_dir = resolve_path(fs_cfg["stacking_output_dir"])
    model_metrics_file = resolve_path(fs_cfg["model_metrics_file"])

    p_value_threshold = fs_cfg.get("p_value_threshold", 0.05)
    alpha = fs_cfg.get("alpha", 0.05)
    pcc = fs_cfg.get("pcc", 0.7)
    outer_iterations = fs_cfg.get("outer_iterations", 10)
    inter_iterations = fs_cfg.get("inter_iterations", 100)
    repeat_times = fs_cfg.get("repeat_times", 0.7)
    cv_splits = fs_cfg.get("cv_splits", 5)
    random_state = fs_cfg.get("random_state", 42)

    print("\n========== Step 1: Correlative feature selection ==========")

    all_pbc_results, correlative_results, all_correlative_features_df = (
        run_correlative_selection(
            train_features_file_dir=train_features_dir,
            correlative_results_dir=correlative_results_dir,
            p_value_threshold=p_value_threshold,
            save_outputs=save_cfg.get("merge_select_correlative_features", False)
        )
    )

    print("\n========== Step 2: Significant feature selection ==========")

    significant_results, all_sig_features_df = run_significant_selection(
        train_features_file_dir=train_features_dir, correlative_results=correlative_results,
        correlative_base_dir=correlative_results_dir / "correlative_features", significant_results_dir=significant_results_dir,
        outer_iterations=outer_iterations, inter_iterations=inter_iterations, repeat_times=repeat_times, alpha=alpha, pcc=pcc,
        file_path=feature_value_file, save_outputs=save_cfg.get("merge_select_significant_features", False), random_state=random_state)

    print("\n========== Step 3: Correlative-significant feature intersection ==========")

    cor_sig_all_df, cor_sig_30_df = run_correlative_significant_selection(
        all_correlative_features_pbc=all_correlative_features_df,
        all_significant_features=all_sig_features_df,
        wrapper_features_dir=wrapper_features_dir,
        top_ratio=fs_cfg.get("top_ratio", 0.3),
        save_outputs=save_cfg.get("correlative_significant_features", False))

    print("\n========== Step 4: Wrapper feature selection ==========")
        
    wrapper_features_df, selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list = run_wrapper_feature_selection(
        train_features_file_dir=train_features_dir,
        correlative_significant_features=cor_sig_30_df,
        wrapper_features_dir=wrapper_features_dir,
        cv_splits=cv_splits,
        random_state=random_state,
        save_outputs=save_cfg.get("wrapper_feature_selection", False))

    print("\n========== Step 5: Stacking model ==========")

    stacking_result, base_results, updated_model_metrics_df, all_curve_data = (
        run_stacking_model(
            train_features_file_dir=train_features_dir,
            test_features_file_dir=test_features_dir,
            wrapper_features_dir=wrapper_features_dir,
            wrapper_features_df=wrapper_features_df,
            cor_sig_features=cor_sig_all_df,
            stacking_output_dir=stacking_output_dir,
            model_metrics_file_path=model_metrics_file,
            cv_splits=cv_splits,
            random_state=random_state,
            save_outputs=save_cfg.get("stacking_merge_model", True))
    )

    print("\n========== Pipeline completed ==========")
    print(f"Model metrics file: {model_metrics_file}")
    print(f"Stacking output directory: {stacking_output_dir}")

    return {
        "all_pbc_results": all_pbc_results,
        "correlative_results": correlative_results,
        "all_correlative_features_df": all_correlative_features_df,
        "significant_results": significant_results,
        "all_sig_features_df": all_sig_features_df,
        "cor_sig_all_df": cor_sig_all_df,
        "cor_sig_30_df": cor_sig_30_df,
        "wrapper_features_df": wrapper_features_df,
        "model_metrics_df": updated_model_metrics_df,
        "stacking_result": stacking_result,
        "base_results": base_results,
        "all_curve_data": all_curve_data,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full feature-selection and Stacking-model pipeline.")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml", help="Path to config.yaml.",)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(config_path=args.config)