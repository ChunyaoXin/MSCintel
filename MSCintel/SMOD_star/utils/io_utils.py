from pathlib import Path


DEFAULT_CONFIG = {
    "repo_root": ".",
    "train_fasta": "Data/TR994.fasta",
    "test_fasta": "Data/TE332.fasta",
    "raw_data_dir": "Data",
    "raw_predicted_features_dir": "predicted_features",
    "standardized_predicted_features_dir": "predicted_features/standardized",
    "feature_output_dir": "outputs/features",
    "selection_output_dir": "outputs/feature_selection",
    "evaluation_output_file": "outputs/evaluation/evaluation_summary.csv",
    "model_output_dir": "outputs/models",
    "prediction_output_dir": "outputs/predictions",
    "save_intermediate": False,
    "random_seed": 42,
}


def _parse_value(value):
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_config(config_path=None, repo_root=None):
    config = dict(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line in {path}: {raw_line}")
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_value(value)
    if repo_root is not None:
        config["repo_root"] = str(repo_root)
    elif config_path:
        root_value = Path(config["repo_root"])
        config["repo_root"] = str(root_value if root_value.is_absolute() else Path(config_path).resolve().parent / root_value)
    else:
        config["repo_root"] = str(Path(__file__).resolve().parents[1])
    return config
