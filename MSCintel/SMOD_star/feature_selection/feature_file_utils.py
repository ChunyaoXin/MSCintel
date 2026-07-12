from pathlib import Path

STANDARD_FEATURE_FILES = [
    "AA_frequency_features.csv",
    "AAindex_features.csv",
    "AA_pairs_features.csv",
    "SS_features.csv",
    "SS_RSA_features.csv",
    "disorder_features.csv",
    "linker_features.csv",
    "Conservation_features.csv",
]


def discover_csv_files(feature_dir):
    feature_dir = Path(feature_dir)

    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    if not feature_dir.is_dir():
        raise NotADirectoryError(f"Feature path is not a directory: {feature_dir}")

    order_map = {
        filename: i for i, filename in enumerate(STANDARD_FEATURE_FILES)
    }

    def sort_key(path):
        filename = path.name
        if filename in order_map:
            return (0, order_map[filename], filename.lower())
        return (1, len(order_map), filename.lower())

    csv_files = sorted(feature_dir.glob("*.csv"), key=sort_key)

    if not csv_files:
        raise FileNotFoundError(f"No CSV feature files found in: {feature_dir}")

    return csv_files