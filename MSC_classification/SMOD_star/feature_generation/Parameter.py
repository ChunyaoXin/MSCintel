
"""Project-relative configuration shared by feature calculation scripts."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import load_config
from utils.path_utils import resolve_path


def _parser():
    parser = argparse.ArgumentParser(description="MSCintel feature calculation")

    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--dataset", choices=("train", "test", "prediction"), default="train")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--standardized-prediction-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--allow-missing", action="store_true", help="Allow incomplete prediction coverage.")

    return parser


args, _unknown = _parser().parse_known_args()
CONFIG = load_config(args.config, repo_root=PROJECT_ROOT)

TRAIN_FEATURE_CONFIG = CONFIG["train"]["feature_generation"]
PREDICT_FEATURE_CONFIG = CONFIG["predict"]["feature_generation"]

TRAIN_FASTA = resolve_path(PROJECT_ROOT, TRAIN_FEATURE_CONFIG["train_fasta"])
TEST_FASTA = resolve_path(PROJECT_ROOT, TRAIN_FEATURE_CONFIG["test_fasta"])
TRAIN_PREDICTION_DIR = resolve_path(PROJECT_ROOT, TRAIN_FEATURE_CONFIG["predicted_features_dir"])
PREDICT_FASTA = resolve_path(PROJECT_ROOT, PREDICT_FEATURE_CONFIG["input_fasta"])
PREDICT_PREDICTION_DIR = resolve_path(PROJECT_ROOT, PREDICT_FEATURE_CONFIG["feature_data_dir"])
TRAIN_FEATURE_OUTPUT_DIR = resolve_path(PROJECT_ROOT, TRAIN_FEATURE_CONFIG["train_features_dir"])
TEST_FEATURE_OUTPUT_DIR = resolve_path(PROJECT_ROOT, TRAIN_FEATURE_CONFIG["test_features_dir"])
PREDICT_FEATURE_OUTPUT_DIR = resolve_path(PROJECT_ROOT, PREDICT_FEATURE_CONFIG["output_dir"])

# Optional: keep old names for backward compatibility
TR994_FASTA = TRAIN_FASTA
TE332_FASTA = TEST_FASTA


def dataset_config(dataset):
    """Resolve paths for train, test, or prediction data."""

    if dataset == "train":
        dataset_name = args.dataset_name or "train"
        prefix = args.prefix or ""

        default_fasta = TRAIN_FASTA
        default_prediction_dir = TRAIN_PREDICTION_DIR / "train"
        default_output_dir = TRAIN_FEATURE_OUTPUT_DIR

    elif dataset == "test":
        dataset_name = args.dataset_name or "test"
        prefix = args.prefix or ""

        default_fasta = TEST_FASTA
        default_prediction_dir = TRAIN_PREDICTION_DIR / "test"
        default_output_dir = TEST_FEATURE_OUTPUT_DIR

    elif dataset == "prediction":
        dataset_name = args.dataset_name or "prediction"
        prefix = args.prefix or ""

        default_fasta = PREDICT_FASTA
        default_prediction_dir = PREDICT_PREDICTION_DIR
        default_output_dir = PREDICT_FEATURE_OUTPUT_DIR

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Command-line arguments override YAML defaults.
    fasta = resolve_path(PROJECT_ROOT, args.input_fasta if args.input_fasta else default_fasta)
    prediction_dir = resolve_path(PROJECT_ROOT, args.standardized_prediction_dir if args.standardized_prediction_dir else default_prediction_dir)
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir if args.output_dir else default_output_dir)
    manifest = resolve_path(PROJECT_ROOT, args.manifest_path if args.manifest_path else prediction_dir / "prediction_manifest.csv")

    return {
        "dataset_name": dataset_name,
        "prefix": prefix,
        "fasta": fasta,
        "manifest": manifest,
        "linker": prediction_dir / "linker.apod",
        "standardized_prediction_dir": prediction_dir,
        "output_dir": output_dir,
        "reference_dir": None,
    }
