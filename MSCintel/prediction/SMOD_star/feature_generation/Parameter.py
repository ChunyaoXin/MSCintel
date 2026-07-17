"""Project-relative configuration shared by feature calculation scripts."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_utils import load_config, resolve_path


def _parser():
    parser = argparse.ArgumentParser(description="MSCintel feature calculation")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--dataset", choices=("train", "test", "prediction"), default="train")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--standardized-prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write only accessions with predictions instead of failing on incomplete coverage",
    )
    return parser


args, _unknown = _parser().parse_known_args()
CONFIG = load_config(args.config, repo_root=PROJECT_ROOT)

DATA_DIR = resolve_path(PROJECT_ROOT, CONFIG["raw_data_dir"])
RAW_PREDICTIONS_DIR = resolve_path(PROJECT_ROOT, CONFIG["raw_predicted_features_dir"])
STANDARDIZED_PREDICTIONS_DIR = resolve_path(PROJECT_ROOT, CONFIG["standardized_predicted_features_dir"])
FEATURE_OUTPUT_DIR = resolve_path(PROJECT_ROOT, CONFIG["feature_output_dir"])

TRAIN_FASTA = resolve_path(PROJECT_ROOT, CONFIG["train_fasta"])
TEST_FASTA = resolve_path(PROJECT_ROOT, CONFIG["test_fasta"])

# Optional: keep old names for backward compatibility
TR994_FASTA = TRAIN_FASTA
TE332_FASTA = TEST_FASTA


def dataset_config(dataset):
    if dataset == "train":
        dataset_name = args.dataset_name or "train"
        prefix = args.prefix or ""
        fasta = Path(args.input_fasta) if args.input_fasta else TRAIN_FASTA
    elif dataset == "test":
        dataset_name = args.dataset_name or "test"
        prefix = args.prefix or ""
        fasta = Path(args.input_fasta) if args.input_fasta else TEST_FASTA
    elif dataset == "prediction":
        dataset_name = args.dataset_name or "prediction"
        prefix = "predict" if args.prefix is None else args.prefix
        if not args.input_fasta:
            raise ValueError("--input-fasta is required when --dataset prediction")
        fasta = Path(args.input_fasta)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    standardized_root = (
        Path(args.standardized_prediction_dir)
        if args.standardized_prediction_dir
        else STANDARDIZED_PREDICTIONS_DIR / dataset_name
    )
    output_dir = Path(args.output_dir) if args.output_dir else FEATURE_OUTPUT_DIR / dataset_name
    if not fasta.is_absolute():
        fasta = PROJECT_ROOT / fasta
    if not standardized_root.is_absolute():
        standardized_root = PROJECT_ROOT / standardized_root
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return {
        "dataset_name": dataset_name,
        "prefix": prefix,
        "fasta": fasta,
        "manifest": standardized_root / "prediction_manifest.csv",
        "linker": standardized_root / "linker.apod",
        "standardized_prediction_dir": standardized_root,
        "output_dir": output_dir,
        "reference_dir": None,
    }
