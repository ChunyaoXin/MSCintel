"""Command-line argument and path configuration for feature calculation scripts."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def _parser():
    parser = argparse.ArgumentParser(description="MSCintel feature calculation")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--dataset", choices=("train", "test", "prediction"), default="train")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--input-fasta", default=None)
    parser.add_argument("--standardized-prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--allow-missing", action="store_true", help="write only accessions with predictions instead of failing on incomplete coverage")
    return parser


args, _unknown = _parser().parse_known_args()


def dataset_config(dataset):
    if dataset == "train":
        dataset_name = args.dataset_name or "train"
        prefix = args.prefix or ""
        fasta = Path(args.input_fasta) if args.input_fasta else Path("Data/train.fasta")
    elif dataset == "test":
        dataset_name = args.dataset_name or "test"
        prefix = args.prefix or ""
        fasta = Path(args.input_fasta) if args.input_fasta else Path("Data/test.fasta")
    elif dataset == "prediction":
        dataset_name = args.dataset_name or "prediction"
        prefix = args.prefix or "predict"
        if not args.input_fasta:
            raise ValueError("--input-fasta is required when --dataset prediction")
        fasta = Path(args.input_fasta)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    standardized_root = (
        Path(args.standardized_prediction_dir)
        if args.standardized_prediction_dir
        else Path("Data/standardized_predictions") / dataset_name
    )
        
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/features") / dataset_name

    if not fasta.is_absolute():
        fasta = PROJECT_ROOT / fasta
    if not standardized_root.is_absolute():
        standardized_root = PROJECT_ROOT / standardized_root
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
        
    manifest = Path(args.manifest_path) if args.manifest_path else standardized_root / "prediction_manifest.csv"
    
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest

    return {
        "dataset_name": dataset_name,
        "prefix": prefix,
        "fasta": fasta,
        "manifest": manifest,
        "linker": standardized_root / "linker.apod",
        "standardized_prediction_dir": standardized_root,
        "output_dir": output_dir,
        "reference_dir": None,
    }
