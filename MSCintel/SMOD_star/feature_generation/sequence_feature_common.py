"""Common output handling for FASTA-derived feature scripts."""

from Parameter import args, dataset_config
from workflow_utils import read_optional_label_fasta, finalize_features


FEATURE_OUTPUT_NAMES = {
    "AA": "AA_frequency",
    "AA_pairs": "AA_pairs",
    "AAindex": "AAindex",
    "Conservation": "Conservation",
    "Disorder": "disorder",
    "Linker": "linker",
    "SS": "SS",
    "SS_RSA": "SS_RSA",
}


def output_name(config, feature_name):
    key = FEATURE_OUTPUT_NAMES.get(feature_name, feature_name)
    prefix = f"{config['prefix']}_" if config.get("prefix") else ""
    return config["output_dir"] / f"{prefix}{key}_features.csv"


def load_context(feature_name):
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    return config, records, output_name(config, feature_name)


def save_sequence_features(frame, records, output):
    result = finalize_features(frame, records)
    result.to_csv(output, index=False)
#    print(f"Wrote {len(result)} rows to {output}")
