"""Calculate mean AAindex features from labelled FASTA."""

import pandas as pd
from aaindex import aaindex1

from sequence_feature_common import load_context, save_sequence_features


EXCLUDED = {
    "AVBF000101", "AVBF000102", "AVBF000103", "AVBF000104", "AVBF000105",
    "AVBF000106", "AVBF000107", "AVBF000108", "AVBF000109", "YANJ020101",
    "GUYH850103", "ROSM880104", "ROSM880105",
}


def calculate(records):
    codes = [code for code in aaindex1.record_codes() if code not in EXCLUDED]
    data = {"AccessionName": [record.accession for record in records]}
    for code in codes:
        values = aaindex1[code]["values"]
        data[code] = [
            sum(values.get(amino_acid, 0) for amino_acid in record.sequence) / len(record.sequence)
            for record in records
        ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    _, records, output = load_context("AAindex")
    save_sequence_features(calculate(records), records, output)
