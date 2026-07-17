"""Calculate amino-acid frequencies from labelled FASTA files."""

from collections import Counter
import pandas as pd

from sequence_feature_common import load_context, save_sequence_features


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def calculate(records):
    rows = []
    for record in records:
        counts = Counter(record.sequence)
        rows.append(
            [record.accession]
            + [counts.get(amino_acid, 0) / len(record.sequence) for amino_acid in AMINO_ACIDS]
        )
    return pd.DataFrame(rows, columns=["AccessionName", *AMINO_ACIDS])


if __name__ == "__main__":
    _, records, output = load_context("AA")
    save_sequence_features(calculate(records), records, output)
