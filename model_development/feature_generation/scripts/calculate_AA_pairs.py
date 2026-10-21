"""Calculate spaced amino-acid-pair frequencies from labelled FASTA."""

import pandas as pd

from sequence_feature_common import load_context, save_sequence_features


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
COLUMNS = [a + "*" * gap + b for a in AMINO_ACIDS for b in AMINO_ACIDS for gap in range(6)]


def calculate(records):
    rows = []
    for record in records:
        counts = {pair: 0 for pair in COLUMNS}
        for gap in range(6):
            for index in range(len(record.sequence) - gap - 1):
                pair = record.sequence[index] + "*" * gap + record.sequence[index + gap + 1]
                if pair in counts:
                    counts[pair] += 1
        values = [
            counts[pair] / (len(record.sequence) - pair.count("*") - 1)
            if len(record.sequence) - pair.count("*") - 1 > 0 else 0
            for pair in COLUMNS
        ]
        rows.append([record.accession, *values])
    return pd.DataFrame(rows, columns=["AccessionName", *COLUMNS])


if __name__ == "__main__":
    config, records, output = load_context("AA_pairs")
    frame = calculate(records)
    # The uploaded training reference uses NA_A for the adjacent N-A pair,
    # while the test reference uses the original NA name. Preserve that
    # established downstream contract without changing the calculation.
    if "NA" in frame.columns:
        frame.rename(columns={"NA": "NA_A"}, inplace=True)
    save_sequence_features(frame, records, output)
