"""Generate labelled train.fasta and test.fasta from four raw FASTA files.

If train.fasta and test.fasta already exist and are already in labelled format,
this script will not modify them.
"""

import re
from Parameter import DATA_DIR, TRAIN_FASTA, TEST_FASTA
from workflow_utils import read_raw_fasta, write_labelled_fasta


LABEL_PATTERN = re.compile(r"(^|\|)label=[01](\||$)")


def is_labelled_fasta(fasta_path):
    """
    Check whether a FASTA file already has labelled headers:
    >AccessionName|label=0
    or
    >AccessionName|label=1
    """
    if not fasta_path.exists():
        return False

    header_count = 0

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                header_count += 1
                header = line[1:]

                if not LABEL_PATTERN.search(header):
                    return False

    return header_count > 0


def merge_pair(positive_name, negative_name, output_path):
    records = read_raw_fasta(DATA_DIR / positive_name, 1)
    records.extend(read_raw_fasta(DATA_DIR / negative_name, 0))
    write_labelled_fasta(records, output_path)
    return records


def main():
    train_ready = is_labelled_fasta(TRAIN_FASTA)
    test_ready = is_labelled_fasta(TEST_FASTA)

    if train_ready and test_ready:
        print(f"{TRAIN_FASTA} already exists and is in labelled FASTA format.")
        print(f"{TEST_FASTA} already exists and is in labelled FASTA format.")
        print("No changes were made.")
        print("Completed!!")
        return

    if TRAIN_FASTA.exists() and not train_ready:
        raise ValueError(f"{TRAIN_FASTA} already exists but is not in labelled FASTA format. Please check the file or remove it before regenerating.")

    if TEST_FASTA.exists() and not test_ready:
        raise ValueError(f"{TEST_FASTA} already exists but is not in labelled FASTA format. Please check the file or remove it before regenerating.")

    train = merge_pair("positive_train_data.fasta", "negative_train_data.fasta", TRAIN_FASTA,)

    test = merge_pair("positive_test_data.fasta", "negative_test_data.fasta", TEST_FASTA,)

    print(f"Wrote {len(train)} records to {TRAIN_FASTA}")
    print(f"Wrote {len(test)} records to {TEST_FASTA}")
    print("Completed!!")


if __name__ == "__main__":
    main()