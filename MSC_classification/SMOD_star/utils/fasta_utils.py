from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional


LABEL_PATTERN = re.compile(r"(?:^|[|\s;])label[=:](0|1)(?:$|[|\s;])")


@dataclass(frozen=True)
class ProteinRecord:
    accession: str
    sequence: str
    label: Optional[int] = None


def parse_header(header):
    """Parse accession and optional label from FASTA header."""
    header = header.strip()
    first_token = header.split()[0]

    # UniProt FASTA: sp|Q5YCC5|TMC7_CHICK or tr|ACCESSION|ENTRY
    parts = first_token.split("|")

    if len(parts) >= 3 and parts[0].lower() in ("sp", "tr"):
        accession = parts[1]
    else:
        # Standard or labelled FASTA: Q5YCC5 or Q5YCC5|label=1
        accession = parts[0]

    match = LABEL_PATTERN.search(header)
    label = int(match.group(1)) if match else None

    return accession, label


def read_fasta(path, require_labels=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required FASTA file is missing: {path}")
    records = []
    header = None
    sequence_parts = []

    def finish(line_number):
        if header is None:
            return
        accession, label = parse_header(header)
        if require_labels and label is None:
            raise ValueError(
                f"Missing label before line {line_number} in {path}; expected >AccessionName|label=0/1"
            )
        sequence = "".join(sequence_parts)
        if not sequence:
            raise ValueError(f"Empty sequence for {accession} in {path}")
        records.append(ProteinRecord(accession, sequence, label))

    with path.open(encoding="utf-8") as handle:
        line_number = 0
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish(line_number)
                header = line[1:]
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(f"Sequence before first header at line {line_number} in {path}")
                sequence_parts.append(line)
        finish(line_number + 1)
    accessions = [record.accession for record in records]
    duplicates = sorted({name for name in accessions if accessions.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicated accessions in {path}: {duplicates[:10]}")
    return records


def write_labelled_fasta(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if record.label is None:
                raise ValueError(f"Cannot write labelled FASTA for {record.accession}: label is missing")
            handle.write(f">{record.accession}|label={record.label}\n{record.sequence}\n")
