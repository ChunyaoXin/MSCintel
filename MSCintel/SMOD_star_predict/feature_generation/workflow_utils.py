"""Shared FASTA, feature validation, and path utilities for MSCintel."""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import pandas as pd

try:
    import yaml
except ImportError:  
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ProteinRecord:
    accession: str
    sequence: str
    label: int = None


def parse_label_from_header(header):
    match = re.search(r"(?:^|\|)label=([01])(?:\||$)", str(header), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_fasta(path, require_labels=False):
    """Read a FASTA file and return ProteinRecord objects.

    The accession is kept as the raw header token here and normalized later by
    normalize_accession(), so UniProt headers such as sp|P0C0S1|MSCS_ECOLI
    remain compatible.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records = []
    header = None
    seq_lines = []

    def flush_record():
        if header is None:
            return
        sequence = "".join(seq_lines).replace(" ", "").replace("\t", "").upper()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA record: {header}")
        label = parse_label_from_header(header)
        if require_labels and label is None:
            raise ValueError(
                f"Missing label in FASTA header: {header}. Expected a suffix such as |label=1 or |label=0."
            )
        records.append(ProteinRecord(header, sequence, label))

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                seq_lines = []
            else:
                if header is None:
                    raise ValueError(f"Sequence line appears before first FASTA header in {path}")
                seq_lines.append(line)
        flush_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def write_labelled_fasta(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            label = "" if record.label is None else f"|label={record.label}"
            handle.write(f">{record.accession}{label}\n")
            seq = str(record.sequence)
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i+80] + "\n")


def load_config(config_path, repo_root=None):
    config_path = Path(config_path)
    if repo_root is not None and not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        config = yaml.safe_load(text) or {}
    else:
        config = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip().strip('"').strip("'")

    if repo_root is None:
        repo_root = config_path.resolve().parent
    config.setdefault("repo_root", str(Path(repo_root).resolve()))
    return config


def resolve_path(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


LABEL_SUFFIX_RE = re.compile(r"\|label=[01](?:\|.*)?$", flags=re.IGNORECASE)
UNIPROT_PREFIXES = {"sp", "tr"}


def normalize_accession(accession):
    """
    Normalize FASTA/APOD accessions to the identifier used in prediction filenames.

    Supported FASTA headers include:
        >AccessionName
        >AccessionName description text
        >AccessionName|label=1
        >sp|P0C0S1|MSCS_ECOLI Small-conductance mechanosensitive channel ...
        >tr|A0Axxxx|ENTRY_NAME Description ...

    Rules:
        1. Remove the leading '>' if present.
        2. Use only the first whitespace-separated token.
        3. Remove a trailing |label=0 or |label=1 suffix.
        4. For UniProt-style tokens sp|ACCESSION|ENTRY or tr|ACCESSION|ENTRY,
           return ACCESSION.
        5. Otherwise, return the cleaned first token.
    """
    header = str(accession).strip()
    if header.startswith(">"):
        header = header[1:].strip()

    token = header.split()[0] if header else ""
    token = LABEL_SUFFIX_RE.sub("", token)

    parts = token.split("|")
    if len(parts) >= 3 and parts[0] in UNIPROT_PREFIXES and parts[1]:
        return parts[1]

    return token


def normalize_records(records):
    return [
        ProteinRecord(normalize_accession(record.accession), record.sequence, record.label)
        for record in records
    ]


def read_labelled_fasta(path):
    return normalize_records(read_fasta(path, require_labels=True))


def read_optional_label_fasta(path):
    return normalize_records(read_fasta(path, require_labels=False))


def read_raw_fasta(path, label):
    return [
        ProteinRecord(normalize_accession(record.accession), record.sequence, label)
        for record in read_fasta(path, require_labels=False)
    ]


def records_frame(records):
    rows = []
    for record in records:
        row = {"AccessionName": record.accession}
        if record.label is not None:
            row["label"] = record.label
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_features(frame, records, require_complete=True):
    """Attach FASTA labels when available and validate accession coverage."""
    index = records_frame(records)
    if "AccessionName" not in frame.columns:
        raise ValueError("Feature frame does not contain AccessionName")

    frame = frame.copy()
    frame["AccessionName"] = frame["AccessionName"].map(normalize_accession)

    if frame["AccessionName"].duplicated().any():
        duplicates = frame.loc[frame["AccessionName"].duplicated(), "AccessionName"].tolist()
        raise ValueError(f"Duplicate prediction results: {duplicates[:5]}")

    missing = sorted(set(index.AccessionName) - set(frame.AccessionName))
    extra = sorted(set(frame.AccessionName) - set(index.AccessionName))
    if extra:
        raise ValueError(f"Prediction results not present in FASTA: {extra[:10]}")
    if missing and require_complete:
        raise ValueError(
            f"Missing prediction results for {len(missing)} FASTA accessions; first missing: {missing[:10]}"
        )

    result = index.merge(frame, on="AccessionName", how="inner", validate="one_to_one")
    result.fillna(0, inplace=True)

    if "label" in result.columns:
        result = result[[column for column in result.columns if column != "label"] + ["label"]]
        return result.sort_values(["label", "AccessionName"], ascending=[False, True]).reset_index(drop=True)

    return result.sort_values(["AccessionName"], ascending=[True]).reset_index(drop=True)


def resolve_project_path(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path
