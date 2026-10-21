"""Shared FASTA, feature validation, and path utilities for MSCintel."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .fasta_utils import ProteinRecord, read_fasta, write_labelled_fasta
except ImportError:
    from fasta_utils import ProteinRecord, read_fasta, write_labelled_fasta

def read_labelled_fasta(path):
    return read_fasta(path, require_labels=True)


def read_optional_label_fasta(path):
    return read_fasta(path, require_labels=False)


def read_raw_fasta(path, label):
    return [ProteinRecord(record.accession, record.sequence, label) for record in read_fasta(path, require_labels=False)]


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
