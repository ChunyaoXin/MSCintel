"""Standardize raw external prediction outputs into dataset-level inputs."""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_utils import ProteinRecord, read_fasta, load_config, resolve_path


GROUPS = {
    ("train", 1): "positive_train",
    ("train", 0): "negative_train",
    ("test", 1): "positive_test",
    ("test", 0): "negative_test",
}

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


def relative(path, repo_root):
    """Return a repo-relative path when possible; otherwise keep an absolute path."""
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    return False


def copy_tree_or_file(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        return False
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return True


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def find_linker_apod(raw_root):
    raw_root = Path(raw_root)
    candidates = [
        raw_root / "Linker" / "linker.apod",
        raw_root / "Linker" / "Linker.apod",
        raw_root / "linker.apod",
        raw_root / "Linker_file_for_select_features" / "linker.apod",
    ]
    found = first_existing(candidates)
    if found is not None:
        return found
    linker_dir = raw_root / "Linker"
    if linker_dir.is_dir():
        apods = sorted(linker_dir.glob("*.apod"))
        if apods:
            return apods[0]
    return raw_root / "Linker" / "linker.apod"


def read_apod_blocks(path):
    blocks = {}
    accession = None
    lines = []
    if not Path(path).is_file():
        return blocks
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if accession is not None:
                    blocks[accession] = lines
                accession = normalize_accession(line[1:].strip())
                lines = [f">{accession}"]
            elif accession is not None:
                lines.append(line)
        if accession is not None:
            blocks[accession] = lines
    return blocks


def is_standardized_dataset_ready(dataset_dir, fasta_path, require_labels=True):
    """
    Check whether the dataset has already been standardized.

    Required standardized structure:
        dataset_dir/
            prediction_manifest.csv
            linker.apod
            hhm/
            disorder/
            SS/

    The accession list in prediction_manifest.csv must match the FASTA file.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "prediction_manifest.csv"
    linker_path = dataset_dir / "linker.apod"

    if not manifest_path.is_file():
        return False
    if not linker_path.is_file():
        return False
    if not (dataset_dir / "hhm").is_dir():
        return False
    if not (dataset_dir / "disorder").is_dir():
        return False
    if not (dataset_dir / "SS").is_dir():
        return False

    records = normalize_records(read_fasta(fasta_path, require_labels=require_labels))
    fasta_accessions = [record.accession for record in records]

    manifest = pd.read_csv(manifest_path)
    if "accession" not in manifest.columns:
        return False

    manifest_accessions = manifest["accession"].astype(str).map(normalize_accession).tolist()
    if manifest_accessions != fasta_accessions:
        return False

    required_columns = [
        "hhm_available",
        "disorder_available",
        "ss_available",
        "linker_available",
    ]
    for col in required_columns:
        if col not in manifest.columns:
            return False
        if not manifest[col].astype(bool).all():
            return False

    return True


def build_labelled_dataset(split, dataset_name, fasta_path, raw_root, standardized_root, repo_root):
    dataset_dir = standardized_root / dataset_name
    if is_standardized_dataset_ready(dataset_dir, fasta_path, require_labels=True):
        print(f"{dataset_dir} is already standardized. Skip standardization.")
        return pd.read_csv(dataset_dir / "prediction_manifest.csv")

    records = normalize_records(read_fasta(fasta_path, require_labels=True))
    linker_root = raw_root / "Linker_file_for_select_features"
    raw_linkers = {}
    for label in (1, 0):
        group = GROUPS[(split, label)]
        raw_path = linker_root / f"Linker_{group}_data.apod"
        raw_linkers[group] = (raw_path, read_apod_blocks(raw_path))

    rows = []
    merged_linker_lines = []
    for record in records:
        group = GROUPS[(split, record.label)]
        raw_hhm = raw_root / "HHM_file_for_select_features" / f"hhm_{group}_data" / f"{record.accession}.hhm"
        raw_disorder = raw_root / "IDR_file_for_select_features" / f"IDR_{group}_data" / f"{record.accession}.txt"
        raw_ss = raw_root / "SS_file_for_select_features" / f"SS_{group}_data" / record.accession / "seq.phipsi"

        hhm = dataset_dir / "hhm" / f"{record.accession}.hhm"
        disorder = dataset_dir / "disorder" / f"{record.accession}.txt"
        ss = dataset_dir / "SS" / record.accession / "seq.phipsi"
        hhm_available = copy_file(raw_hhm, hhm)
        disorder_available = copy_file(raw_disorder, disorder)
        ss_available = copy_file(raw_ss, ss)

        _, linker_blocks = raw_linkers[group]
        linker_available = record.accession in linker_blocks
        if linker_available:
            merged_linker_lines.extend(linker_blocks[record.accession])

        rows.append({
            "accession": record.accession,
            "label": record.label,
            "source_group": group,
            "sequence_length": len(record.sequence),
            "hhm_raw_path": relative(hhm, repo_root),
            "hhm_available": hhm_available,
            "disorder_raw_path": relative(disorder, repo_root),
            "disorder_available": disorder_available,
            "ss_raw_path": relative(ss, repo_root),
            "ss_available": ss_available,
            "linker_available": linker_available,
            "linker_standardized_path": relative(dataset_dir / "linker.apod", repo_root),
        })

    dataset_dir.mkdir(parents=True, exist_ok=True)
    linker_output = dataset_dir / "linker.apod"
    linker_output.write_text("\n".join(merged_linker_lines) + ("\n" if merged_linker_lines else ""), encoding="utf-8")
    manifest_path = dataset_dir / "prediction_manifest.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    return frame


def build_prediction_dataset(dataset_name, fasta_path, standardized_root, repo_root):
    """Build only a manifest for an already standardized prediction dataset."""
    dataset_dir = standardized_root / dataset_name
    if is_standardized_dataset_ready(dataset_dir, fasta_path, require_labels=False):
        print(f"{dataset_dir} is already standardized. Skip standardization.")
        return pd.read_csv(dataset_dir / "prediction_manifest.csv")

    records = normalize_records(read_fasta(fasta_path, require_labels=False))
    rows = []
    for record in records:
        hhm = dataset_dir / "hhm" / f"{record.accession}.hhm"
        disorder = dataset_dir / "disorder" / f"{record.accession}.txt"
        ss = dataset_dir / "SS" / record.accession / "seq.phipsi"
        rows.append({
            "accession": record.accession,
            "label": "" if record.label is None else record.label,
            "source_group": dataset_name,
            "sequence_length": len(record.sequence),
            "hhm_raw_path": relative(hhm, repo_root),
            "hhm_available": hhm.is_file(),
            "disorder_raw_path": relative(disorder, repo_root),
            "disorder_available": disorder.is_file(),
            "ss_raw_path": relative(ss, repo_root),
            "ss_available": ss.is_file(),
            "linker_available": (dataset_dir / "linker.apod").is_file(),
            "linker_standardized_path": relative(dataset_dir / "linker.apod", repo_root),
        })

    manifest_path = dataset_dir / "prediction_manifest.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    return frame


def build_single_raw_dataset(dataset_name, fasta_path, raw_root, standardized_root, repo_root):
    """
    Standardize one unlabeled/labeled FASTA dataset from this raw layout:

        raw_root/
            HHM/AccessionName.hhm
            IDR/AccessionName.txt
            Linker/linker.apod
            SS/AccessionName/seq.phipsi
    """
    dataset_dir = Path(standardized_root) / dataset_name
    if is_standardized_dataset_ready(dataset_dir, fasta_path, require_labels=False):
        print(f"{dataset_dir} is already standardized. Skip standardization.")
        return pd.read_csv(dataset_dir / "prediction_manifest.csv")

    raw_root = Path(raw_root)
    records = normalize_records(read_fasta(fasta_path, require_labels=False))

    hhm_root = raw_root / "HHM"
    disorder_root = raw_root / "IDR"
    ss_root = raw_root / "SS"
    linker_input = find_linker_apod(raw_root)
    linker_blocks = read_apod_blocks(linker_input)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    merged_linker_lines = []
    rows = []

    for record in records:
        raw_hhm = hhm_root / f"{record.accession}.hhm"
        raw_disorder = disorder_root / f"{record.accession}.txt"
        raw_ss_dir = ss_root / record.accession
        raw_ss = raw_ss_dir / "seq.phipsi"

        hhm = dataset_dir / "hhm" / f"{record.accession}.hhm"
        disorder = dataset_dir / "disorder" / f"{record.accession}.txt"
        ss_dir = dataset_dir / "SS" / record.accession
        ss = ss_dir / "seq.phipsi"

        hhm_available = copy_file(raw_hhm, hhm)
        disorder_available = copy_file(raw_disorder, disorder)

        ss_available = False
        if raw_ss_dir.is_dir():
            copy_tree_or_file(raw_ss_dir, ss_dir)
            ss_available = ss.is_file()
        elif raw_ss.is_file():
            ss_available = copy_file(raw_ss, ss)

        linker_available = record.accession in linker_blocks
        if linker_available:
            merged_linker_lines.extend(linker_blocks[record.accession])

        rows.append({
            "accession": record.accession,
            "label": "" if record.label is None else record.label,
            "source_group": dataset_name,
            "sequence_length": len(record.sequence),
            "hhm_raw_path": relative(hhm, repo_root),
            "hhm_available": hhm_available,
            "disorder_raw_path": relative(disorder, repo_root),
            "disorder_available": disorder_available,
            "ss_raw_path": relative(ss, repo_root),
            "ss_available": ss_available,
            "linker_available": linker_available,
            "linker_standardized_path": relative(dataset_dir / "linker.apod", repo_root),
        })

    linker_output = dataset_dir / "linker.apod"
    linker_output.write_text("\n".join(merged_linker_lines) + ("\n" if merged_linker_lines else ""), encoding="utf-8")

    frame = pd.DataFrame(rows)
    manifest_path = dataset_dir / "prediction_manifest.csv"
    frame.to_csv(manifest_path, index=False)
    return frame


def coverage_report(frame):
    return {name: int(frame[f"{name}_available"].sum()) for name in ("hhm", "disorder", "ss", "linker")}


def standardize_training_and_test(config):
    repo_root = Path(config["repo_root"]).resolve()
    raw_root = resolve_path(repo_root, config["raw_predicted_features_dir"])
    standardized_root = resolve_path(repo_root, config["standardized_predicted_features_dir"])
    train = build_labelled_dataset("train", "train", resolve_path(repo_root, config["train_fasta"]), raw_root, standardized_root, repo_root)
    test = build_labelled_dataset("test", "test", resolve_path(repo_root, config["test_fasta"]), raw_root, standardized_root, repo_root)
    report = {"train": coverage_report(train), "test": coverage_report(test)}
    standardized_root.mkdir(parents=True, exist_ok=True)
    (standardized_root / "prediction_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--mode", choices=("train-test", "prediction", "single"), default="train-test")
    parser.add_argument("--predict-fasta", default=None, help="Backward-compatible alias for --input-fasta in prediction mode.")
    parser.add_argument("--input-fasta", default=None, help="FASTA file for single/prediction mode.")
    parser.add_argument("--raw-prediction-dir", default=None, help="Raw single-dataset prediction root containing HHM/ IDR/ Linker/ SS/.")
    parser.add_argument("--standardized-prediction-dir", default=None, help="Root directory for standardized predictions.")
    parser.add_argument("--dataset-name", default="prediction")
    args = parser.parse_args()

    config = load_config(args.config, repo_root=PROJECT_ROOT)
    repo_root = Path(config["repo_root"]).resolve()
    standardized_root = (
        resolve_path(repo_root, args.standardized_prediction_dir)
        if args.standardized_prediction_dir
        else resolve_path(repo_root, config["standardized_predicted_features_dir"])
    )

    if args.mode == "single":
        if not args.input_fasta:
            raise ValueError("--input-fasta is required for --mode single")
        raw_root = (
            resolve_path(repo_root, args.raw_prediction_dir)
            if args.raw_prediction_dir
            else resolve_path(repo_root, config["raw_predicted_features_dir"])
        )
        frame = build_single_raw_dataset(
            args.dataset_name,
            resolve_path(repo_root, args.input_fasta),
            raw_root,
            standardized_root,
            repo_root,
        )
        report = {args.dataset_name: coverage_report(frame)}
        standardized_root.mkdir(parents=True, exist_ok=True)
        (standardized_root / f"{args.dataset_name}_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

    elif args.mode == "prediction":
        fasta = args.input_fasta or args.predict_fasta
        if not fasta:
            raise ValueError("--input-fasta or --predict-fasta is required for prediction mode")
        frame = build_prediction_dataset(
            args.dataset_name,
            resolve_path(repo_root, fasta),
            standardized_root,
            repo_root,
        )
#        print(f"Wrote {len(frame)} prediction manifest rows")

    else:
        print(json.dumps(standardize_training_and_test(config), indent=2))

    print("Completed!!")


if __name__ == "__main__":
    main()
