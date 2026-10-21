# -*- coding: utf-8 -*-

"""
Predict mechanosensitive ion channels using the pretrained ESM2-LR model.

Usage:
    python ESM2-LR_predict.py \
        --config config.yaml \
        --input example/input.fasta \
        --output outputs/predict/predictions.csv
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from esm import FastaBatchedDataset, pretrained


ESM_LAYER = 33
ESM_EMBEDDING_DIM = 1280
MAX_ESM2_SEQUENCE_LENGTH = 1022

    
def safe_accession(label: str) -> str:
    header = label.strip().split()[0]
    fields = header.split("|")

    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        accession = fields[1]
    else:
        accession = fields[0]

    accession = re.sub(r"[^\w.\-]", "_", accession)

    if not accession:
        raise ValueError(f"Could not extract an accession from FASTA header: {label}")

    return accession


def read_fasta_headers(fasta_file: str) -> List[Tuple[str, str]]:

    records: List[Tuple[str, str]] = []

    with open(fasta_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line.startswith(">"):
                header = line[1:].strip()

                if not header:
                    raise ValueError("An empty FASTA header was detected.")

                records.append((header, safe_accession(header)))

    if not records:
        raise ValueError(f"No FASTA records were found in: {fasta_file}")

    headers = [header for header, _ in records]
    accessions = [accession for _, accession in records]

    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate FASTA headers were detected. Each sequence must have a unique header.")

    if len(accessions) != len(set(accessions)):
        raise ValueError("Duplicate accession names were detected after parsing FASTA headers. The text before the first '|' must be unique for each sequence.")

    return records


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-compatible device is available.")

    return torch.device(device_name)


def load_esm2_model(device: torch.device, esm_model_path: Optional[str] = None):

    if esm_model_path is None:
        print("[INFO] Loading official ESM2 t33 650M model...")
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
    else:
        if not os.path.isfile(esm_model_path):
            raise FileNotFoundError(f"ESM2 model checkpoint was not found: {esm_model_path}")

        print(f"[INFO] Loading ESM2 checkpoint: {esm_model_path}")
        model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)

    model.eval()
    model = model.to(device)

    return model, alphabet


def generate_sequence_embeddings(
    fasta_file: str,
    device: torch.device,
    esm_model_path: Optional[str] = None,
    toks_per_batch: int = 4096,
    save_embeddings: bool = False,
    embedding_output_dir: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:

    original_records = read_fasta_headers(fasta_file)
    original_headers = [header for header, _ in original_records]
    original_accessions = [accession for _, accession in original_records]

    model, alphabet = load_esm2_model(device=device, esm_model_path=esm_model_path)

    dataset = FastaBatchedDataset.from_file(fasta_file)

    if len(dataset) != len(original_records):
        raise ValueError("The number of FASTA records parsed by ESM does not match the number of detected FASTA headers.")

    batches = dataset.get_batch_indices(toks_per_batch=toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)

    print(f"[INFO] Number of sequences: {len(dataset)}")
    print(f"[INFO] Number of batches: {len(batches)}")
    print(f"[INFO] Device: {device}")

    # Store embeddings by the complete FASTA header so that predictions
    # can subsequently be restored to the original FASTA order.
    embedding_by_header: Dict[str, np.ndarray] = {}
    
    # Create embedding output directory if saving is enabled.
    if save_embeddings:
        if not embedding_output_dir:
            raise ValueError("embedding_output_dir is required when save_embeddings is true.")
        embedding_output_path = Path(embedding_output_dir)
        embedding_output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_index, (labels, sequences, tokens) in enumerate(data_loader, start=1,):
            print(f"[INFO] Processing batch {batch_index}/{len(batches)} ({len(labels)} sequences)")

            for label, sequence in zip(labels, sequences):
                sequence_length = len(sequence)

                if sequence_length == 0:
                    raise ValueError(f"An empty protein sequence was detected: {label}")

                if sequence_length > MAX_ESM2_SEQUENCE_LENGTH:
                    raise ValueError(
                        f"Sequence '{label}' has {sequence_length} residues. ESM2 t33 supports at most {MAX_ESM2_SEQUENCE_LENGTH} residues per sequence. The sequence was not truncated automatically.")

            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)

            representations = results["representations"][ESM_LAYER]

            for index, (label, sequence) in enumerate(zip(labels, sequences)):
                sequence_length = len(sequence)

                # Save token-level embeddings, including <cls> and <eos>.
                if save_embeddings:
                    token_embedding = representations[index, :sequence_length + 2, :]
                    token_embedding = token_embedding.detach().cpu().numpy().astype(np.float32)
                    np.save(embedding_output_path / f"{safe_accession(label)}.npy", token_embedding)

                # Mean pooling for LR classification (unchanged).
                residue_representation = representations[index, 1:sequence_length + 1, :]
                sequence_embedding = residue_representation.mean(dim=0).detach().cpu().numpy().astype(np.float32)

                if sequence_embedding.shape != (ESM_EMBEDDING_DIM,):
                    raise ValueError(f"Unexpected embedding shape for '{label}': {sequence_embedding.shape}; expected ({ESM_EMBEDDING_DIM},).")

                if not np.isfinite(sequence_embedding).all():
                    raise ValueError(f"Non-finite values were detected in the embedding for sequence '{label}'.")

                embedding_by_header[label] = sequence_embedding

            # Release batch-level GPU memory.
            del tokens, results, representations

            if device.type == "cuda":
                torch.cuda.empty_cache()

    missing_headers = [header for header in original_headers if header not in embedding_by_header]

    if missing_headers:
        raise ValueError("Embeddings were not generated for the following FASTA headers: " + ", ".join(missing_headers[:10]))

    # Restore the original FASTA order.
    feature_matrix = np.stack([embedding_by_header[header] for header in original_headers], axis=0)

    if feature_matrix.ndim != 2:
        raise ValueError(f"Invalid feature matrix shape: {feature_matrix.shape}")

    if feature_matrix.shape[1] != ESM_EMBEDDING_DIM:
        raise ValueError(f"Invalid embedding dimension: {feature_matrix.shape[1]}; expected {ESM_EMBEDDING_DIM}.")

    if save_embeddings:
        print(f"[INFO] Saved {len(original_accessions)} token-level embeddings to: {embedding_output_path}")

    return original_accessions, feature_matrix

def load_existing_embeddings(fasta_file: str, embedding_dir: str) -> Tuple[List[str], np.ndarray]:

    if not os.path.isdir(embedding_dir):
        raise NotADirectoryError(f"Embedding directory was not found: {embedding_dir}")

    records = read_fasta_headers(fasta_file)

    accessions = []
    sequence_embeddings = []

    for _, accession in records:
        npy_file = os.path.join(embedding_dir, accession + ".npy")

        if not os.path.isfile(npy_file):
            raise FileNotFoundError(f"Embedding file was not found for '{accession}': {npy_file}")

        embedding = np.load(npy_file)

        # Format 1: already mean-pooled sequence embedding.
        if embedding.ndim == 1:
            if embedding.shape != (ESM_EMBEDDING_DIM,):
                raise ValueError(f"Invalid embedding shape in '{npy_file}': {embedding.shape}; expected ({ESM_EMBEDDING_DIM},).")

            sequence_embedding = embedding

        # Format 2: token-level embedding containing <cls> and <eos>.
        elif embedding.ndim == 2:
            if embedding.shape[1] != ESM_EMBEDDING_DIM:
                raise ValueError(f"Invalid embedding dimension in '{npy_file}': {embedding.shape}; expected (*, {ESM_EMBEDDING_DIM}).")

            if embedding.shape[0] < 3:
                raise ValueError(f"Token-level embedding in '{npy_file}' must contain <cls>, at least one residue, and <eos>.")

            # Remove <cls> and <eos>, then perform mean pooling.
            residue_embedding = embedding[1:-1]
            sequence_embedding = residue_embedding.mean(axis=0)

        else:
            raise ValueError(f"Unsupported embedding shape in '{npy_file}': {embedding.shape}")

        sequence_embedding = np.asarray(sequence_embedding, dtype=np.float32)

        if not np.isfinite(sequence_embedding).all():
            raise ValueError(f"Non-finite values were detected in: {npy_file}")

        accessions.append(accession)
        sequence_embeddings.append(sequence_embedding)

    feature_matrix = np.stack(sequence_embeddings, axis=0)

    if feature_matrix.shape != (len(accessions), ESM_EMBEDDING_DIM,):
        raise ValueError(f"Invalid feature matrix shape: {feature_matrix.shape}; expected ({len(accessions)}, {ESM_EMBEDDING_DIM}).")

    print(f"[INFO] Loaded {len(accessions)} existing embeddings from: {embedding_dir}")

    return accessions, feature_matrix


def load_classifier(model_file: str):
    """Load the trained StandardScaler + LogisticRegression pipeline."""
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"ESM2-LR model file was not found: {model_file}")

    classifier = joblib.load(model_file)

    if not hasattr(classifier, "predict_proba"):
        raise TypeError("The loaded model does not provide predict_proba(). Please check esm2_lr_model.joblib.")

    if not hasattr(classifier, "classes_"):
        raise TypeError("The loaded model does not contain class information.")

    return classifier


def get_positive_class_probabilities(
    classifier,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Return probabilities corresponding to class 1."""
    probabilities = classifier.predict_proba(feature_matrix)
    classes = np.asarray(classifier.classes_)

    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        raise ValueError(f"Could not identify class 1 in model classes: {classes.tolist()}")

    positive_probabilities = probabilities[:, positive_indices[0]]

    return positive_probabilities


def predict_esm2_lr(
    fasta_file: str,
    model_file: str,
    output_file: str,
    embedding_dir: Optional[str] = None,
    esm_model_path: Optional[str] = None,
    toks_per_batch: int = 4096,
    device_name: str = "auto",
    threshold: float = 0.5,
    save_embeddings: bool = False,
    embedding_output_dir: Optional[str] = None,
) -> None:
    """Run the complete ESM2-LR prediction workflow."""
    if not os.path.isfile(fasta_file):
        raise FileNotFoundError(f"Input FASTA file was not found: {fasta_file}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if toks_per_batch <= 0:
        raise ValueError("--toks-per-batch must be greater than 0.")

    if embedding_dir is not None:
        print("[INFO] Existing embedding directory was provided.")
        print("[INFO] Skipping ESM2 model loading and embedding generation.")
        accessions, feature_matrix = load_existing_embeddings(fasta_file=fasta_file, embedding_dir=embedding_dir,)
    
    else:
        print("[INFO] No existing embedding directory was provided.")
        print("[INFO] Generating embeddings using ESM2.")
    
        device = resolve_device(device_name)

        accessions, feature_matrix = generate_sequence_embeddings(
            fasta_file=fasta_file,
            device=device,
            esm_model_path=esm_model_path,
            toks_per_batch=toks_per_batch,
            save_embeddings=save_embeddings,
            embedding_output_dir=embedding_output_dir,
        )

    print(f"[INFO] Loading trained classifier: {model_file}")
    classifier = load_classifier(model_file)

    print("[INFO] Generating prediction probabilities...")
    prediction_probability = get_positive_class_probabilities(classifier=classifier, feature_matrix=feature_matrix)

    predicted_label = (prediction_probability >= threshold).astype(int)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({"AccessionName": accessions, "prediction_probability": prediction_probability, "predicted_label": predicted_label})

    result_df.to_csv(output_path, index=False)

    print(f"[INFO] Prediction threshold: {threshold}")
    print(f"[INFO] Saved predictions: {output_path}")
    print("Completed!")


def load_yaml_config(config_file):
    """Load the unified TRAIN and PREDICT configuration."""
    config_path = Path(config_file).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")

    train_cfg = config["train"]
    embedding_cfg = config["predict"]["embedding"]
    prediction_cfg = config["predict"]["prediction"]

    repo_root = Path(config.get("repo_root", ".")).expanduser()
    repo_root = repo_root.resolve() if repo_root.is_absolute() else (config_path.parent / repo_root).resolve()

    return train_cfg, embedding_cfg, prediction_cfg, repo_root


def parse_arguments():
    """Parse command-line arguments and merge them with YAML configuration."""
    parser = argparse.ArgumentParser(description="Predict mechanosensitive ion channels using ESM2-LR.")

    parser.add_argument("--config", default="config.yaml", help="Path to the unified YAML configuration file.")
    parser.add_argument("--input", default=None, help="Input protein FASTA file.")
    parser.add_argument("--model", default=None, help="Path to the trained ESM2-LR model.")
    parser.add_argument("--embedding-dir", default=None, help="Directory containing existing .npy embeddings.")
    parser.add_argument("--output", default=None, help="Output prediction CSV file.")
    parser.add_argument("--esm-model-path", default=None, help="Optional local ESM2 checkpoint.")
    parser.add_argument("--toks-per-batch", type=int, default=None, help="Maximum tokens per batch.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Computation device.")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold.")
    parser.add_argument("--embedding-output-dir", default=None, help="Directory for newly generated embeddings.")

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-embeddings", dest="save_embeddings", action="store_true", help="Save newly generated embeddings.")
    save_group.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false", help="Do not save newly generated embeddings.")
    parser.set_defaults(save_embeddings=None)

    cli_args = parser.parse_args()
    train_cfg, embedding_cfg, prediction_cfg, repo_root = load_yaml_config(cli_args.config)

    def choose(cli_value, config_value, default=None):
        return cli_value if cli_value is not None else (config_value if config_value is not None else default)

    def resolve(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (repo_root / path).resolve())

    input_fasta = choose(cli_args.input, embedding_cfg.get("input_fasta"))
    embedding_dir = choose(cli_args.embedding_dir, embedding_cfg.get("embedding_dir"))
    esm_model_path = choose(cli_args.esm_model_path, embedding_cfg.get("esm_model_path"))
    save_embeddings = choose(cli_args.save_embeddings, embedding_cfg.get("save_embeddings"), False)
    embedding_output_dir = choose(cli_args.embedding_output_dir, embedding_cfg.get("embedding_output_dir"))

    train_output_dir = train_cfg["model_training"]["output_dir"]
    default_model_path = Path(train_output_dir) / "esm2_lr_model.joblib"

    model_file = choose(cli_args.model, prediction_cfg.get("model"), default_model_path)
    output_file = choose(cli_args.output, prediction_cfg.get("output"))
    toks_per_batch = choose(cli_args.toks_per_batch, prediction_cfg.get("toks_per_batch"), 4096)
    device = choose(cli_args.device, prediction_cfg.get("device"), "auto")
    threshold = choose(cli_args.threshold, prediction_cfg.get("threshold"), 0.5)

    if not input_fasta:
        parser.error("Input FASTA is required. Set predict.embedding.input_fasta or provide --input.")

    if not output_file:
        parser.error("Prediction output path is required. Set predict.prediction.output or provide --output.")

    if not isinstance(save_embeddings, bool):
        parser.error("save_embeddings must be true or false.")

    if not 0.0 <= float(threshold) <= 1.0:
        parser.error("threshold must be between 0 and 1.")

    if save_embeddings and embedding_dir is None and embedding_output_dir is None:
        parser.error("embedding_output_dir is required when generating and saving embeddings.")

    return argparse.Namespace(
        config=cli_args.config,
        input=resolve(input_fasta),
        model=resolve(model_file),
        embedding_dir=resolve(embedding_dir),
        output=resolve(output_file),
        esm_model_path=resolve(esm_model_path),
        toks_per_batch=int(toks_per_batch),
        device=device,
        threshold=float(threshold),
        save_embeddings=save_embeddings,
        embedding_output_dir=resolve(embedding_output_dir),
    )


def main():
    args = parse_arguments()

    predict_esm2_lr(
        fasta_file=args.input,
        model_file=args.model,
        output_file=args.output,
        embedding_dir=args.embedding_dir,
        esm_model_path=args.esm_model_path,
        toks_per_batch=args.toks_per_batch,
        device_name=args.device,
        threshold=args.threshold,
        save_embeddings=args.save_embeddings,
        embedding_output_dir=args.embedding_output_dir,
    )


if __name__ == "__main__":
    main()