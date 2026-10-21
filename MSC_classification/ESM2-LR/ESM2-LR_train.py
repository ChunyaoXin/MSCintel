#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate ESM2-LR directly from labelled train/test FASTA files.

Usage:
    python ESM2-LR_train.py --config config.yaml
"""

import argparse
import re
from pathlib import Path
import yaml

import joblib
import numpy as np
import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LAYER = 33
DIM = 1280

def load_config(config_path):
    """Load YAML configuration and resolve paths relative to repo_root."""

    config_path = Path(config_path).expanduser().resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_root = Path(config.get("repo_root", "."))
    if not repo_root.is_absolute():
        repo_root = (config_path.parent / repo_root).resolve()

    embedding_config = config["train"]["embedding"]
    training_config = config["train"]["model_training"]

    def resolve_path(path):
        if path is None:
            return None
        path = Path(path).expanduser()
        return path if path.is_absolute() else repo_root / path

    return {
        "train_fasta": resolve_path(embedding_config["train_fasta"]),
        "test_fasta": resolve_path(embedding_config["test_fasta"]),
        "esm_model_path": resolve_path(embedding_config.get("esm_model_path")),
        "train_embedding_dir": resolve_path(embedding_config["train_embedding_dir"]),
        "test_embedding_dir": resolve_path(embedding_config["test_embedding_dir"]),
        "save_embeddings": embedding_config.get("save_embeddings", True),
        "output_dir": resolve_path(training_config["output_dir"]),
    }


def parse_labelled_fasta(path):
    """Return accession -> (sequence, binary label), preserving FASTA order."""
    entries = {}
    current_header = None
    seq_parts = []

    def add_record(header, parts):
        if header is None:
            return
        accession = header.split()[0].split('|')[0]
        if not accession or not re.fullmatch(r'[\w.-]+', accession):
            raise ValueError(f'Invalid accession in {path}: {header}')
        match = re.search(r'(?:^|[\s|])label=([01])(?:$|[\s|])', header)
        if not match:
            raise ValueError(f'Missing binary label=0/1 in {path}: {header}')
        sequence = ''.join(parts).upper()
        if not sequence:
            raise ValueError(f'Empty sequence for {accession} in {path}')
        if accession in entries:
            raise ValueError(f'Duplicate accession {accession} in {path}')
        entries[accession] = (sequence, int(match.group(1)))

    with open(path, encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                add_record(current_header, seq_parts)
                current_header = line[1:].strip()
                seq_parts = []
            elif current_header is None:
                raise ValueError(f'Sequence before FASTA header in {path}')
            else:
                seq_parts.append(line)
    add_record(current_header, seq_parts)
    if not entries:
        raise ValueError(f'No FASTA records in {path}')
    return entries


def mean_pool_token_embedding(array, sequence_length, source):
    """Discard CLS/EOS and average residue vectors, like the original scripts."""
    expected = (sequence_length + 2, DIM)
    if array.shape != expected:
        raise ValueError(
            f'Incompatible ESM2 embedding at {source}: shape {array.shape}; '
            f'expected {expected}. Delete stale cache or use the correct ESM2 model.'
        )
    return array[1:-1].mean(axis=0)


def load_esm2(checkpoint, device):
    if checkpoint is None:
        print('Loading ESM2-t33-650M (automatic download if not cached)...')
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
    else:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'ESM2 checkpoint not found: {path}')
        print(f'Loading ESM2 from: {path}')
        model, alphabet = pretrained.load_model_and_alphabet(str(path))
    model.eval().to(device)
    return model, alphabet


def obtain_features(records, cache_dir, save_embeddings, model_provider,
                    toks_per_batch):
    """Use cached token-level .npy files; compute missing ones only."""
    accessions = list(records)
    features = {}
    missing = []
    for accession in accessions:
        seq, _ = records[accession]
        filepath = cache_dir / f'{accession}.npy' if cache_dir else None
        if filepath is not None and filepath.is_file():
            embedding = np.load(filepath, allow_pickle=False)
            features[accession] = mean_pool_token_embedding(embedding, len(seq), filepath)
        else:
            missing.append(accession)

    print(f'Cached sequences: {len(features)}; embeddings to generate: {len(missing)}')
    if missing:
        model, alphabet, device = model_provider()
        # Preserve the batching and tokenization behavior of the original script.
        dataset = FastaBatchedDataset([accession for accession in missing], [records[accession][0] for accession in missing])
        batch_indices = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batch_indices)
        if save_embeddings and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for batch_number, (labels, sequences, tokens) in enumerate(loader, start=1):
                print(f'  ESM2 batch {batch_number}/{len(batch_indices)}')
                valid_lengths = (tokens != alphabet.padding_idx).sum(dim=1).tolist()
                output = model(tokens.to(device), repr_layers=[LAYER], return_contacts=False)
                reps = output['representations'][LAYER].cpu().numpy()
                for i, accession in enumerate(labels):
                    array = reps[i, :valid_lengths[i]]
                    features[accession] = mean_pool_token_embedding(array, len(records[accession][0]), accession)
                    if save_embeddings and cache_dir is not None:
                        np.save(cache_dir / f'{accession}.npy', array)

    X = np.stack([features[acc] for acc in accessions])
    y = np.asarray([records[acc][1] for acc in accessions], dtype=int)
    return X, y, accessions


def main():
    # ============================================================
    # 1. Load configuration
    # ============================================================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    train_fasta = config['train_fasta']
    test_fasta = config['test_fasta']
    output_dir = config['output_dir']
    train_cache = config['train_embedding_dir']
    test_cache = config['test_embedding_dir']
    esm_model_path = config['esm_model_path']
    save_embeddings = config['save_embeddings']

    # Fixed ESM2 parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    toks_per_batch = 4096

    if not isinstance(save_embeddings, bool):
        raise ValueError('save_embeddings must be true or false')

    # ============================================================
    # 2. Load and validate FASTA datasets
    # ============================================================
    train_records = parse_labelled_fasta(train_fasta)
    test_records = parse_labelled_fasta(test_fasta)

    overlap = set(train_records) & set(test_records)
    if overlap:
        raise ValueError(f'Train/test accession overlap (examples): {sorted(overlap)[:5]}')

    if set(label for _, label in train_records.values()) != {0, 1}:
        raise ValueError('Training FASTA must contain both label=0 and label=1')

    if set(label for _, label in test_records.values()) != {0, 1}:
        raise ValueError('Test FASTA must contain both label=0 and label=1 to compute AUC')

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}')

    # ============================================================
    # 3. Initialize ESM2 model (lazy loading)
    # ============================================================
    holder = {}

    def model_provider():
        if 'model' not in holder:
            holder['model'], holder['alphabet'] = load_esm2(esm_model_path, device)
        return holder['model'], holder['alphabet'], device

    # ============================================================
    # 4. Generate or load ESM2 embeddings
    # ============================================================
    print('Preparing training embeddings...')
    X_train, y_train, _ = obtain_features(train_records, train_cache, save_embeddings, model_provider, toks_per_batch)

    print('Preparing test embeddings...')
    X_test, y_test, test_ids = obtain_features(test_records, test_cache, save_embeddings, model_provider, toks_per_batch)

    # Release GPU memory before training the CPU LR model.
    holder.clear()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ============================================================
    # 5. Train Logistic Regression model
    # ============================================================
    print(f'Train: {X_train.shape}; test: {X_test.shape}')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=10000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    model_path = output_dir / 'esm2_lr_model.joblib'
    joblib.dump(pipeline, model_path)

    # ============================================================
    # 6. Evaluate model
    # ============================================================
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    auprc = average_precision_score(y_test, probabilities)

    pd.DataFrame({'Model': ['ESM2-LR'], 'AUC': [auc], 'AUPRC': [auprc]}).to_csv(output_dir / 'esm2_lr_test_metrics.csv', index=False)
    pd.DataFrame({'AccessionName': test_ids, 'label': y_test, 'prediction_probability': probabilities}).to_csv(output_dir / 'esm2_lr_test_predictions.csv', index=False)

    print(f'Test AUC: {auc:.4f}; AUPRC: {auprc:.4f}')
    print(f'Saved model and results in: {output_dir}')



if __name__ == '__main__':
    main()
