# ESM2-LR Prediction Pipeline

This document describes the ESM2-LR prediction pipeline, which generates or loads ESM2 sequence embeddings, applies a trained Logistic Regression model, and outputs prediction probabilities.

## 1. Required Files

Prepare the input FASTA file:

```text
Data/input.fasta
```

The FASTA file should contain one or more protein sequences without class labels:

```text
>AccessionName
SEQUENCE
```

Each accession should be unique.

Existing per-sequence ESM2 embedding files may optionally be provided as `.npy` files. Each filename should match the corresponding accession:

```text
outputs/embeddings/AccessionName.npy
```

## 2. Configuration

Set the following paths and options in `config.yaml`:

- input FASTA file;
- prediction output file;
- optional directory containing existing `.npy` embeddings;
- optional local ESM2 checkpoint;
- whether newly generated sequence embeddings should be saved.

If `embedding_dir` is provided, the pipeline loads the existing `.npy` files and skips ESM2 model loading and embedding generation.

If `embedding_dir` is set to `null`, embeddings are generated from the input FASTA file. When no local checkpoint is specified, the pretrained `esm2_t33_650M_UR50D` model is downloaded automatically.

## 3. Run Prediction

Enter the prediction directory:

```bash
cd prediction/ESM2_LR
```

Run:

```bash
python -u ESM2_LR_predict.py --config config.yaml
```

Run in the background:

```bash
nohup python -u ESM2_LR_predict.py --config config.yaml > prediction.log 2>&1 &
```

## 4. Output

Prediction results are saved to the path specified in `config.yaml`.

The output file contains the accession, prediction probability, and predicted class for each input protein.