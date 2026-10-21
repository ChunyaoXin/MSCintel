# Feature Calculation Pipeline

This document describes the feature calculation stage of MSCintel. The pipeline calculates eight categories of sequence-derived and prediction-derived features from labeled protein FASTA files and precomputed external prediction results. Each feature category is saved as a separate CSV file for downstream feature selection.


## 1. Required Input Files

### 1.1 FASTA Input

Prepare the following labeled FASTA files:

```text
Data/train.fasta
Data/test.fasta
```

Each header must contain a unique accession and a binary classification label:

```fasta
>Protein001|label=1
MAVLS...
>Protein002|label=0
MKTLL...
```

`label=1` denotes a mechanosensitive ion channel (MSC), and `label=0` denotes a non-MSC protein. Accessions must be unique within each FASTA file. This pipeline expects already labeled FASTA files.

### 1.2 External Prediction Results

Standardized external predictions are required for conservation, disorder, flexible linker, and secondary-structure-related feature calculations. Prepare the following structure:

```text
predicted_features/
├── train/
│   ├── hhm/
│   │   └── Protein001.hhm
│   ├── disorder/
│   │   └── Protein001.txt
│   ├── SS/
│   │   └── Protein001/
│   │       └── seq.phipsi
│   └── linker.apod
└── test/
    ├── hhm/
    ├── disorder/
    ├── SS/
    └── linker.apod
```

Replace `Protein001` with each corresponding FASTA accession. The `linker.apod` file must contain records with headers matching those accessions. All four external prediction types must be available for each processed protein.


## 2. Run Feature Calculation

Run the feature_generation pipeline from the `model_development/feature_generation/` directory.

```bash
python FeatureGen.py \
    --train-fasta Data/train.fasta \
    --test-fasta Data/test.fasta \
    --prediction-dir predicted_features \
    --output-dir ../example
```

## 3. Expected Outputs

After successful execution, the output files are organized as follows:

```text
../example/
├── train/
│   ├── AA_frequency_features.csv
│   ├── AA_pairs_features.csv
│   ├── AAindex_features.csv
│   ├── Conservation_features.csv
│   ├── disorder_features.csv
│   ├── linker_features.csv
│   ├── SS_features.csv
│   └── SS_RSA_features.csv
└── test/
    ├── AA_frequency_features.csv
    ├── AA_pairs_features.csv
    ├── AAindex_features.csv
    ├── Conservation_features.csv
    ├── disorder_features.csv
    ├── linker_features.csv
    ├── SS_features.csv
    └── SS_RSA_features.csv
```

Each CSV contains protein accessions in the `AccessionName` column, calculated feature columns, and the binary `label` column. Each category is retained as its own file.

The eight feature CSV files in each dataset directory can serve as inputs to the downstream feature selection pipeline.
