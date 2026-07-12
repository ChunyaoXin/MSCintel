# Feature Calculation Pipeline

This document describes the feature-calculation stage of the SMOD_star pipeline, which converts training and test FASTA files together with their corresponding external prediction results into feature-value CSV files.

## 1. Required Input Files

Before running the pipeline, prepare the FASTA files and the external prediction result files.

### 1.1 FASTA Input

There are two supported input formats.

#### Option 1. Separate Positive and Negative FASTA Files

Prepare the following four files in the `Data/` directory:

```text
Data/positive_train_data.fasta
Data/negative_train_data.fasta
Data/positive_test_data.fasta
Data/negative_test_data.fasta
```

In this case, the pipeline will first run `merge_fasta.py` and generate:

```text
Data/train.fasta
Data/test.fasta
```

Positive sequences will be assigned `label=1`, and negative sequences will be assigned `label=0`.

#### Option 2. Merged Labeled FASTA Files

Alternatively, prepare the following two files directly:

```text
Data/train.fasta
Data/test.fasta
```

The FASTA headers should contain label information, for example:

```text
>AccessionName|label=1
SEQUENCE
```

Each accession should be unique within a dataset, and the label must be either `0` or `1`.

When `Data/train.fasta` and `Data/test.fasta` already exist and are in the correct labeled format, `merge_fasta.py` will skip regeneration.

### 1.2 External Prediction Result Files

There are also two supported formats for external prediction results.

#### Option 1. Raw Prediction Results

Place raw HHM-based conservation profiles, disorder predictions, linker predictions, and secondary-structure predictions under:


```text
predicted_features/
```

The pipeline will run `standardize_predictions.py` to standardize these files into:

```text
predicted_features/standardized/train/
predicted_features/standardized/test/
```

#### Option 2. Already Standardized External Prediction Results

Alternatively, prepare the standardized prediction results directly:

```text
predicted_features/standardized/train/
predicted_features/standardized/test/
```

For example, each standardized dataset directory should contain:

```text
predicted_features/standardized/train/
├── prediction_manifest.csv
├── linker.apod
├── hhm/
├── disorder/
└── SS/
```
When the required standardized files are complete and the accession list in `prediction_manifest.csv` exactly matches the corresponding FASTA file, `standardize_predictions.py` will skip regeneration.

## 2. Run Feature Calculation

From the repository root, enter the `SMOD_star/` directory:

```bash
cd SMOD_star/
```
Run the pipeline directly:

```bash
python feature_generation/FeatureGen.py --config config.yaml
```
Alternatively, run the pipeline in the background using `nohup`:

```bash
nohup python feature_generation/FeatureGen.py --config config.yaml > FeatureGen.log 2>&1 &
```
## 3. Optional Settings

### Allow Missing External Prediction Results

This option passes `--allow-missing` to the feature calculation scripts:

```bash
python feature_generation/FeatureGen.py --config config.yaml --allow-missing
```

Use `--allow-missing` only when some expected external prediction files are unavailable and the downstream analysis is designed to tolerate incomplete inputs.

## 4. Expected Outputs

After running `FeatureGen.py`, the calculated feature files will be saved under:

```text
outputs/features/train/
outputs/features/test/
```
Each feature category will be saved as a separate CSV file. Each output CSV contains one row per protein. The first column is `AccessionName`, the final column is `label`, and the intermediate columns contain calculated feature values.
