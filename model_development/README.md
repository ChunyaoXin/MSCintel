# Feature Selection and Stacking Model Pipeline

This document describes a flexible and extensible pipeline for feature selection, classification model construction, and performance evaluation. Starting from training and test feature matrices, the pipeline integrates correlation-based screening, resampling-based statistical screening, feature intersection, wrapper-based feature selection, and Stacking ensemble learning.


## 1. Required Input Files

Before running the pipeline, prepare the training and test feature CSV files. These files can be provided directly or generated using the feature calculation pipeline included in MSCintel.

### 1.1 Feature Preparation

**Option 1. Provide Precomputed Feature Files**

Users can prepare their own sequence-derived or prediction-derived feature CSV files, provided that they follow the input format and consistency requirements described in Section 1.3.

**Option 2. Generate Features according to SMOD_star**

Alternatively, users can calculate features using the `feature_generation` module provided in this repository.

The feature calculation pipeline extracts **3,019 features across eight categories**, including amino acid composition, amino acid pair composition, AAindex-based physicochemical properties, sequence conservation, disorder, flexible linkers, secondary structure, and relative solvent accessibility.

For detailed input requirements, external prediction file preparation, and execution instructions, refer to the `feature_generation/README.md` documentation.

The resulting feature CSV files can be used directly as inputs to the feature-selection pipeline.

### 1.2 Dataset Size and Feature Requirements

Insufficient training samples or too few initial features may prevent the feature-selection and model-construction pipeline from proceeding. In such cases, the pipeline may raise an error and terminate execution.

### 1.3 Training and Test Feature Files

The training and independent test feature files should be placed in the `example/train/` and `example/test/` directories, respectively.

Each feature CSV file must follow this format:

```text
AccessionName,feature_1,feature_2,...,feature_n,label
```

Requirements:

1. Each CSV file must contain `AccessionName` as the first column, `label` (binary: `0` or `1`) as the last column, and numeric feature values in between.
2. All CSV files within the same dataset must have identical accession orders and corresponding labels.
3. Training and test datasets must contain matching CSV filenames and corresponding feature columns.

**Important:** Every `.csv` file in these directories is treated as a feature file. Do not place temporary files, summary tables, comparison reports, or unrelated CSV files in these folders.

## 2. Run the Pipeline

Run the feature-selection and Stacking pipeline from the `model_development/` directory.

The `config.yaml` file controls the feature-selection parameters, cross-validation settings, output directories, and result-saving options. Users can adjust these settings according to their datasets and experimental requirements.

Input feature directories can also be specified through command-line arguments, which override the corresponding settings in `config.yaml`.

**Option 1. Use settings from `config.yaml`**

Modify the corresponding settings in `config.yaml` as needed, then run:

```bash
python pipeline.py --config config.yaml
```

**Option 2. Specify input feature directories**

```bash
python pipeline.py \
    --config config.yaml \
    --train-features-dir example/train \
    --test-features-dir example/test
```

Command-line arguments override the corresponding input directory settings in `config.yaml`. Other pipeline settings remain controlled by `config.yaml`.


## 3. Output Control

Intermediate and final output saving is controlled through the `feature_selection.save_outputs` settings in `config.yaml`.

Set an option to `false` to execute the corresponding stage without retaining its output files. The default settings can be used for normal execution.

Disabling intermediate output saving does not prevent the subsequent stages from receiving the corresponding feature-selection results, because these results are passed directly between pipeline functions.

## 4. Expected Outputs

After successful execution, the main results are saved under `outputs/`:

```text
outputs/model_metrics.csv
outputs/stacking_model/roc_curves.png
outputs/stacking_model/prc_curves.png
outputs/stacking_model/model/SMOD_union_model.joblib
outputs/stacking_model/model/SMOD_star_model.joblib
outputs/stacking_model/model/SMOD_intersection_model.joblib
```

`model_metrics.csv` summarizes the cross-validation and independent-test performance of the four base models and the three Stacking feature-set strategies.

The three `.joblib` files contain the fitted base models, meta-model, selected feature list, feature-set strategy, and prediction threshold for the corresponding Stacking model.
