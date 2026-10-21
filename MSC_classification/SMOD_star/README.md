# SMOD_star

SMOD_star is a sequence-based classification model for identifying mechanosensitive ion channels (MSCs). This module provides a pretrained Stacking model for predicting MSCs from protein sequences and corresponding external prediction results, along with a training pipeline for reproducibility.


## 1. Model Prediction

### 1.1 Required Input Files

Prediction requires an input FASTA file and its corresponding external prediction results.

**Input FASTA**

The input FASTA file (`example/input.fasta`) supports both standard and UniProt FASTA formats. Each protein must have a unique accession. Class labels are not required.

**External Prediction Results**

Prepare the corresponding conservation (HHM), disorder, flexible linker, and secondary-structure prediction results. Details of the feature calculation methods are provided in the associated manuscript.

The expected directory structure is:

```text
example/
├── input.fasta
└── predicted_features/
    ├── hhm/
    │   ├── Q5YCC5.hhm
    │   └── ...
    ├── disorder/
    │   ├── Q5YCC5.txt
    │   └── ...
    ├── SS/
    │   ├── Q5YCC5/
    │   │   └── seq.phipsi
    │   └── ...
    └── linker.apod
```

All external prediction files must correspond to the accessions in the input FASTA and follow the required formats.


### 1.2 Configuration

Prediction parameters are defined under the `predict` section of `config.yaml`:

- `input_fasta`: Input protein FASTA file.
- `feature_data_dir`: Directory containing the corresponding external prediction results.
- `output_dir`: Directory for calculated feature CSV files.
- `model`: Filename of the pretrained SMOD_star model.
- `output`: Path to the final prediction results.
- `save_calculated_features`: Whether to retain intermediate calculated feature CSV files. If set to false, temporary feature files are automatically removed after loading.


### 1.3 Run Prediction

Run the prediction pipeline from the `MSC_classification/SMOD_star/` directory. Input and output paths can be configured directly in `config.yaml` or specified through command-line arguments.

**Option 1. Use settings from `config.yaml`**

Modify the corresponding settings in `config.yaml` as needed, then run:

```bash
python SMOD_star_predict.py --config config.yaml
```

**Option 2. Specify input and output paths**

```bash
python SMOD_star_predict.py \
    --config config.yaml \
    --input-fasta example/input.fasta \
    --feature-data-dir example/predicted_features \
    --output-dir outputs/predict/features \
    --output outputs/predict/predictions.csv
```

Command-line arguments override the corresponding settings in `config.yaml`.


### 1.4 Prediction Outputs

By default, prediction results are saved to (`outputs/predict/predictions.csv`). The CSV file contains the following columns:

| Column | Description |
|---|---|
| `AccessionName` | Protein accession from the input FASTA |
| `prediction_probability` | Predicted probability of belonging to the MSC class |
| `predicted_label` | Predicted class (1: MSC; 0: non-MSC), determined by the model's classification threshold.|


## 2. Model Training and Reproduction (Optional)

For reproducibility, the repository also includes the training pipeline used to perform feature calculation, feature selection, Stacking model construction, and evaluation.

**Run the Training Pipeline**

```bash
python SMOD_star_train.py --config config.yaml
```

The corresponding training parameters and output paths are defined in the `train` section of `config.yaml`.

## 3. Notes

- The pretrained SMOD_star model uses a fixed set of selected features; no additional feature selection is required for prediction.
- For methodological details of feature calculation, feature selection, and model construction, refer to the associated manuscript and the `model_development/` module.