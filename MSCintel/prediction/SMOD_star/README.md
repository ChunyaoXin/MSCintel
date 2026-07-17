# SMOD_star Prediction Pipeline

This document describes the application-stage SMOD_star prediction pipeline, which calculates the required sequence features from an input FASTA file and its corresponding external prediction results, loads a trained Stacking model, and outputs the predicted probability for each protein.

## 1. Required Input Files

Before running prediction, prepare the input FASTA file, the corresponding external prediction results, and a trained Stacking model.

### 1.1 Prediction FASTA

Place the input FASTA file under:

```text
Data/input.fasta
```

The FASTA file should contain one or more protein sequences without class labels, for example:

```text
>AccessionName
SEQUENCE
```

Each accession should be unique within the input dataset.

### 1.2 External Prediction Results

Prepare the external prediction results required by the model, including HHM-based conservation profiles, disorder predictions, linker predictions, and secondary-structure predictions.

The required directory structure and file formats should match those used during model training. The corresponding paths are specified in `config.yaml`.

### 1.3 Trained Stacking Model

A trained Stacking model file is required for prediction. The model path is specified in `config.yaml`.

The model file must contain the fitted base models, meta-model, selected feature list, and prediction threshold used by the prediction pipeline.

## 2. Configuration

Before running prediction, verify the following settings in `config.yaml`:

* input FASTA path;
* external prediction input directory;
* trained model path;
* prediction output path;
* whether calculated feature files should be retained.

Brief descriptions of the configuration groups are provided directly in `config.yaml`.

## 3. Run Prediction

From the repository root, enter the `SMOD_star/` prediction directory:

```bash
cd prediction/SMOD_star
```

Run prediction:

```bash
python -u SMOD_star_predict.py --config config.yaml
```

Alternatively, run the pipeline in the background using `nohup`:

```bash
nohup python -u SMOD_star_predict.py --config config.yaml > prediction.log 2>&1 &
```

## 4. Optional Settings

### Save Calculated Feature Values

By default, intermediate calculated feature files are not retained. To save them, add:

```bash
python -u SMOD_star_predict.py --config config.yaml --save-calculated-features true
```

The same option can be appended to the `nohup` command when running in the background.

## 5. Expected Outputs

Prediction results are saved to the output path specified in `config.yaml`.

The output file contains one row per input protein, including the accession and the predicted probability of being a mechanosensitive ion channel.
