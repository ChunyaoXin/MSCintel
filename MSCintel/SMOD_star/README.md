# SMOD_star Pipeline

This directory contains the complete SMOD_star workflow for feature calculation, feature selection, and Stacking model construction. Starting from training and test FASTA files and their corresponding external prediction results, the pipeline calculates sequence-derived features, identifies features associated with mechanosensitive ion channel function, trains the Stacking model, and reports the selected feature set and model performance. The workflow automatically executes feature calculation followed by feature selection and Stacking model construction.

Before running this workflow, create and activate the conda environment as described in the repository-level [`README.md`](../../README.md).

## 1. Configuration

The pipeline is configured through `config.yaml`.

Before running the workflow, verify the paths to the training and test FASTA files, the raw or standardized external prediction directories, and the output locations.

The default statistical, resampling, cross-validation, and random-seed settings can usually be retained unless a different analysis design is required.

Brief descriptions of the configuration groups are provided directly in `config.yaml`.


## 2. Required Input Files

The full pipeline requires two types of input files:

1. Labeled training and independent test FASTA files.
2. External per-residue prediction results.

By default, the pipeline uses HHM-based conservation profiles, disorder predictions, linker predictions, and secondary-structure predictions. Existing prediction types may be excluded, and additional externally derived feature types may be incorporated if they are converted into the required standardized format.

Both raw input files and already standardized input files are supported.

For detailed FASTA formats, external prediction file organization, and feature-calculation requirements, see [`feature_generation/README.md`](feature_generation/README.md).

For statistical screening, wrapper-based feature selection, and Stacking model construction, see [`feature_selection/README.md`](feature_selection/README.md).


## 3. Run the Full Pipeline

From the repository root, enter the `SMOD_star/` directory:

```bash
cd SMOD_star
```
Run the pipeline:

```bash
python run_full_pipeline.py --config config.yaml
```

Alternatively, run the pipeline in the background using `nohup`:

```bash
nohup python run_full_pipeline.py --config config.yaml > full_pipeline.log 2>&1 &
```

The results will be saved to the output paths specified in `config.yaml`, for example:

```text
outputs/feature_selection/
```

## 4. Optional Settings

### 4.1 Allow Missing External Prediction Results

If some expected external prediction files are unavailable, add `--allow-missing`:

```bash
python run_full_pipeline.py --config config.yaml --allow-missing
```
Use this option only when missing external prediction inputs are expected and can be handled appropriately by the downstream analysis.

### 4.2 Save Intermediate Feature-Selection Results

The pipeline can save intermediate result files generated during the feature-selection process. Whether these files are saved is controlled by the corresponding settings in `config.yaml`.


