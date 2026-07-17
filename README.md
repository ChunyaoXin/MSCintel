# MSCintel

MSCintel is a sequence-based computational framework for investigating sequence-derived features associated with mechanosensitivity and identifying mechanosensitive ion channels (MSCs).

This repository provides two primary workflows:

1. **Feature selection and model construction**: calculate sequence-derived features, identify features associated with mechanosensitive ion channel function, train a Stacking classification model, and evaluate model performance.
2. **Protein prediction**: predict whether new protein sequences are likely to be mechanosensitive ion channels using either the interpretable SMOD_star model or the complementary ESM2-LR model.

## Main Modules

| Module        | Purpose                                                                                                                                         | Documentation                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `SMOD_star/`  | Calculate sequence-derived features, perform feature selection, train the Stacking model, and evaluate model performance.                       | [`SMOD_star/README.md`](SMOD_star/README.md)   |
| `prediction/` | Apply the trained prediction models to new protein sequences. This module includes SMOD_star and ESM2-LR as two complementary prediction tools. | [`prediction/README.md`](prediction/README.md) |

SMOD_star is the interpretable prediction model implemented within the MSCintel framework. ESM2-LR is provided as a complementary tool for applications that prioritize classification accuracy over mechanistic interpretability.

## Environment Setup

The workflows in this repository were tested with Python 3.8 in a conda environment. The required dependencies are listed in `environment.yml`.

From the repository root, create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate mscintel
```

Run all subsequent commands after activating this environment.

## External Prediction Requirements

The SMOD_star feature-selection and prediction workflows use external per-residue prediction results, including conservation profiles, disorder predictions, linker predictions, and secondary-structure predictions. These results must be generated using the corresponding external tools before running the relevant SMOD_star workflow.

The external prediction framework is extensible. Users may remove existing external prediction types or incorporate additional prediction sources according to their analysis requirements. When introducing a new externally derived feature, the corresponding prediction results must first be generated using the appropriate prediction tool and converted into the standardized input format required by the pipeline.

Detailed file formats, directory structures, and configuration instructions are provided in the corresponding module README files.

When using an existing trained SMOD_star model for prediction, all feature types required by that model must be provided. Any modification to the feature system requires retraining the model.

ESM2-LR can either generate ESM2 sequence embeddings from the input FASTA file or load previously generated embeddings.

