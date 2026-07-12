# MSCintel

MSCintel is a sequence-based computational framework for investigating the sequence-derived features associated with mechanosensitivity and identifying mechanosensitive ion channels.

This repository provides two primary workflows:

1. **Feature selection and model construction**: identify sequence-derived features associated with mechanosensitive ion channel function and train a Stacking classification model.
2. **Protein prediction**: predict whether new protein sequences are likely to be mechanosensitive ion channels using a trained Stacking model.

## Main Modules

| Module | Purpose | Documentation |
|---|---|---|
| `SMOD_star/` | Calculate sequence-derived features, perform feature selection, train the Stacking model, and evaluate model performance. | [SMOD_star/README.md](MSCintel/SMOD_star/README.md) |
| `SMOD_star_predict/` | Calculate the required features for new sequences and predict mechanosensitive ion channels using the trained Stacking model. | [SMOD_star_predict/README.md](MSCintel/SMOD_star_predict/README.md) |

## Environment Setup

The workflows in this repository were tested with Python 3.8 in a conda
environment. The required dependencies are listed in `environment.yml`.

From the repository root, create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate mscintel
```
Run all subsequent commands after activating this environment.

## External Prediction Requirements

The default workflows use external per-residue prediction results, including conservation profiles, disorder predictions, linker predictions, and secondary-structure predictions. These results must be generated using the corresponding external tools before running MSCintel.

The external prediction framework is extensible. Users may remove existing external prediction types or incorporate additional prediction sources according to their analysis requirements. When introducing a new externally derived feature, the corresponding prediction results must first be generated using the appropriate prediction tool and converted into the standardized input format required by the pipeline. 

Detailed file formats, directory structures, and instructions for configuring feature categories are provided in the README file of each module.

When using an existing trained model for prediction, all feature types required by that model must be provided. Changes to the feature system require retraining the model.

