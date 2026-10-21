# MSCintel

MSCintel is a sequence-based computational framework for investigating sequence-derived features associated with mechanosensitivity and identifying mechanosensitive ion channels (MSCs).

This repository provides two main modules:

1. **Model development**: A flexible and extensible pipeline for feature selection, classification model construction, and performance evaluation. It supports user-provided features and is not limited to MSC classification.
2. **MSC classification**: Identify potential mechanosensitive ion channels from protein sequences using two complementary methods: SMOD_star, an interpretable classification model based on selected sequence-derived features, and ESM2-LR, which combines ESM2 protein sequence embeddings with Logistic Regression.

## Main Modules

| Module | Purpose | Documentation |
|---|---|---|
| `model_development/` | Perform statistical feature screening, wrapper-based feature selection, Stacking classifier construction, and model evaluation. | [`model_development/README.md`](model_development/README.md) |
| `MSC_classification/` | Predict whether protein sequences are mechanosensitive ion channels using SMOD_star or ESM2-LR. | [`SMOD_star/README.md`](MSC_classification/SMOD_star/README.md), [`ESM2-LR/README.md`](MSC_classification/ESM2-LR/README.md) |


## System Requirements

- **Operating systems:** CentOS 7 and Windows 11.
- **Python:** 3.8.
- **Environment management:** Conda.
- **Dependencies:** See [`environment.yml`](environment.yml).
- **Hardware:** GPU is optional, but recommended for accelerating ESM2 embedding generation and model training.


## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/ChunyaoXin/MSCintel.git
cd MSCintel

conda env create -f environment.yml
conda activate mscintel
```

Run all subsequent commands after activating this environment.


## Installation and Demo Runtime

Approximate runtimes were measured on the following hardware:

- **Linux server:** NVIDIA RTX A6000 GPU (48 GB), 52-core CPU.
- **Windows desktop:** 8-core CPU (no CUDA-enabled GPU).


| Task | Linux Server (GPU) | Windows Desktop (CPU) |
|---|---|---|
| Environment installation | ~12 min | ~5 min |
| Model development demo | ~13 min | ~25 min |
| SMOD_star prediction demo | ~15 s | ~15 s |
| ESM2-LR prediction demo | ~20 s | ~50 s |

Runtimes are based on the provided example datasets and may vary with hardware and execution conditions.