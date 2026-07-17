# Prediction Tools
This directory provides two complementary tools for predicting mechanosensitive ion channels (MSCs) from protein sequences: SMOD_star and ESM2-LR.

## Method Overview

SMOD_star is the interpretable prediction model implemented within the MSCintel framework. ESM2-LR is provided as a complementary tool for applications that prioritize classification accuracy over mechanistic interpretability.

| Method        | Input                                                            | Main advantage               | Documentation                                |
| ------------- | ---------------------------------------------------------------- | ---------------------------- | -------------------------------------------- |
| **SMOD_star** | Protein FASTA file and corresponding external prediction results | Mechanistic interpretability | [`SMOD_star/README.md`](SMOD_star/README.md) |
| **ESM2-LR**   | Protein FASTA file or existing ESM2 embeddings                   | Higher predictive performance       | [`ESM2_LR/README.md`](ESM2_LR/README.md)     |

The two tools can be used independently. Users may choose either method according to their application requirements.


