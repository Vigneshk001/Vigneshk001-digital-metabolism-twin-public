# Methodology

This document outlines the study design, data sources, and modeling approach for the digital metabolism twin. Summaries:

- **Objective:** Use dietary temporal representations plus clinical features to predict diabetes risk and stratify phenotypes.
- **Data:** NHANES dietary recalls (Day 1/Day 2), clinical and laboratory variables. See `docs/citations.md` for references.
- **Models:** Variational autoencoder (VAE) to encode paired dietary days; gradient boosting, random forest, and CatBoost classifiers for supervised risk prediction.
- **Evaluation:** AUC, average precision, calibration analysis, and latent-space correlations with biomarkers.

Fill in study-specific details as you finalize experiments.
