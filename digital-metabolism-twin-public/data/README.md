# Data Directory

This repository does **not** ship datasets. Use this folder to store data locally:

- `raw/` — externally sourced data (NHANES dietary and clinical files). Do not commit.
- `processed/` — cached feature tables, latent states, and inference-ready matrices. Do not commit.

## Access & preparation
1. Download NHANES dietary (DR1TOT/DR2TOT), clinical, and laboratory files for the study years of interest.
2. Place raw files under `data/raw/` (organized by survey cycle/year).
3. Run the preprocessing steps documented in the top-level README to generate `processed/` artifacts.

Add small synthetic samples only if needed for tests; never commit identifiable or full datasets.
