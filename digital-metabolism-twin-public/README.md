# Digital Metabolism Twin (public release)

An open, research-grade codebase for studying how dietary temporal patterns and clinical features predict diabetes risk. The pipeline encodes paired NHANES dietary recall days with a variational autoencoder (VAE), fuses latent diet embeddings with clinical variables, and evaluates ensemble tree models for risk prediction, calibration, and phenotyping.

## Dataset
- **Source:** NHANES (National Health and Nutrition Examination Survey) dietary recalls (DR1TOT/DR2TOT), clinical, and laboratory files.
- **Access:** Download from CDC/NCHS. Place raw files under `data/raw/` (not committed). See `data/README.md` for notes.
- **Labels:** Diabetes status from `DIQ010` (1 = diabetes, 2 = no diabetes).
- **Privacy:** Do not commit any NHANES data to the repository. Use synthetic placeholders for tests only.

## Repository layout
```
project-root/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training_colab.ipynb
│   └── README.md
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   ├── utils/
│   └── main.py
├── scripts/
│   ├── train_local.py
│   ├── evaluate.py
│   └── inference.py
├── models/
│   ├── saved/
│   └── README.md
├── results/
│   ├── figures/
│   ├── tables/
│   └── README.md
├── tests/
└── docs/
```

## Setup
### Option A: pip
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate digital-metabolism-twin
```

## Quick start (inference)
1. Place trained models and preprocessing artifacts in `models/saved/<experiment>/` (xgb_model.joblib, lgb_model.txt, rf_model.joblib, cb_model.cbm, imputer_simple.joblib, scaler_robust.joblib, training_summary.json, feature_importance.csv).
2. Ensure `data/processed/inference_features.parquet` exists and matches the feature contract in `feature_importance.csv`.
3. Run:
```bash
python -m src.main --experiment <experiment> --mode inference --model-type ensemble
```
Outputs are written to `results/predictions.csv` and `results/inference_summary.json`.

## Training (Colab or local)
- **Colab (recommended for GPUs):** Use `notebooks/03_training_colab.ipynb` to mount Drive, pull NHANES data, and call `src.models.train.train_baseline_and_augmented`.
- **Local (CPU):** If you have processed features locally, run:
```bash
python scripts/train_local.py --experiment exp_YYYYMMDD_HHMMSS --save
```
Trained artifacts will be stored under `models/saved/<experiment>/`.

## Evaluation
Compare performance with and without latent diet embeddings:
```bash
python scripts/evaluate.py --experiment <experiment> --model-type ensemble
```
This produces `results/auc_comparison.json`.

## Explainability
Generate SHAP summaries for the XGBoost model:
```bash
python -m src.main --experiment <experiment> --mode shap
```
Outputs: `results/shap_global.csv`, `results/shap_local_high_risk.csv`, `results/shap_summary.json`.

## Reproducibility
- Deterministic seeds set in config (`random_seed: 42`).
- Model artifacts include `training_summary.json` and `feature_importance.csv` for contract checking.
- Hashes of model files are stored in metadata for integrity.
- Tests runnable via `pytest tests/`.

## License
MIT License (see LICENSE).

## Citation
```
@misc{digital_metabolism_twin_2026,
  title  = {Digital Metabolism Twin: Dietary Latents for Diabetes Risk},
  author = {Open-source contributors},
  year   = {2026},
  note   = {Code available at https://github.com/<org>/digital-metabolism-twin}
}
```
