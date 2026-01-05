# Models

Trained model artifacts are **not** committed.

Expected contents after training or download:
- `saved/` — versioned checkpoints (e.g., `exp_YYYYMMDD_HHMMSS/xgb_model.joblib`, `vae_model.pt`).
- `metadata.json` — optional index of available runs with metrics and dataset notes.

To reproduce or fetch models, follow the instructions in the top-level README. Ensure every saved model directory contains:
- `training_summary.json` — dataset splits, metrics, seed, and hyperparameters.
- Preprocessing artifacts (imputer, scaler) if required by the model.
- Hash of model files for integrity checks.
