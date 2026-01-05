# Notebooks

Use these notebooks as lightweight orchestration layers; all core logic lives in `src/`.

Recommended order:
- `01_exploration.ipynb` — quick EDA on NHANES dietary + clinical variables (no model training).
- `02_preprocessing.ipynb` — sanity-check feature preparation outputs; call into `src.data` utilities.
- `03_training_colab.ipynb` — Colab-only notebook to train heavy models; uses data from Google Drive.

Keep notebooks small, avoid storing outputs, and clear cell outputs before committing.
