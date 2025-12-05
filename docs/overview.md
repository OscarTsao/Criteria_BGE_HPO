# DSM-5 Criteria Matching – Architecture Overview

## Components

- `configs/`: Hydra root config plus overrides for model, training, and HPO.
- `src/criteria_bge_hpo/`: CLI entrypoint plus data, model, training, evaluation, and utility modules.
- `tests/`: Pytest suite mirroring the package layout; add new files for every module you touch.

## Execution Flow

1. `criteria_bge_hpo.cli` loads Hydra config, sets up MLflow/Optuna, and kicks off the requested command.
2. `data.preprocessing` reads the unified ground-truth CSV and DSM-5 JSON metadata, then prepares `(post, criterion)` pairs.
3. `data.dataset` tokenizes each pair and exposes PyTorch loaders with safe fallbacks when multiprocessing is unavailable.
4. `models.bert_classifier` wraps a Hugging Face encoder for binary scoring (single- or two-logit heads).
5. `training.trainer` handles optimization, early stopping, mixed precision, and checkpointing per fold.
6. `evaluation.evaluator` reports aggregate plus per-criterion metrics and logs them to MLflow.

## Experiment Tracking

- Set `MLFLOW_TRACKING_URI` and `OPTUNA_STORAGE` via `.env` (copy `env.example`), or override per command.
- Outputs remain under `outputs/<experiment>/` while MLflow runs live in `mlruns/` (ignored by git).

## Testing

- Run `pytest` for the fast unit suite, `ruff check` for linting, and `black` for formatting.
- Use `tests/test_data_pipeline.py` as an example for lightweight fixtures that avoid touching the full dataset.
