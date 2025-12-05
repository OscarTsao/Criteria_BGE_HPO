# DSM-5 Criteria Matching

End-to-end experimentation framework for classifying DSM-5 symptoms via natural language inference (NLI).
Built on PyTorch + Hugging Face Transformers with Hydra configuration, MLflow tracking, and
Optuna hyper-parameter optimization.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Copy `env.example` to `.env` (or export the variables another way) so MLflow and Optuna know where
to persist experiment metadata locally.

## Dataset

The project now relies on a single unified ground-truth file:

- `data/groundtruth/criteria_matching_groundtruth.csv`
  - Columns: `post_id`, `post`, `DSM5_symptom`, `groundtruth`
  - Each row already represents a `(post, criterion)` pair with a binary label.
  - The preprocessing step maps `DSM5_symptom` to the DSM-5 criterion text and renames the
    label column internally.

Supporting metadata remains available for reference:

- `data/DSM5/MDD_Criteria.json` – canonical DSM-5 criterion definitions
- `data/redsm5/*` – original posts/annotations (kept for provenance)
- Other CSVs under `data/groundtruth/` – additional tasks that share the same schema

## Running Experiments

Hydra drives all CLI commands via `configs/config.yaml`. Examples:

```bash
# K-fold training (100 epochs w/ patience 20, default: 5 folds)
python -m criteria_bge_hpo.cli command=train training.num_epochs=100 training.early_stopping_patience=20

# Evaluate a saved fold checkpoint
python -m criteria_bge_hpo.cli command=eval fold=0

# Hyper-parameter search with Optuna (LoRA/QLoRA + threshold tuning; defaults to 100 trials)
python -m criteria_bge_hpo.cli command=hpo n_trials=100
```

Training logs accuracy, F1, precision, recall, and AUC per fold and saves the best checkpoint for each
fold to `outputs/<experiment>/checkpoints/fold_<n>_best.pt`.

Key config knobs (see `configs/hpo/optuna.yaml` for the HPO search space):

- Model backbone: `microsoft/deberta-v3-base` by default (overrideable with any HF encoder)
- `data.groundtruth_csv` – location of the unified dataset
- `data.criteria_json` – DSM-5 criterion metadata
- `model.*`, `training.*`, `hpo.*` – composable overrides under `configs/`
- HPO search space covers learning rate, batch size, weight decay, and warmup ratio.

## Development Workflow

- Format & lint: `ruff check src tests` and `black src tests`
- Type check: `mypy src`
- Tests: `pytest`
- End-to-end training check: `python -m criteria_bge_hpo.cli command=train training.num_epochs=100 training.early_stopping_patience=20`
- Evaluate a trained fold: `python -m criteria_bge_hpo.cli command=eval fold=0` (requires checkpoint under `outputs/<experiment>/checkpoints`)

## Outputs & Tracking

- `outputs/<experiment>` – checkpoints, fold artifacts, reports
- `mlruns/` – MLflow tracking storage (default local SQLite backend)
- `optuna.db` – Optuna study storage when using SQLite

## Project Structure (partial)

```
configs/                # Hydra configs (root + model/training/hpo overrides)
data/                   # Groundtruth CSVs, DSM-5 metadata, legacy sources
src/criteria_bge_hpo/   # CLI entrypoint, data pipeline, models, trainer, utils
tests/                  # Pytest suite covering configs/models/datasets
```
