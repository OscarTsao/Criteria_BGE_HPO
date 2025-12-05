# Repository Guidelines

## Project Structure & Module Organization
Primary code sits in `src/criteria_bge_hpo`, which exposes the Hydra CLI (`cli.py`), data prep, model, training, and evaluation modules. Configurations live in `configs/` (root `config.yaml` plus overrides in `model/`, `training/`, and `hpo/`); edit YAML rather than hardcoding defaults. Keep raw data in `data/`, generated checkpoints or reports in `outputs/`, MLflow runs under `mlruns/`, and Optuna artifacts in `optuna.db`.

## Build, Test, and Development Commands
Set up the toolchain with `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'`. Typical workflows: `python -m criteria_bge_hpo.cli train training.num_epochs=100 training.early_stopping_patience=20` for end-to-end K-fold training, `python -m criteria_bge_hpo.cli eval --fold 0` to sanity-check changes, and `python -m criteria_bge_hpo.cli hpo --n-trials 500` to spawn Optuna jobs. Run `ruff check src tests`, `black src tests`, `mypy src`, and `pytest` (optionally `--maxfail=1`) before committing.

## Coding Style & Naming Conventions
The project targets Python 3.10 with Black/Ruff defaults (100-character lines, double quotes). Use `snake_case` for functions/variables, `PascalCase` for classes, and ALL_CAPS for constants. Create loggers via `criteria_bge_hpo.utils` helpers and surface tunable values through Hydra configs or CLI overrides instead of literals.

## Testing Guidelines
Add fast unit tests for every feature plus, where practical, a thin integration test that executes the CLI against a tiny CSV/JSON fixture. Mirror package layout inside `tests/`, prefix files with `test_`, and seed randomness with `criteria_bge_hpo.utils.reproducibility.set_seed` so metrics stay stable. Assert on numeric metrics and filesystem side effects (e.g., checkpoint folders under `outputs/<experiment>`). Target ~80% coverage on new code and note justified gaps in the PR description.

## Commit & Pull Request Guidelines
Write imperative, scope-prefixed commits (`train: add bf16 flag`) and keep the summary under 72 characters; use bodies only when additional context is required. Pull requests should state the problem being solved, summarize the implementation, list the exact commands/tests executed, and link related issues or Optuna/MLflow run IDs. Include screenshots or log excerpts whenever you change reported metrics and call out any schema or config updates.

## Experiment & Tracking Tips
Hydra overrides live under `configs/`, so prefer commands like `python -m criteria_bge_hpo.cli train model=deberta_v3_base training=default training.num_epochs=100 training.early_stopping_patience=20` to capture provenance in MLflow. Keep long-lived artifacts in `mlruns/` (untracked) and Optuna trials in `optuna.db`; fork those stores instead of deleting records. Reference sensitive values via `${oc.env:VAR}` in configs and load them from environment variables or a local `.env` file that remains untracked.
