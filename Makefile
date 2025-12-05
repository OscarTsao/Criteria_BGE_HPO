# ============================================================================
# DSM-5 NLI Binary Classification - Makefile
# ============================================================================
# Automation for common development and training tasks
# Usage: make <target>
# Example: make setup && make train
# ============================================================================

# Declare all targets as phony (not actual files)
.PHONY: help setup train hpo hpo-noaug hpo-aug nested-cv nested-cv-fast eval clean test lint format

# Default tracking backends (override via env if needed)
MLFLOW_URI ?= file:mlruns
OPTUNA_URI ?= sqlite:///optuna.db
PYTHON ?= python3
N_TRIALS ?= 100
N_OUTER_FOLDS ?= 5
N_INNER_FOLDS ?= 3
EXTRA_ARGS ?=
# Ensure this repo's src/ is first on PYTHONPATH (avoids picking up other editable installs)
PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH),)
export PYTHONPATH

# ============================================================================
# HELP - Display available targets and their descriptions
# ============================================================================
help:
	@echo "DSM-5 NLI Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and setup environment"
	@echo "  train         - Run 5-fold cross-validation training (100 epochs, patience 20)"
	@echo "  hpo           - Run hyperparameter optimization (100 trials)"
	@echo "  hpo-noaug     - Explicit HPO run with augmentation disabled (100 trials)"
	@echo "  hpo-aug       - Run hyperparameter optimization with augmentation enabled (100 trials, 30% prob)"
	@echo "  nested-cv     - Run nested cross-validation with HPO (5 outer × 100 trials × 3 inner)"
	@echo "  nested-cv-fast- Run nested CV with reduced trials (5 outer × 50 trials × 3 inner)"
	@echo "  eval          - Evaluate fold 0"
	@echo "  clean         - Clean outputs, cache, and logs"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black"

# ============================================================================
# SETUP - Install dependencies in editable mode
# ============================================================================
# Upgrades pip and installs the package with development dependencies
# Run this once after cloning the repository
# Creates: .venv/lib/python3.10/site-packages/criteria_bge_hpo.egg-link
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev]'  # Editable install with dev dependencies (pytest, ruff, black)
	@echo "✓ Setup complete!"

# ============================================================================
# TRAIN - Run full K-fold cross-validation training
# ============================================================================
# Trains 5 separate models (one per fold) with default hyperparameters
# HPC Optimizations: Flash Attention 2, BF16 mixed precision, persistent workers
# Logs results to MLflow (mlruns/ directory)
# Runtime: ~30-60 minutes depending on GPU (A100: ~20 min with Flash Attn 2)
# Output: mlruns/, outputs/dsm5_criteria_matching/checkpoints/
train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m criteria_bge_hpo.cli command=train model=deberta_nli training.num_epochs=100 training.early_stopping_patience=20 $(EXTRA_ARGS)

# ============================================================================
# HPO - Run hyperparameter optimization with Optuna
# ============================================================================
# Uses configs/hpo/optuna.yaml (lr/batch/weight decay/warmup search space)
# Defaults to 100 trials with aggressive pruning for efficiency
# HPC Optimizations: Flash Attention 2, BF16, 100 max epochs, 20 patience
# Results stored in SQLite (Optuna) and MLflow by default
# Pass EXTRA_ARGS for Hydra overrides (e.g., hpo.search_space.warmup_ratio.low=0.02)
# Runtime: multi-hour for 100 trials with pruning (A100-class GPU)
hpo:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo model=deberta_nli n_trials=$(N_TRIALS) training.num_epochs=100 training.early_stopping_patience=20 $(EXTRA_ARGS)

# ============================================================================
# HPO-NOAUG - Explicitly disable augmentation for HPO
# ============================================================================
hpo-noaug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo model=deberta_nli n_trials=$(N_TRIALS) training.num_epochs=100 training.early_stopping_patience=20 augmentation.enable=false augmentation.prob=0.0 $(EXTRA_ARGS)

# ============================================================================
# HPO-AUG - Hyperparameter optimization with augmentation enabled
# ============================================================================
# Same search as HPO but turns on evidence-span augmentation (synonym aug by default)
# Override augmentation.prob/type via EXTRA_ARGS if desired
hpo-aug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo model=deberta_nli n_trials=$(N_TRIALS) training.num_epochs=100 training.early_stopping_patience=20 augmentation.enable=true augmentation.prob=0.3 augmentation.type=synonym $(EXTRA_ARGS)

# ============================================================================
# NESTED-CV - Run nested cross-validation with per-fold HPO (HPC optimized)
# ============================================================================
# NESTED-CV - Run nested cross-validation with per-fold HPO (HPC optimized)
# ============================================================================
# Nested CV Structure:
#   - Outer loop: 5 folds for unbiased performance estimation
#   - Inner loop: 3 folds for hyperparameter optimization (100 trials each by default)
#   - Per outer fold: Run full HPO study, train final model, evaluate on test
#
# HPC Optimizations (see src/criteria_bge_hpo/training/nested_cv.py):
#   - Flash Attention 2 with BF16 mixed precision (2-3x speedup on Ampere+ GPUs)
#   - HyperbandPruner for aggressive trial pruning (30-50% reduction)
#   - Persistent dataloader workers (reduced overhead)
#   - Batch size search space: [16, 32] with gradient accumulation
#   - MAX_EPOCHS: 100, PATIENCE: 20
#
# Computational Cost: 5 outer × 100 trials × 3 inner × ~50 avg epochs ≈ 7,500 fold-epochs
#   With pruning: ~5,000 fold-epochs (≈33% reduction)
#   Estimated runtime (A100): multi-hour; use Hyperband pruning + parallelization
#
# Override defaults: make nested-cv N_TRIALS=150 N_OUTER_FOLDS=3
# Pass extra config: make nested-cv EXTRA_ARGS="training.use_augmentation=true"
nested-cv:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) \
	$(PYTHON) -m criteria_bge_hpo.training.nested_cv \
		--n-outer-splits $(N_OUTER_FOLDS) \
		--n-inner-splits $(N_INNER_FOLDS) \
		--n-trials $(N_TRIALS) \
		$(EXTRA_ARGS)

# ============================================================================
# NESTED-CV-FAST - Quick nested CV for testing/debugging (reduced trials)
# ============================================================================
# Faster version with 50 trials per outer fold (half of default)
# Useful for:
#   - Testing pipeline before full run
#   - Rapid prototyping of new features
#   - CI/CD integration tests
# Runtime: ~3-5 hours (A100)
nested-cv-fast:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) \
	$(PYTHON) -m criteria_bge_hpo.training.nested_cv \
		--n-outer-splits $(N_OUTER_FOLDS) \
		--n-inner-splits $(N_INNER_FOLDS) \
		--n-trials 50 \
		$(EXTRA_ARGS)

# ============================================================================
# EVAL - Evaluate a specific fold
# ============================================================================
# Loads trained model from fold 0 and runs evaluation
# Displays per-criterion metrics and aggregate performance
eval:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m criteria_bge_hpo.cli command=eval fold=0 $(EXTRA_ARGS)

# ============================================================================
# CLEAN - Remove all generated files, outputs, and cache
# ============================================================================
# Deletes:
#   - outputs/ - Model checkpoints and training artifacts
#   - mlruns/ - MLflow experiment tracking data
#   - optuna.db - HPO trial history
#   - .pytest_cache/ - pytest cache
#   - __pycache__/ - Python bytecode cache (all directories)
#   - *.pyc - Compiled Python files
# WARNING: This is destructive. Trained models will be lost.
clean:
	rm -rf outputs/
	rm -rf mlruns/
	rm -rf optuna.db
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	# Find and remove all __pycache__ directories recursively (ignore errors)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	# Find and remove all .pyc files recursively
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned all outputs and cache"

# ============================================================================
# TEST - Run pytest test suite with coverage reporting
# ============================================================================
# Runs all tests in tests/ directory
# Generates HTML coverage report in htmlcov/
# Flags:
#   -v: Verbose output (show individual test results)
#   --cov: Measure code coverage for src/criteria_bge_hpo
#   --cov-report=html: Generate HTML coverage report
# Output: htmlcov/index.html (open in browser to view coverage)
test:
	pytest tests/ -v --cov=src/criteria_bge_hpo --cov-report=html

# ============================================================================
# LINT - Run code quality checks with ruff
# ============================================================================
# Checks for:
#   - PEP 8 style violations
#   - Common bugs and code smells
#   - Import sorting issues
#   - Unused imports and variables
# Does NOT modify files (use 'make format' to auto-fix)
# Exit code: 0 if clean, 1 if issues found
lint:
	ruff check src tests
	@echo "✓ Linting complete"

# ============================================================================
# FORMAT - Auto-format code with black and fix linting issues
# ============================================================================
# Steps:
#   1. black: Formats all Python files to consistent style (line length 100)
#   2. ruff --fix: Auto-fixes safe linting issues (imports, unused vars, etc.)
# Modifies files in-place
# Always run before committing code
format:
	black src tests                # Format with black (line length from pyproject.toml)
	ruff check --fix src tests     # Auto-fix safe linting issues
	@echo "✓ Code formatted"
