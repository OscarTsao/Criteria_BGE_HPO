# ============================================================================
# DSM-5 NLI Binary Classification - Makefile
# ============================================================================
# Automation for common development and training tasks
# Usage: make <target>
# Example: make setup && make train
# ============================================================================

# Declare all targets as phony (not actual files)
.PHONY: help setup train hpo eval clean test lint format

# ============================================================================
# HELP - Display available targets and their descriptions
# ============================================================================
help:
	@echo "DSM-5 NLI Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Install dependencies and setup environment"
	@echo "  train    - Run 5-fold cross-validation training"
	@echo "  hpo      - Run hyperparameter optimization (50 trials)"
	@echo "  eval     - Evaluate fold 0"
	@echo "  clean    - Clean outputs, cache, and logs"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code with black"

# ============================================================================
# SETUP - Install dependencies in editable mode
# ============================================================================
# Upgrades pip and installs the package with development dependencies
# Run this once after cloning the repository
# Creates: .venv/lib/python3.10/site-packages/dsm5-nli.egg-link
setup:
	python -m pip install --upgrade pip
	pip install -e '.[dev]'  # Editable install with dev dependencies (pytest, ruff, black)
	@echo "✓ Setup complete!"

# ============================================================================
# TRAIN - Run full K-fold cross-validation training
# ============================================================================
# Trains 5 separate models (one per fold) with default hyperparameters
# Logs results to MLflow (mlruns/ directory)
# Runtime: ~30-60 minutes depending on GPU
# Output: mlruns/, outputs/dsm5_criteria_matching/checkpoints/
train:
	python -m dsm5_nli.cli command=train

# ============================================================================
# HPO - Run hyperparameter optimization with Optuna
# ============================================================================
# Searches 50 hyperparameter combinations using Optuna
# Each trial runs abbreviated 3-epoch K-fold CV
# Results stored in optuna.db (SQLite) and MLflow
# Runtime: ~2-4 hours for 50 trials
# Output: optuna.db, mlruns/
hpo:
	python -m dsm5_nli.cli command=hpo n_trials=50

# ============================================================================
# EVAL - Evaluate a specific fold (not yet implemented)
# ============================================================================
# Loads trained model from fold 0 and runs evaluation
# Displays per-criterion metrics and aggregate performance
eval:
	python -m dsm5_nli.cli command=eval fold=0

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
#   --cov: Measure code coverage for src/dsm5_nli
#   --cov-report=html: Generate HTML coverage report
# Output: htmlcov/index.html (open in browser to view coverage)
test:
	pytest tests/ -v --cov=src/dsm5_nli --cov-report=html

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
