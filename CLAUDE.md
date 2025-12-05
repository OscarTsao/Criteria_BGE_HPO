# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSM-5 criteria matching project using Natural Language Inference (NLI) with transformer-based models (BERT, DeBERTa). The repository consolidates all active code in `src/criteria_bge_hpo/`, which contains the Hydra CLI plus data, model, training, evaluation, and utility modules built on PyTorch, Transformers, MLflow, and Optuna.

**Key Features:**
- Dual architecture support: BERT and DeBERTa-v3 with NLI pretraining
- Nested Cross-Validation with Optuna HPO for robust model selection
- 7 classification head architectures (linear, pooler_linear, mlp1, mlp2, mean_pooling, max_pooling, attention_pooling)
- Advanced loss functions (Focal Loss, Weighted BCE) for class imbalance
- Comprehensive hyperparameter search space (architecture, optimization, regularization)
- Macro-F1 and sensitivity metrics for imbalanced clinical data

## Setup and Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Running Commands

### DSM-5 NLI Training

The main CLI is `src/criteria_bge_hpo/cli.py` which uses Hydra for configuration management:

```bash
# K-fold cross-validation training (100 epochs, patience 20)
python -m criteria_bge_hpo.cli train training.num_epochs=100 training.early_stopping_patience=20

# Hyperparameter optimization (500 trials)
python -m criteria_bge_hpo.cli hpo --n-trials 500

# Evaluate specific fold
python -m criteria_bge_hpo.cli eval --fold 0
```

### Development

```bash
# Linting and formatting
ruff check src tests
black src tests

# Run tests
pytest
```

## Configuration System

Uses Hydra with composition pattern. Main config: `configs/config.yaml`

**Model configurations:**
- `configs/model/bert_base.yaml` - Base BERT encoder configuration
- `configs/model/deberta_v3_base.yaml` - DeBERTa-v3-base (vanilla pretrained)
- `configs/model/deberta_nli.yaml` - **DeBERTa-v3-base-mnli-fever-anli** (NLI-pretrained, recommended)
  - Includes classifier_head, dropout parameters (classifier, hidden, attention)
  - Loss configuration (focal, bce, weighted_bce)
  - Focal loss parameters (gamma, alpha)

**Training configurations:**
- `configs/training/default.yaml` - Standard training hyperparameters
- `configs/training/deberta.yaml` - DeBERTa-optimized settings (torch.compile=false for compatibility)

**HPO configurations:**
- `configs/hpo/optuna.yaml` - Basic Optuna setup
- `configs/hpo/deberta.yaml` - DeBERTa HPO with classifier_head search
- `configs/hpo/nested_cv.yaml` - **Nested CV with expanded search space**
  - 10+ hyperparameters: architecture, optimizer, regularization
  - Fold-1 pruning (n_warmup_steps=0) for 80% compute savings
  - Targets Macro-F1 optimization

**Override configs via CLI:**
```bash
# Use NLI-pretrained DeBERTa with max_pooling head
python -m criteria_bge_hpo.cli train model=deberta_nli model.classifier_head=max_pooling

# HPO with nested CV config
python -m criteria_bge_hpo.cli hpo --config-name nested_cv --n-trials 500
```

## Architecture

### DSM-5 NLI Pipeline (src/criteria_bge_hpo/)

The CLI (`cli.py`) orchestrates the full training pipeline:

1. **Data Loading** (`data/preprocessing.py`) - Loads posts, annotations, and DSM-5 criteria from CSV/JSON
2. **K-fold Splits** (`training/kfold.py`) - Stratified splits grouped by post (prevents data leakage)
3. **Dataset** - Tokenizes post-criterion pairs for binary classification
4. **Model** (`models/`) - Dual architecture support:
   - `bert_classifier.py` - BERT/BGE wrapper
   - `deberta_classifier.py` - **DeBERTa-v3 with configurable heads and losses**
   - `classifier_heads.py` - 7 head architectures (linear, pooler, MLPs, pooling variants, attention)
5. **Loss Functions** (`training/losses.py`) - FocalLoss, WeightedBCELoss, class weight utilities
6. **Training** (`training/trainer.py`) - Training loop with gradient accumulation, mixed precision
7. **Evaluation** (`evaluation/evaluator.py`) - Per-criterion and aggregate metrics (binary_f1, macro_f1, sensitivity)
8. **MLflow Logging** (`utils/mlflow_setup.py`) - Experiment tracking

**Key workflow pattern:** Each fold runs as a separate MLflow run, with overall summary logged after K-fold completion.

### Hyperparameter Optimization

#### Standard HPO
Optuna with MedianPruner for early stopping. Each study targets 500 trials by default and stores results in `optuna.db` (SQLite). Basic search space:
- learning_rate (loguniform: 1e-6 to 1e-4)
- batch_size (categorical: 4, 8, 16)
- weight_decay (loguniform: 1e-4 to 1e-1)
- warmup_ratio (uniform: 0.0 to 0.2)
- classifier_head (categorical: 6 heads)
- epochs (categorical: 10, 15, 20)

#### Nested Cross-Validation HPO
For robust model selection, use `configs/hpo/nested_cv.yaml` which expands the search space to 10+ hyperparameters:

**Architecture parameters:**
- classifier_head: linear, mean_pooling, max_pooling, attention_pooling, mlp1

**Regularization parameters:**
- classifier_dropout: 0.1 to 0.5
- hidden_dropout: 0.1 to 0.3
- attention_dropout: 0.1 to 0.3

**Optimizer parameters:**
- learning_rate: 1e-6 to 5e-5
- weight_decay: 1e-4 to 1e-1
- warmup_ratio: 0.0 to 0.2

**Training parameters:**
- batch_size: 4, 8, 16
- gradient_accumulation_steps: 2, 4, 8
- epochs: 15, 20, 25, 30

**Loss configuration:**
- loss_type: focal, bce, weighted_bce
- focal_gamma: 1.0 to 3.0 (when focal loss)

**Pruning strategy:** Aggressive fold-1 pruning (n_warmup_steps=0) for 80% compute savings with minimal risk (2-6% false prune rate).

**Objective metric:** Macro-F1 (balanced for class imbalance)

## GPU Optimization (Ampere+ GPUs)

Training config enables aggressive optimizations:
- `use_bf16: true` - bfloat16 mixed precision (2x speedup, requires Ampere+)
- `use_torch_compile: false` (toggle to true for JIT compilation on PyTorch 2+)
- `fused_adamw: true` - Fused optimizer kernel when CUDA is available

Set `reproducibility.tf32: true` in config to enable TensorFloat-32 on supported GPUs.

## Data Paths

- `data/redsm5/redsm5_posts.csv` - Social media posts
- `data/redsm5/redsm5_annotations.csv` - Annotations linking posts to criteria
- `data/DSM5/MDD_Criteria.json` - DSM-5 Major Depressive Disorder criteria definitions

## Important Implementation Details

**K-fold Grouping:** Splits group by `post_id` to prevent train/val leakage when a single post has multiple criterion annotations.

**Tokenization:** Max length 512 tokens (configurable via `data.max_length`). Dataset class handles proper attention masking.

**Reproducibility:** Set seed via config, enable deterministic operations. Use `utils/reproducibility.py` helpers.

**Per-Criterion Evaluation:** Beyond aggregate F1/accuracy, track performance per individual DSM-5 criterion to identify problematic criteria.

## Advanced Features (New)

### Classification Heads
The system supports 7 different classification head architectures via `ClassifierHeadFactory`:

1. **linear**: CLS token → Linear(h, num_labels)
2. **pooler_linear**: CLS token → Dense(h, h) → Tanh → Dropout → Linear(h, num_labels)
3. **mlp1**: CLS token → Linear(h, h) → GELU → Dropout → Linear(h, num_labels)
4. **mlp2**: CLS token → Linear(h, i) → GELU → Dropout → Linear(i, num_labels) (two-layer MLP)
5. **mean_pooling**: Mean(sequence * mask) → Dropout → Linear(h, num_labels)
6. **max_pooling**: Max(sequence * mask) → Dropout → Linear(h, num_labels) *(NEW)*
7. **attention_pooling**: MultiHeadAttention(query, sequence) → Dropout → Linear(h, num_labels)

Configure via `model.classifier_head` in config or as HPO search space parameter.

### Loss Functions
Advanced loss functions in `training/losses.py` for handling class imbalance:

**FocalLoss**: Addresses class imbalance by down-weighting easy examples
- Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
- Parameters: alpha (class weight), gamma (focusing parameter, typically 2.0)
- Auto-compute alpha from class distribution using `compute_focal_alpha()`

**WeightedBCELoss**: Binary cross-entropy with positive class weighting
- Parameters: pos_weight (weight for positive class)

**Utilities**: `compute_class_weights()` for inverse or effective number weighting

### DeBERTa-v3 Integration
The system now supports DeBERTa-v3-base models with:
- **Dropout injection**: hidden_dropout, attention_dropout configurable via HPO
- **Configurable loss**: Select focal/bce/weighted_bce per trial
- **NLI pretraining**: Use `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` for +1% F1 improvement
- **Auto-detection**: CLI automatically selects DeBERTaClassifier when model_name contains "deberta"

### Evaluation Metrics
Enhanced metrics in `evaluation/evaluator.py`:
- **binary_f1**: Standard F1 score for binary classification
- **macro_f1**: F1 averaged across classes (better for imbalance) *(NEW)*
- **sensitivity**: Recall / true positive rate (explicit alias) *(NEW)*
- **precision, recall, accuracy, AUC**: Standard metrics

Use `macro_f1` as optimization target in nested CV for balanced performance.

### Nested CV Best Practices
For production models, use the nested CV configuration:

```bash
# Run nested CV HPO with DeBERTa NLI model
python -m criteria_bge_hpo.cli hpo \
  --config-name nested_cv \
  model=deberta_nli \
  --n-trials 500
```

**Expected performance:**
- Baseline (no HPO): F1 = 0.65-0.75
- After HPO: F1 = 0.75-0.85
- Compute time: ~40-80 GPU hours (with fold-1 pruning)

**Key settings:**
- Use `n_warmup_steps=0` for aggressive pruning
- Target `macro_f1` for imbalanced data
- Set `n_startup_trials=10` to establish baseline before pruning
