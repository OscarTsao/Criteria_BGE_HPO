"""Nested cross-validation system for hyperparameter optimization.

This module implements true nested CV with per-outer-fold HPO to avoid
data leakage and provide unbiased performance estimates.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from criteria_bge_hpo.data.nli_templates import NLITemplateGenerator
from criteria_bge_hpo.data.augmentation import ENABLE_TEXTATTACK_LIBRARY
from criteria_bge_hpo.data.dataset import DSM5NLIDataset, seed_worker
from criteria_bge_hpo.evaluation.metrics import compute_all_metrics, matthews_correlation_coefficient
from criteria_bge_hpo.models.deberta_classifier import DeBERTaClassifier
from criteria_bge_hpo.training.kfold import create_nested_kfold_splits

console = Console()

# HPC Configuration Constants
MAX_EPOCHS = 100  # Maximum epochs per training run
PATIENCE = 20  # Early stopping patience (increased from 5)
N_TRIALS_DEFAULT = 100  # Default number of Optuna trials per fold

# BFloat16 support check
HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@dataclass
class NestedCVResults:
    """Results from nested cross-validation.

    Attributes:
        outer_fold_results: List of metric dictionaries per outer fold
        outer_fold_best_hps: List of best hyperparameter dicts per outer fold
        mean_metrics: Mean values for each metric across outer folds
        std_metrics: Standard deviation for each metric across outer folds
        global_best_hps: Optional global best hyperparameters
        n_outer_folds: Number of outer CV folds
        n_inner_folds: Number of inner CV folds for HPO
        n_trials_per_fold: Number of Optuna trials per outer fold
    """

    outer_fold_results: List[Dict[str, Any]]
    outer_fold_best_hps: List[Dict[str, Any]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    global_best_hps: Optional[Dict[str, Any]] = None
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    n_trials_per_fold: int = 100

    def summary(self) -> str:
        """Generate human-readable summary of nested CV results.

        Returns:
            Formatted summary string with aggregated metrics and per-fold results
        """
        lines = [
            "=" * 80,
            "Nested Cross-Validation Results Summary",
            "=" * 80,
            f"Configuration: {self.n_outer_folds} outer folds × "
            f"{self.n_trials_per_fold} trials × {self.n_inner_folds} inner folds",
            "",
            "Aggregated Performance (Mean ± Std across outer folds):",
            "-" * 80,
        ]

        # Priority metrics to display
        priority_metrics = ["mcc", "balanced_accuracy", "f1", "accuracy", "precision", "recall"]

        for metric_name in priority_metrics:
            if metric_name in self.mean_metrics:
                mean_val = self.mean_metrics[metric_name]
                std_val = self.std_metrics.get(metric_name, 0.0)
                lines.append(f"  {metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")

        lines.append("")
        lines.append("Per-Fold Performance:")
        lines.append("-" * 80)

        for fold_idx, fold_result in enumerate(self.outer_fold_results):
            mcc = fold_result.get("mcc", 0.0)
            f1 = fold_result.get("f1", 0.0)
            acc = fold_result.get("accuracy", 0.0)
            bacc = fold_result.get("balanced_accuracy", 0.0)
            lines.append(
                f"  Fold {fold_idx}: MCC={mcc:.4f}, F1={f1:.4f}, "
                f"Acc={acc:.4f}, BAcc={bacc:.4f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


class NestedCVTrainer:
    """Nested cross-validation trainer with per-outer-fold HPO.

    Implements true nested CV to avoid data leakage:
    1. For each outer fold:
        a. Run Optuna study on inner CV folds
        b. Train final model with best HP on full outer train
        c. Evaluate on outer test set
    2. Aggregate results across outer folds
    3. Report mean ± std of metrics
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: DictConfig,
        device: str = "cuda",
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        n_trials: int = N_TRIALS_DEFAULT,
        random_state: int = 42,
    ):
        """Initialize nested CV trainer."""
        self.data = data
        self.config = config
        self.device = device
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.seed = int(getattr(config, "seed", 42))

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

        # Initialize NLI generator
        self.nli_generator = NLITemplateGenerator(
            dsm5_json_path=config.data.dsm5_json_path,
            template_type=config.data.get("nli_template_type", "entailment"),
        )

        console.print(
            f"[bold green]Initialized NestedCVTrainer:[/bold green] "
            f"{n_outer_splits} outer folds, {n_inner_splits} inner folds, "
            f"{n_trials} trials/fold"
        )

    def _build_augment_config(self, hyperparams: Dict[str, Any]) -> Optional[SimpleNamespace]:
        """Construct augmentation config object from sampled hyperparameters."""
        if not hyperparams.get("use_augmentation", False):
            return None

        method = hyperparams.get("aug_method", "nlpaug_synonym")
        lib = "nlpaug"
        aug_type = method
        if method.startswith("textattack"):
            lib = "textattack"
        elif method.startswith("nlpaug_"):
            aug_type = method.split("nlpaug_", 1)[-1]

        return SimpleNamespace(
            enable=True,
            lib=lib,
            type=aug_type,
            prob=hyperparams.get("aug_prob", 0.0),
            imbalance_mode=hyperparams.get("imbalance_mode", "minority_only"),
        )

    def _dataloader_kwargs(self, shuffle: bool, batch_size: int, seed_offset: int = 0) -> Dict[str, Any]:
        """Common DataLoader kwargs with deterministic seeding."""
        num_workers = self.config.training.get("num_workers", 4)
        generator = torch.Generator()
        generator.manual_seed(self.seed + seed_offset)

        return {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True if num_workers > 0 else False,
            "worker_init_fn": seed_worker,
            "generator": generator,
        }

    def run(self) -> NestedCVResults:
        """Run complete nested CV workflow."""
        console.print("\n[bold cyan]Starting Nested Cross-Validation[/bold cyan]")

        # Generate nested splits
        nested_splits = list(create_nested_kfold_splits(
            self.data,
            n_outer_splits=self.n_outer_splits,
            n_inner_splits=self.n_inner_splits,
            random_state=self.random_state,
        ))

        outer_fold_results = []
        outer_fold_best_hps = []

        # Outer CV loop
        for outer_fold, outer_train_idx, outer_test_idx, inner_splits in nested_splits:
            console.print(
                f"\n[bold cyan]Outer Fold {outer_fold}/{self.n_outer_splits}[/bold cyan]"
            )

            # Step 1: HPO on inner folds
            best_hps = self._run_hpo_for_outer_fold(
                outer_fold=outer_fold,
                outer_train_idx=outer_train_idx,
                inner_splits=inner_splits,
            )
            outer_fold_best_hps.append(best_hps)

            # Step 2: Train final model with best HPs on full outer train
            final_model = self._train_final_model(
                outer_fold=outer_fold,
                outer_train_idx=outer_train_idx,
                hyperparams=best_hps,
            )

            # Step 3: Evaluate on outer test set
            test_metrics = self._evaluate_outer_test(
                outer_fold=outer_fold,
                outer_test_idx=outer_test_idx,
                model=final_model,
            )
            outer_fold_results.append(test_metrics)

            console.print(
                f"[green]Outer Fold {outer_fold} Complete:[/green] "
                f"MCC={test_metrics.get('mcc', 0.0):.4f}, "
                f"F1={test_metrics.get('f1', 0.0):.4f}"
            )

        # Aggregate results
        results = self._aggregate_results(outer_fold_results, outer_fold_best_hps)

        # Print summary
        console.print("\n" + results.summary())

        return results

    def _run_hpo_for_outer_fold(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
        inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Run Optuna HPO study on inner CV folds."""
        console.print(
            f"  [yellow]Running HPO with {self.n_trials} trials on "
            f"{len(inner_splits)} inner folds...[/yellow]"
        )

        # Create Optuna study with HyperbandPruner for aggressive early termination
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state + outer_fold),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=MAX_EPOCHS,
                reduction_factor=3,
            ),
        )

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hps = self._sample_hyperparameters(trial)

            # Evaluate on inner CV folds
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_splits:
                try:
                    score = self._train_and_evaluate_inner(
                        inner_train_idx=inner_train_idx,
                        inner_val_idx=inner_val_idx,
                        hyperparams=hps,
                        trial=trial,
                    )
                    inner_scores.append(score)
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    return -1.0

            # Return mean score
            return np.mean(inner_scores) if inner_scores else -1.0

        # Run optimization
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        console.print(f"  [green]HPO Complete:[/green] Best MCC = {study.best_value:.4f}")

        return study.best_params

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        hps = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.1, 0.5),
            "hidden_dropout": trial.suggest_float("hidden_dropout", 0.1, 0.3),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.1, 0.3),
            "classifier_head": trial.suggest_categorical(
                "classifier_head",
                ["linear", "mean_pooling", "max_pooling", "attention_pooling", "mlp1"],
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
            "num_epochs": trial.suggest_int("num_epochs", 20, MAX_EPOCHS),
            "use_augmentation": trial.suggest_categorical("use_augmentation", [True, False]),
            "loss_type": trial.suggest_categorical("loss_type", ["focal", "bce", "weighted_bce"]),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
            "pos_weight": trial.suggest_float("pos_weight", 1.0, 5.0),
            "imbalance_mode": trial.suggest_categorical("imbalance_mode", ["minority_only", "all"]),
        }

        # Augmentation configuration (conditional based on flag)
        if hps["use_augmentation"]:
            hps["aug_prob"] = trial.suggest_float("aug_prob", 0.1, 0.7)

            # Conditional aug_method based on ENABLE_TEXTATTACK_LIBRARY
            if ENABLE_TEXTATTACK_LIBRARY:
                hps["aug_method"] = trial.suggest_categorical(
                    "aug_method",
                    ["nlpaug_synonym", "nlpaug_contextual", "textattack_eda", "textattack_embedding"]
                )
            else:
                hps["aug_method"] = trial.suggest_categorical(
                    "aug_method",
                    ["nlpaug_synonym", "nlpaug_contextual"]
                )

        return hps

    def _train_and_evaluate_inner(
        self,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hyperparams: Dict[str, Any],
        trial: optuna.Trial,
    ) -> float:
        """Train and evaluate model on single inner fold."""
        # Create datasets
        augment_config = self._build_augment_config(hyperparams)
        train_dataset = DSM5NLIDataset(
            data=self.data.iloc[inner_train_idx].reset_index(drop=True),
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            nli_generator=self.nli_generator,
            augment_config=augment_config,
        )
        val_dataset = DSM5NLIDataset(
            data=self.data.iloc[inner_val_idx].reset_index(drop=True),
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            nli_generator=self.nli_generator,
        )

        # Create dataloaders with deterministic worker seeding
        train_loader = DataLoader(
            train_dataset,
            **self._dataloader_kwargs(True, hyperparams["batch_size"], seed_offset=1),
        )
        val_loader = DataLoader(
            val_dataset,
            **self._dataloader_kwargs(False, hyperparams["batch_size"], seed_offset=2),
        )

        # Initialize model with Flash Attention 2 and BF16 optimization
        try:
            # Attempt Flash Attention 2 (requires flash-attn package installed)
            model = DeBERTaClassifier(
                model_name=self.config.model.model_name,
                num_labels=1,
                classifier_head=hyperparams["classifier_head"],
                classifier_dropout=hyperparams["classifier_dropout"],
                hidden_dropout=hyperparams["hidden_dropout"],
                attention_dropout=hyperparams["attention_dropout"],
                loss_type=hyperparams.get("loss_type", self.config.model.get("loss_type", "bce")),
                focal_gamma=hyperparams.get("focal_gamma", self.config.model.get("loss", {}).get("focal_gamma", 2.0)),
                pos_weight=hyperparams.get("pos_weight", None),
                attn_implementation="flash_attention_2" if HAS_BF16 else "sdpa",
                torch_dtype=torch.bfloat16 if HAS_BF16 else torch.float32,
            )
        except Exception as e:
            # Fallback to SDPA or eager attention
            console.print(f"[yellow]Flash Attention 2 not available, falling back to SDPA: {e}[/yellow]")
            model = DeBERTaClassifier(
                model_name=self.config.model.model_name,
                num_labels=1,
                classifier_head=hyperparams["classifier_head"],
                classifier_dropout=hyperparams["classifier_dropout"],
                hidden_dropout=hyperparams["hidden_dropout"],
                attention_dropout=hyperparams["attention_dropout"],
                loss_type=hyperparams.get("loss_type", self.config.model.get("loss_type", "bce")),
                focal_gamma=hyperparams.get("focal_gamma", self.config.model.get("loss", {}).get("focal_gamma", 2.0)),
                pos_weight=hyperparams.get("pos_weight", None),
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if HAS_BF16 else torch.float32,
            )
        model.to(self.device)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )

        grad_accum = max(1, int(hyperparams.get("gradient_accumulation_steps", 1)))
        max_grad_norm = float(self.config.training.get("max_grad_norm", 1.0) or 0.0)

        # Training loop with early stopping
        best_val_mcc = -1.0
        patience_counter = 0
        patience = PATIENCE

        for epoch in range(hyperparams["num_epochs"]):  # Use full epochs from search space
            # Train with BF16 mixed precision
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "criterion_id"}

                # Use autocast for BF16 mixed precision
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if HAS_BF16 else torch.float32, enabled=HAS_BF16):
                    outputs = model(**batch)
                    loss = outputs["loss"] / grad_accum

                loss.backward()

                if (step + 1) % grad_accum == 0:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # Handle remaining gradients if dataset size not divisible by grad_accum
            if len(train_loader) % grad_accum != 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Validate
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if k != "criterion_id"}
                    outputs = model(**batch)
                    logits = outputs["logits"]
                    probs = torch.sigmoid(logits).squeeze(-1)
                    preds = (probs >= 0.5).long()

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch["labels"].cpu().numpy())

            # Compute MCC
            val_mcc = matthews_correlation_coefficient(val_labels, val_preds)

            # Early stopping
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Pruning
            trial.report(best_val_mcc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_mcc

    def _train_final_model(
        self,
        outer_fold: int,
        outer_train_idx: np.ndarray,
        hyperparams: Dict[str, Any],
    ):
        """Train final model on full outer training set."""
        console.print(
            f"  [yellow]Training final model with best HPs...[/yellow]"
        )

        # Create dataset
        augment_config = self._build_augment_config(hyperparams)
        train_dataset = DSM5NLIDataset(
            data=self.data.iloc[outer_train_idx].reset_index(drop=True),
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            nli_generator=self.nli_generator,
            augment_config=augment_config,
        )
        train_loader = DataLoader(
            train_dataset,
            **self._dataloader_kwargs(True, hyperparams["batch_size"], seed_offset=3),
        )

        # Initialize model with Flash Attention 2 and BF16 optimization
        try:
            # Attempt Flash Attention 2 (requires flash-attn package installed)
            model = DeBERTaClassifier(
                model_name=self.config.model.model_name,
                num_labels=1,
                classifier_head=hyperparams["classifier_head"],
                classifier_dropout=hyperparams["classifier_dropout"],
                hidden_dropout=hyperparams["hidden_dropout"],
                attention_dropout=hyperparams["attention_dropout"],
                loss_type=self.config.model.get("loss_type", "bce"),
                attn_implementation="flash_attention_2" if HAS_BF16 else "sdpa",
                torch_dtype=torch.bfloat16 if HAS_BF16 else torch.float32,
            )
        except Exception as e:
            # Fallback to SDPA or eager attention
            console.print(f"[yellow]Flash Attention 2 not available, falling back to SDPA: {e}[/yellow]")
            model = DeBERTaClassifier(
                model_name=self.config.model.model_name,
                num_labels=1,
                classifier_head=hyperparams["classifier_head"],
                classifier_dropout=hyperparams["classifier_dropout"],
                hidden_dropout=hyperparams["hidden_dropout"],
                attention_dropout=hyperparams["attention_dropout"],
                loss_type=self.config.model.get("loss_type", "bce"),
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if HAS_BF16 else torch.float32,
            )
        model.to(self.device)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )

        grad_accum = max(1, int(hyperparams.get("gradient_accumulation_steps", 1)))
        max_grad_norm = float(self.config.training.get("max_grad_norm", 1.0) or 0.0)

        # Train with full epochs from search space
        num_epochs = hyperparams.get("num_epochs", MAX_EPOCHS)
        for epoch in range(num_epochs):
            # Train with BF16 mixed precision
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "criterion_id"}

                # Use autocast for BF16 mixed precision
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if HAS_BF16 else torch.float32, enabled=HAS_BF16):
                    outputs = model(**batch)
                    loss = outputs["loss"] / grad_accum

                loss.backward()

                if (step + 1) % grad_accum == 0:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            if len(train_loader) % grad_accum != 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        console.print(f"  [green]Final model training complete[/green]")

        return model

    def _evaluate_outer_test(
        self,
        outer_fold: int,
        outer_test_idx: np.ndarray,
        model,
    ) -> Dict[str, float]:
        """Evaluate model on outer test set."""
        console.print(f"  [yellow]Evaluating on outer test set...[/yellow]")

        # Create dataset
        test_dataset = DSM5NLIDataset(
            data=self.data.iloc[outer_test_idx].reset_index(drop=True),
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            nli_generator=self.nli_generator,
        )
        num_workers = self.config.training.get("num_workers", 4)
        test_loader = DataLoader(
            test_dataset,
            **self._dataloader_kwargs(False, 32, seed_offset=4),
        )

        # Evaluate
        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "criterion_id"}
                outputs = model(**batch)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs >= 0.5).long()

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch["labels"].cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

        # Compute metrics
        metrics = compute_all_metrics(
            np.array(test_labels),
            np.array(test_preds),
            np.array(test_probs),
        )

        return metrics

    def _aggregate_results(
        self,
        outer_fold_results: List[Dict[str, float]],
        outer_fold_best_hps: List[Dict[str, Any]],
    ) -> NestedCVResults:
        """Aggregate results across outer folds."""
        if not outer_fold_results:
            return NestedCVResults(
                outer_fold_results=[],
                outer_fold_best_hps=[],
                mean_metrics={},
                std_metrics={},
                n_outer_folds=self.n_outer_splits,
                n_inner_folds=self.n_inner_splits,
                n_trials_per_fold=self.n_trials,
            )

        metric_names = list(outer_fold_results[0].keys())
        mean_metrics = {}
        std_metrics = {}

        for metric_name in metric_names:
            values = [fold_result.get(metric_name, np.nan) for fold_result in outer_fold_results]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean_metrics[metric_name] = float(np.mean(valid_values))
                std_metrics[metric_name] = float(np.std(valid_values))

        # Find global best HPs (from fold with best MCC)
        mcc_scores = [fold_result.get("mcc", -1.0) for fold_result in outer_fold_results]
        best_fold_idx = int(np.argmax(mcc_scores))
        global_best_hps = outer_fold_best_hps[best_fold_idx] if outer_fold_best_hps else None

        return NestedCVResults(
            outer_fold_results=outer_fold_results,
            outer_fold_best_hps=outer_fold_best_hps,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            global_best_hps=global_best_hps,
            n_outer_folds=self.n_outer_splits,
            n_inner_folds=self.n_inner_splits,
            n_trials_per_fold=self.n_trials,
        )
