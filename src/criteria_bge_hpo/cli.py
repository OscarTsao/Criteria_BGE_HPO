"""
Command-line interface for DSM-5 NLI Binary Classification.

Usage:
    python -m criteria_bge_hpo.cli train training.num_epochs=100 training.early_stopping_patience=20  # Run K-fold training
    python -m criteria_bge_hpo.cli hpo --n-trials 100       # Run HPO
    python -m criteria_bge_hpo.cli eval --fold 0            # Evaluate specific fold
"""

import copy
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import mlflow
import optuna
from optuna.pruners import MedianPruner
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from criteria_bge_hpo.data.preprocessing import load_and_preprocess_data
from criteria_bge_hpo.data.dataset import DSM5NLIDataset, create_dataloaders
from criteria_bge_hpo.models.bert_classifier import BERTClassifier
from criteria_bge_hpo.models.deberta_classifier import DeBERTaClassifier
from criteria_bge_hpo.training.kfold import (
    create_kfold_splits,
    get_fold_statistics,
    display_fold_statistics,
)
from criteria_bge_hpo.training.trainer import Trainer, create_optimizer_and_scheduler
from criteria_bge_hpo.evaluation.evaluator import (
    Evaluator,
    display_per_criterion_results,
)
from criteria_bge_hpo.utils.reproducibility import (
    set_seed,
    enable_deterministic,
    get_device,
    verify_cuda_setup,
)
from criteria_bge_hpo.utils.optimizations import (
    configure_sdp_backend,
    resolve_attention_backend,
    select_amp_dtype,
    should_use_grad_scaler,
)
from criteria_bge_hpo.utils.mlflow_setup import setup_mlflow, log_config, start_run
from criteria_bge_hpo.utils.visualization import (
    print_header,
    print_config_summary,
    print_fold_summary,
)

console = Console()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_optuna_storage(storage_uri: str) -> str:
    """
    Ensure Optuna storage URIs using SQLite are absolute paths under Hydra's runtime cwd.

    Args:
        storage_uri: Storage URI from config (e.g., sqlite:///optuna.db)

    Returns:
        Absolute storage URI for local SQLite or the original URI for remote DBs.
    """
    if storage_uri.startswith("sqlite:///") and not storage_uri.startswith("sqlite:////"):
        rel_path = storage_uri[len("sqlite:///") :]
        abs_path = hydra.utils.to_absolute_path(rel_path)
        return f"sqlite:///{abs_path}"
    return storage_uri


def _build_optimization_settings(config: DictConfig) -> Dict[str, Any]:
    """Assemble optimization switches (AMP, attention backend, etc.)."""
    opt_cfg = config.training.optimization
    amp_dtype = select_amp_dtype(
        use_bf16=opt_cfg.use_bf16,
        enable_fp16=opt_cfg.get("enable_fp16_fallback", True),
    )
    attn_backend = resolve_attention_backend(opt_cfg.get("attention_backend", "sdpa"))
    configure_sdp_backend(attn_backend)
    use_compile = False  # torch.compile explicitly excluded

    return {
        "amp_dtype": amp_dtype,
        "use_grad_scaler": should_use_grad_scaler(amp_dtype),
        "attention_backend": attn_backend,
        "gradient_checkpointing": opt_cfg.get("gradient_checkpointing", False),
        "optimizer": opt_cfg.get("optimizer", "adamw_torch_fused"),
        "non_blocking": config.training.pin_memory and torch.cuda.is_available(),
        "use_compile": use_compile,
    }


def run_single_fold(
    config: DictConfig,
    pairs_df,
    train_idx,
    val_idx,
    fold: int,
    tokenizer,
    device,
):
    """
    Train and evaluate a single fold.

    Args:
        config: Hydra configuration
        pairs_df: Full dataset
        train_idx: Training indices
        val_idx: Validation indices
        fold: Fold number
        tokenizer: HuggingFace tokenizer
        device: Device to train on

    Returns:
        Dictionary of validation metrics
    """
    console.print(f"\n[bold cyan]Fold {fold + 1}/{config.kfold.n_splits}[/bold cyan]\n")

    augment_config = config.get("augmentation", None)
    optimization_settings = _build_optimization_settings(config)

    # Create datasets
    train_dataset = DSM5NLIDataset(
        pairs_df.iloc[train_idx],
        tokenizer,
        max_length=config.data.max_length,
        verify_format=(fold == 0),  # Verify format only for first fold
        model_name=config.model.model_name,
        augment_config=augment_config,
    )

    val_dataset = DSM5NLIDataset(
        pairs_df.iloc[val_idx],
        tokenizer,
        max_length=config.data.max_length,
        model_name=config.model.model_name,
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        bucket_by_length=config.training.bucket_by_length,
        bucket_size=config.training.bucket_size,
        seed=config.seed,
    )

    # Create model
    # Detect model type based on model_name
    model_name_lower = config.model.model_name.lower()
    if "deberta" in model_name_lower:
        model = DeBERTaClassifier(
            model_name=config.model.model_name,
            num_labels=config.model.num_labels,
            classifier_head=config.model.get("classifier_head", "linear"),
            classifier_dropout=config.model.get("classifier_dropout", 0.1),
            hidden_dropout=config.model.get("hidden_dropout", None),
            attention_dropout=config.model.get("attention_dropout", None),
            freeze_backbone=config.model.freeze_backbone,
            loss_type=config.model.get("loss", {}).get("type", "bce"),
            focal_gamma=config.model.get("loss", {}).get("focal_gamma", 2.0),
            focal_alpha=config.model.get("loss", {}).get("focal_alpha", None),
            attn_implementation=optimization_settings["attention_backend"],
            gradient_checkpointing=optimization_settings["gradient_checkpointing"],
            torch_dtype=optimization_settings["amp_dtype"],
        )
    else:
        model = BERTClassifier(
            model_name=config.model.model_name,
            num_labels=config.model.num_labels,
            freeze_backbone=config.model.freeze_backbone,
            attn_implementation=optimization_settings["attention_backend"],
            gradient_checkpointing=optimization_settings["gradient_checkpointing"],
            torch_dtype=optimization_settings["amp_dtype"],
        )

    console.print(
        f"Model parameters: {model.get_num_trainable_params():,} trainable / {model.get_num_total_params():,} total"
    )

    logging_cfg = config.training.get("logging", {}) or {}
    enable_progress = logging_cfg.get("rich_progress", True)
    refresh_rate = logging_cfg.get("progress_refresh_per_second", 4)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_loader,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        use_fused=config.training.optimization.fused_adamw,
        optimizer_type=optimization_settings["optimizer"],
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_bf16=config.training.optimization.use_bf16,
        amp_dtype=optimization_settings["amp_dtype"],
        use_grad_scaler=optimization_settings["use_grad_scaler"],
        use_compile=optimization_settings["use_compile"],
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        non_blocking=optimization_settings["non_blocking"],
        mlflow_enabled=True,
        early_stopping_patience=config.training.early_stopping_patience,
        checkpoint_dir=hydra.utils.to_absolute_path(config.checkpoint_dir),
        positive_threshold=config.model.positive_threshold,
        enable_progress=enable_progress,
        total_folds=config.kfold.n_splits,
        progress_refresh_per_second=refresh_rate,
    )

    # Train
    trainer.train(num_epochs=config.training.num_epochs, fold=fold)

    # Evaluate trained model on validation set for aggregate metrics
    evaluator = Evaluator(
        model=trainer.model,
        device=device,
        use_bf16=config.training.optimization.use_bf16,
        amp_dtype=optimization_settings["amp_dtype"],
        non_blocking=optimization_settings["non_blocking"],
        positive_threshold=config.model.positive_threshold,
    )
    val_data = pairs_df.iloc[val_idx].reset_index(drop=True)
    eval_results = evaluator.evaluate(val_loader, val_data)

    # Log aggregate metrics to MLflow if enabled
    if mlflow.active_run():
        for metric_name, metric_value in eval_results["aggregate"].items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)

    # Get best metrics
    best_metrics = {
        "val_f1": trainer.best_val_f1,
        "fold": fold,
        "aggregate": eval_results["aggregate"],
        "per_criterion": eval_results["per_criterion"],
    }

    return best_metrics


def run_kfold_training(config: DictConfig):
    """
    Run complete K-fold cross-validation training.

    Args:
        config: Hydra configuration
    """
    print_header("DSM-5 NLI K-Fold Training", f"Experiment: {config.experiment_name}")

    # Set up reproducibility
    set_seed(config.seed)
    enable_deterministic(config.reproducibility.deterministic, config.reproducibility.tf32)

    # Verify CUDA
    verify_cuda_setup()
    device = get_device()

    # Print configuration
    print_config_summary(config)

    # Load data
    pairs_df = load_and_preprocess_data(config)

    # Create K-fold splits (store to reuse for stats and training)
    splits = list(create_kfold_splits(pairs_df, n_splits=config.kfold.n_splits))

    # Display split statistics
    stats_df = get_fold_statistics(pairs_df, splits)
    display_fold_statistics(stats_df)

    # Set up MLflow
    setup_mlflow(config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    console.print(f"[green]✓[/green] Loaded tokenizer: {config.model.model_name}\n")

    # Train each fold
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        with start_run(run_name=f"fold_{fold}", tags={"fold": str(fold)}):
            # Log config
            log_config(config)

            # Train fold
            metrics = run_single_fold(
                config,
                pairs_df,
                train_idx,
                val_idx,
                fold,
                tokenizer,
                device,
            )

            fold_results.append(metrics)

    # Print summary
    console.print("\n")
    mean_metrics = print_fold_summary(fold_results)

    # Log overall results
    with start_run(run_name="overall_results", tags={"type": "summary"}):
        for key, value in mean_metrics.items():
            mlflow.log_metric(key, value)

    console.print("\n[green]✓[/green] K-fold training complete!")
    console.print(
        f"Mean F1: {mean_metrics['mean_f1']:.4f} "
        f"± {np.std([r['val_f1'] for r in fold_results]):.4f}"
    )
    console.print(
        f"Mean AUC: {mean_metrics['mean_auc']:.4f} "
        f"± {np.std([r['aggregate']['auc'] for r in fold_results]):.4f}\n"
    )


def get_study_state(config: DictConfig) -> Dict:
    """
    Get current state of the Optuna study.

    Args:
        config: Hydra configuration

    Returns:
        Dictionary with study state information
    """
    storage = _resolve_optuna_storage(config.hpo.storage)
    try:
        study = optuna.load_study(
            study_name=config.hpo.study_name,
            storage=storage,
        )

        # Count trials by state
        trials = study.trials
        completed = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
        running = sum(1 for t in trials if t.state == optuna.trial.TrialState.RUNNING)
        pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
        failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
        waiting = sum(1 for t in trials if t.state == optuna.trial.TrialState.WAITING)
        total = len(trials)

        best_value = study.best_value if completed > 0 else None
        best_trial = study.best_trial.number if completed > 0 else None

        return {
            "exists": True,
            "total_trials": total,
            "completed": completed,
            "running": running,
            "pruned": pruned,
            "failed": failed,
            "waiting": waiting,
            "best_value": best_value,
            "best_trial": best_trial,
            "study_name": config.hpo.study_name,
        }
    except KeyError:
        # Study doesn't exist yet
        return {
            "exists": False,
            "total_trials": 0,
            "completed": 0,
            "running": 0,
            "pruned": 0,
            "failed": 0,
            "waiting": 0,
            "best_value": None,
            "best_trial": None,
            "study_name": config.hpo.study_name,
        }


def display_study_state(state: Dict):
    """Display study state in a formatted table."""
    table = Table(
        title=f"Study: {state['study_name']}", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Study Exists", "✓ Yes" if state["exists"] else "✗ No")
    table.add_row("Total Trials", str(state["total_trials"]))
    table.add_row("Completed", str(state["completed"]))
    table.add_row("Running", str(state["running"]))
    table.add_row("Pruned", str(state["pruned"]))
    table.add_row("Failed", str(state["failed"]))
    table.add_row("Waiting", str(state["waiting"]))

    if state["best_value"] is not None:
        table.add_row("Best F1", f"{state['best_value']:.4f}")
        table.add_row("Best Trial", f"#{state['best_trial']}")

    console.print("\n")
    console.print(table)
    console.print("\n")


def run_hpo_worker(config: DictConfig, n_trials: int, worker_id: Optional[int] = None):
    """
    Run a single HPO worker process.

    This is the same as run_hpo but designed to be called from parallel processes.
    Each worker will coordinate through the shared Optuna storage backend.

    Args:
        config: Hydra configuration
        n_trials: Number of trials for this worker to run
        worker_id: Optional worker identifier for logging
    """
    worker_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""
    print_header(f"{worker_prefix}DSM-5 NLI HPO Worker", f"Trials: {n_trials}")

    # Set up reproducibility
    set_seed(config.seed)
    enable_deterministic(config.reproducibility.deterministic, config.reproducibility.tf32)
    device = get_device()

    # Load data
    pairs_df = load_and_preprocess_data(config)

    # Create K-fold splits (will be reused across trials)
    splits = list(create_kfold_splits(pairs_df, n_splits=config.kfold.n_splits))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Set up MLflow
    setup_mlflow(config)
    optimization_settings = _build_optimization_settings(config)
    logging_cfg = config.training.get("logging", {}) or {}

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Samples hyperparameters and evaluates via K-fold CV.
        """
        lr_space = config.hpo.search_space.learning_rate
        wd_space = config.hpo.search_space.weight_decay

        lr_type = str(lr_space.get("type", "loguniform")).lower()
        wd_type = str(wd_space.get("type", "loguniform")).lower()

        lr_log = lr_type.startswith("log")
        wd_log = wd_type.startswith("log")

        # Avoid Optuna errors when a log-uniform space has non-positive lower bound
        if wd_log and wd_space.low <= 0:
            wd_log = False
            console.print(
                f"{worker_prefix}[yellow]• weight_decay.low<=0 detected; using linear sampling[/yellow]"
            )

        # Sample hyperparameters (using Optuna 3.x API)
        learning_rate = trial.suggest_float(
            "learning_rate",
            lr_space.low,
            lr_space.high,
            log=lr_log,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            config.hpo.search_space.batch_size.choices,
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            wd_space.low,
            wd_space.high,
            log=wd_log,
        )

        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            config.hpo.search_space.warmup_ratio.low,
            config.hpo.search_space.warmup_ratio.high,
        )

        # Sample architecture hyperparameters (if configured)
        classifier_head = None
        head_space = config.hpo.search_space.get("classifier_head", None)
        if head_space is not None:
            classifier_head = trial.suggest_categorical(
                "classifier_head",
                head_space.choices,
            )

        # Sample dropout hyperparameters (if configured)
        classifier_dropout = None
        cd_space = config.hpo.search_space.get("classifier_dropout", None)
        if cd_space is not None:
            classifier_dropout = trial.suggest_float(
                "classifier_dropout",
                cd_space.low,
                cd_space.high,
            )

        hidden_dropout = None
        hd_space = config.hpo.search_space.get("hidden_dropout", None)
        if hd_space is not None:
            hidden_dropout = trial.suggest_float(
                "hidden_dropout",
                hd_space.low,
                hd_space.high,
            )

        attention_dropout = None
        ad_space = config.hpo.search_space.get("attention_dropout", None)
        if ad_space is not None:
            attention_dropout = trial.suggest_float(
                "attention_dropout",
                ad_space.low,
                ad_space.high,
            )

        # Sample loss configuration (if configured)
        loss_type = None
        loss_space = config.hpo.search_space.get("loss_type", None)
        if loss_space is not None:
            loss_type = trial.suggest_categorical(
                "loss_type",
                loss_space.choices,
            )

        focal_gamma = None
        fg_space = config.hpo.search_space.get("focal_gamma", None)
        if fg_space is not None and (loss_type == "focal" or loss_type is None):
            focal_gamma = trial.suggest_float(
                "focal_gamma",
                fg_space.low,
                fg_space.high,
            )

        pos_weight = None
        pos_space = config.hpo.search_space.get("pos_weight", None)
        if pos_space is not None and (loss_type == "weighted_bce" or loss_type is None):
            pos_weight = trial.suggest_float(
                "pos_weight",
                pos_space.low,
                pos_space.high,
            )

        # Sample gradient accumulation steps (if configured)
        gradient_accumulation_steps = config.training.gradient_accumulation_steps
        ga_space = config.hpo.search_space.get("gradient_accumulation_steps", None)
        if ga_space is not None:
            gradient_accumulation_steps = trial.suggest_categorical(
                "gradient_accumulation_steps",
                ga_space.choices,
            )

        augmentation_cfg = copy.deepcopy(config.get("augmentation", None))
        aug_enable_space = config.hpo.search_space.get("aug_enable", None)
        if augmentation_cfg is not None and aug_enable_space is not None:
            aug_enable = trial.suggest_categorical("aug_enable", aug_enable_space.choices)
            if aug_enable:
                aug_prob_space = config.hpo.search_space.aug_prob
                aug_method_space = config.hpo.search_space.aug_method
                augmentation_cfg.enable = True
                augmentation_cfg.prob = trial.suggest_float(
                    "aug_prob",
                    aug_prob_space.low,
                    aug_prob_space.high,
                )
                augmentation_cfg.type = trial.suggest_categorical(
                    "aug_method", aug_method_space.choices
                )
                imbalance_mode_space = config.hpo.search_space.get("imbalance_mode", None)
                if imbalance_mode_space is not None:
                    augmentation_cfg.imbalance_mode = trial.suggest_categorical(
                        "imbalance_mode",
                        imbalance_mode_space.choices,
                    )
            else:
                augmentation_cfg.enable = False
        else:
            augmentation_cfg = None

        num_epochs = config.training.num_epochs
        epoch_space = config.hpo.search_space.get("epochs", None)
        if epoch_space is not None:
            num_epochs = trial.suggest_categorical(
                "epochs",
                epoch_space.choices,
            )

        console.print(f"\n{worker_prefix}[bold magenta]Trial {trial.number}[/bold magenta]")
        console.print(f"  LR: {learning_rate:.2e}, BS: {batch_size}")

        # Run K-fold CV with sampled hyperparameters
        fold_scores = []

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            # Log trial parameters
            mlflow.log_params(trial.params)

            for fold, (train_idx, val_idx) in enumerate(splits):
                # Create datasets
                train_dataset = DSM5NLIDataset(
                    pairs_df.iloc[train_idx],
                    tokenizer,
                    max_length=config.data.max_length,
                    model_name=config.model.model_name,
                    augment_config=augmentation_cfg,
                )

                val_dataset = DSM5NLIDataset(
                    pairs_df.iloc[val_idx],
                    tokenizer,
                    max_length=config.data.max_length,
                    model_name=config.model.model_name,
                )

                # Create dataloaders with trial batch size
                train_loader, val_loader = create_dataloaders(
                    train_dataset,
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=config.training.num_workers,
                    pin_memory=config.training.pin_memory,
                    persistent_workers=config.training.persistent_workers,
                    bucket_by_length=config.training.bucket_by_length,
                    bucket_size=config.training.bucket_size,
                    seed=config.seed,
                )

                # Create model with sampled configuration
                # Detect model type based on model_name
                model_name_lower = config.model.model_name.lower()
                if "deberta" in model_name_lower:
                    model = DeBERTaClassifier(
                        model_name=config.model.model_name,
                        num_labels=config.model.num_labels,
                        classifier_head=classifier_head
                        or config.model.get("classifier_head", "linear"),
                        classifier_dropout=classifier_dropout
                        or config.model.get("classifier_dropout", 0.1),
                        hidden_dropout=hidden_dropout
                        or config.model.get("hidden_dropout", None),
                        attention_dropout=attention_dropout
                        or config.model.get("attention_dropout", None),
                        freeze_backbone=config.model.freeze_backbone,
                        loss_type=loss_type or config.model.get("loss", {}).get("type", "bce"),
                        focal_gamma=focal_gamma
                        or config.model.get("loss", {}).get("focal_gamma", 2.0),
                        focal_alpha=config.model.get("loss", {}).get("focal_alpha", None),
                        pos_weight=pos_weight,
                        attn_implementation=optimization_settings["attention_backend"],
                        gradient_checkpointing=optimization_settings["gradient_checkpointing"],
                        torch_dtype=optimization_settings["amp_dtype"],
                    )
                else:
                    model = BERTClassifier(
                        model_name=config.model.model_name,
                        num_labels=config.model.num_labels,
                        freeze_backbone=config.model.freeze_backbone,
                        attn_implementation=optimization_settings["attention_backend"],
                        gradient_checkpointing=optimization_settings["gradient_checkpointing"],
                        torch_dtype=optimization_settings["amp_dtype"],
                    )

                # Create optimizer with trial hyperparameters
                optimizer, scheduler = create_optimizer_and_scheduler(
                    model,
                    train_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    use_fused=config.training.optimization.fused_adamw,
                    optimizer_type=optimization_settings["optimizer"],
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )

                # Create trainer
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    use_bf16=config.training.optimization.use_bf16,
                    amp_dtype=optimization_settings["amp_dtype"],
                    use_grad_scaler=optimization_settings["use_grad_scaler"],
                    use_compile=optimization_settings["use_compile"],
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=config.training.max_grad_norm,
                    non_blocking=optimization_settings["non_blocking"],
                    mlflow_enabled=False,  # Disable per-step logging during HPO
                    early_stopping_patience=config.training.early_stopping_patience,
                    positive_threshold=config.model.positive_threshold,
                    enable_progress=logging_cfg.get("hpo_progress", False),
                    total_folds=config.kfold.n_splits,
                    progress_refresh_per_second=logging_cfg.get(
                        "progress_refresh_per_second", 4
                    ),
                )

                # Train with configured epochs (HPO will control runtime via trials/early stopping)
                trainer.train(num_epochs=num_epochs, fold=fold)

                # Get best F1
                fold_f1 = trainer.best_val_f1
                fold_scores.append(fold_f1)

                # Report intermediate value for pruning
                trial.report(fold_f1, fold)

                # Prune if unpromising
                if trial.should_prune():
                    console.print(
                        f"{worker_prefix}[yellow]⚠[/yellow] Trial {trial.number} pruned at fold {fold}"
                    )
                    raise optuna.TrialPruned()

            # Calculate mean F1 across folds
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)

            # Log results
            mlflow.log_metric("mean_f1", mean_f1)
            mlflow.log_metric("std_f1", std_f1)

            console.print(f"{worker_prefix}  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

        return mean_f1

    # Create Optuna study (load existing if available)
    pruner_cfg = config.hpo.pruner
    pruner_type = str(getattr(pruner_cfg, "type", "MedianPruner")).lower()
    if pruner_type in {"hyperbandpruner", "hyperband"}:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=getattr(pruner_cfg, "min_resource", 1),
            max_resource=getattr(pruner_cfg, "max_resource", config.training.num_epochs),
            reduction_factor=getattr(pruner_cfg, "reduction_factor", 3),
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=getattr(pruner_cfg, "n_startup_trials", 5),
            n_warmup_steps=getattr(pruner_cfg, "n_warmup_steps", 3),
            interval_steps=getattr(pruner_cfg, "interval_steps", 1),
        )

    storage = _resolve_optuna_storage(config.hpo.storage)
    study = optuna.create_study(
        study_name=config.hpo.study_name,
        storage=storage,
        direction=config.hpo.direction,
        pruner=pruner,
        load_if_exists=True,
    )

    # Run optimization (this worker will coordinate with others via storage)
    study.optimize(objective, n_trials=n_trials)

    console.print(f"\n{worker_prefix}[bold green]Worker Complete![/bold green]\n")


def run_hpo(config: DictConfig, n_trials: int):
    """
    Run Optuna hyperparameter optimization (single process).

    Args:
        config: Hydra configuration
        n_trials: Number of trials to run
    """
    run_hpo_worker(config, n_trials, worker_id=None)

    # Print final results
    storage = _resolve_optuna_storage(config.hpo.storage)
    study = optuna.load_study(
        study_name=config.hpo.study_name,
        storage=storage,
    )

    console.print("\n[bold green]HPO Complete![/bold green]\n")
    console.print(f"Best trial: {study.best_trial.number}")
    console.print(f"Best F1: {study.best_value:.4f}")
    console.print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        console.print(f"  {key}: {value}")

    # Log best trial to MLflow
    with start_run(run_name="best_trial", tags={"type": "hpo_best"}):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)


def run_hpo_parallel(
    config: DictConfig,
    n_workers: int,
    n_trials_per_worker: Optional[int] = None,
    max_total_trials: Optional[int] = None,
):
    """
    Launch parallel HPO workers that coordinate through shared Optuna storage.

    Each worker runs as a separate process and coordinates trial execution
    through the shared storage backend (SQLite/PostgreSQL/MySQL).

    Args:
        config: Hydra configuration
        n_workers: Number of parallel worker processes to launch
        n_trials_per_worker: Number of trials each worker should run (None = run until max_total_trials)
        max_total_trials: Maximum total trials across all workers (None = unlimited)
    """
    print_header("DSM-5 NLI Parallel HPO", f"Workers: {n_workers}")

    # Check current study state
    state = get_study_state(config)
    display_study_state(state)

    # Warn about SQLite limitations
    if config.hpo.storage.startswith("sqlite"):
        console.print(
            "[yellow]⚠[/yellow] Warning: SQLite has limited concurrent write support.\n"
            "For better parallel performance, consider using PostgreSQL:\n"
            "  hpo.storage=postgresql://user:pass@host/db\n"
        )

    # Calculate trials per worker
    if n_trials_per_worker is None:
        if max_total_trials is None:
            n_trials_per_worker = (
                config.hpo.n_trials // n_workers if n_workers > 0 else config.hpo.n_trials
            )
        else:
            remaining = max(0, max_total_trials - state["completed"])
            n_trials_per_worker = remaining // n_workers if n_workers > 0 else remaining
            console.print(
                f"[cyan]Target: {max_total_trials} total trials, {remaining} remaining[/cyan]\n"
            )

    console.print(
        f"[green]Launching {n_workers} workers, {n_trials_per_worker} trials each[/green]\n"
    )

    # Launch worker processes
    processes = []
    for worker_id in range(n_workers):
        # Build command to run worker
        cmd = [
            sys.executable,
            "-m",
            "criteria_bge_hpo.cli",
            "command=hpo_worker",
            f"hpo_worker.worker_id={worker_id}",
            f"hpo_worker.n_trials={n_trials_per_worker}",
        ]

        # Add config overrides if needed
        if max_total_trials is not None:
            cmd.append(f"hpo_worker.max_total_trials={max_total_trials}")

        console.print(f"[cyan]Starting worker {worker_id}...[/cyan]")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((worker_id, process))

    console.print(f"\n[green]✓[/green] Launched {n_workers} workers\n")
    console.print("[yellow]Workers are running in background. Monitor progress with:[/yellow]")
    console.print("  python -m criteria_bge_hpo.cli command=hpo_status\n")

    return processes


def run_hpo_status(config: DictConfig):
    """Display current HPO study status."""
    print_header("HPO Study Status", "")
    state = get_study_state(config)
    display_study_state(state)

    if state["exists"] and state["completed"] > 0:
        # Load study to show best params
        storage = _resolve_optuna_storage(config.hpo.storage)
        study = optuna.load_study(
            study_name=config.hpo.study_name,
            storage=storage,
        )
        console.print("\n[bold]Best Hyperparameters:[/bold]")
        for key, value in study.best_params.items():
            console.print(f"  {key}: {value}")


def run_eval(config: DictConfig, fold: int = 0):
    """
    Run evaluation for a specific fold using saved checkpoint.

    Args:
        config: Hydra configuration
        fold: Fold number to evaluate
    """
    print_header("DSM-5 NLI Evaluation", f"Fold {fold}")

    set_seed(config.seed)
    device = get_device()
    optimization_settings = _build_optimization_settings(config)

    # Load data
    pairs_df = load_and_preprocess_data(config)

    # Create K-fold splits
    splits = list(create_kfold_splits(pairs_df, n_splits=config.kfold.n_splits))

    if fold >= len(splits):
        console.print(f"[red]Error:[/red] Fold {fold} out of range (0-{len(splits)-1})")
        sys.exit(1)

    _, val_idx = splits[fold]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create validation dataset
    val_dataset = DSM5NLIDataset(
        pairs_df.iloc[val_idx],
        tokenizer,
        max_length=config.data.max_length,
        model_name=config.model.model_name,
    )

    # Create dataloader
    persistent = config.training.persistent_workers and config.training.num_workers > 0
    use_pin_memory = config.training.pin_memory and torch.cuda.is_available()
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent,
    )

    # Load model from checkpoint
    checkpoint_path = os.path.join(
        hydra.utils.to_absolute_path(config.checkpoint_dir), f"fold_{fold}"
    )

    console.print(f"Loading model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        console.print(f"[red]Error:[/red] Checkpoint not found at {checkpoint_path}")
        console.print("Run training first: python -m criteria_bge_hpo.cli command=train")
        sys.exit(1)

    model = BERTClassifier.from_pretrained(
        checkpoint_path,
        attn_implementation=optimization_settings["attention_backend"],
        gradient_checkpointing=optimization_settings["gradient_checkpointing"],
        torch_dtype=optimization_settings["amp_dtype"],
    )
    model.to(device)

    # Evaluate
    evaluator = Evaluator(
        model=model,
        device=device,
        use_bf16=config.training.optimization.use_bf16,
        amp_dtype=optimization_settings["amp_dtype"],
        non_blocking=optimization_settings["non_blocking"],
        positive_threshold=config.model.positive_threshold,
    )

    val_data = pairs_df.iloc[val_idx].reset_index(drop=True)
    eval_results = evaluator.evaluate(val_loader, val_data)

    # Display results
    console.print("\n[bold green]Evaluation Results:[/bold green]")
    for metric, value in eval_results["aggregate"].items():
        console.print(f"  {metric}: {value:.4f}")

    display_per_criterion_results(eval_results["per_criterion"])


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    # Get command from Hydra config (set via command=train, command=hpo, etc.)
    command = config.get("command", None)

    if command is None:
        console.print(
            "[red]Error:[/red] No command specified. Use command=train, command=hpo, or command=eval"
        )
        console.print("\nExamples:")
        console.print(
            "  python -m criteria_bge_hpo.cli command=train training.num_epochs=100 training.early_stopping_patience=20"
        )
        console.print("  python -m criteria_bge_hpo.cli command=hpo n_trials=100")
        console.print("  python -m criteria_bge_hpo.cli command=eval fold=0")
        sys.exit(1)

    if command == "train":
        run_kfold_training(config)
    elif command == "hpo":
        n_trials = config.get("n_trials", 100)
        run_hpo(config, n_trials)
    elif command == "hpo_parallel":
        n_workers = config.get("n_workers", 2)
        n_trials_per_worker = config.get("n_trials_per_worker", None)
        max_total_trials = config.get("max_total_trials", None)
        run_hpo_parallel(config, n_workers, n_trials_per_worker, max_total_trials)
    elif command == "hpo_worker":
        # Internal command for parallel workers
        worker_config = config.get("hpo_worker", {})
        worker_id = worker_config.get("worker_id", None)
        n_trials = worker_config.get("n_trials", 100)
        max_total_trials = worker_config.get("max_total_trials", None)

        # Run worker with optional max_total_trials limit
        if max_total_trials is not None:
            state = get_study_state(config)
            remaining = max(0, max_total_trials - state["completed"])
            n_trials = min(n_trials, remaining)
            if n_trials <= 0:
                console.print(f"[yellow]Worker {worker_id}: Target reached, exiting[/yellow]")
                return

        run_hpo_worker(config, n_trials, worker_id=worker_id)
    elif command == "hpo_status":
        run_hpo_status(config)
    elif command == "eval":
        fold = config.get("fold", 0)
        run_eval(config, fold)
    else:
        console.print(f"[red]Error:[/red] Unknown command '{command}'")
        console.print("Available commands: train, hpo, hpo_parallel, hpo_status, eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
