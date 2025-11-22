"""Visualization utilities for terminal output.

Provides Rich-based terminal formatting for headers, tables, and summaries.
Used throughout the CLI to create visually appealing training logs.
"""

from rich.console import Console
from rich.table import Table

# Global console instance for all Rich output
console = Console()


def print_header(title: str, subtitle: str = ""):
    """Print a formatted header with optional subtitle.

    Creates a visually distinct header section using Rich formatting.

    Args:
        title: Main header text (e.g., "DSM-5 NLI K-Fold Training")
        subtitle: Optional subtitle text (e.g., "Experiment: dsm5_criteria_matching")

    Example:
        >>> print_header("Training Started", "Fold 1/5")
        ============================================================
                           Training Started
                              Fold 1/5
        ============================================================
    """
    console.print(f"\n[cyan bold]{'=' * 60}[/cyan bold]")
    console.print(f"[cyan bold]{title.center(60)}[/cyan bold]")
    if subtitle:
        console.print(f"[cyan]{subtitle.center(60)}[/cyan]")
    console.print(f"[cyan bold]{'=' * 60}[/cyan bold]\n")


def print_config_summary(config):
    """Print configuration summary table.

    Displays key hyperparameters and optimization settings in a formatted table.
    Useful for quick verification of experiment configuration at training start.

    Args:
        config: Hydra DictConfig containing all experiment settings

    Example output:
        CONFIGURATION SUMMARY
        ══════════════════════════════════════════════════════════
        Model             bert-base-uncased
        Batch Size        32
        Learning Rate     2e-05
        Epochs            10
        K-Folds           5
        BF16              True
        TF32              True
    """
    print_header("CONFIGURATION SUMMARY")

    # Create simple two-column table (key-value pairs)
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow")

    # Add key configuration parameters
    table.add_row("Model", config.model.model_name)
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Epochs", str(config.training.num_epochs))
    table.add_row("K-Folds", str(config.kfold.n_splits))
    table.add_row("BF16", str(config.training.optimization.use_bf16))
    table.add_row("TF32", str(config.training.optimization.use_tf32))

    console.print(table)
    console.print()


def print_fold_summary(fold_results):
    """Print K-fold cross-validation results summary.

    Displays per-fold and mean metrics in a formatted table.
    Called after all K folds complete to show final results.

    Args:
        fold_results: List of dicts, each containing:
            - "aggregate": dict with keys: f1, accuracy, precision, recall
            - "per_criterion": dict with per-criterion metrics (unused here)

    Returns:
        dict: Mean metrics across all folds:
            - mean_f1: Average F1 score
            - mean_accuracy: Average accuracy
            - mean_precision: Average precision
            - mean_recall: Average recall

    Example output:
        K-FOLD CROSS-VALIDATION RESULTS
        ══════════════════════════════════════════════════════════
        Fold    F1      Accuracy  Precision  Recall
        0       0.8523  0.8712    0.8456     0.8591
        1       0.8612  0.8801    0.8534     0.8691
        ...
        Mean    0.8567  0.8756    0.8495     0.8641
    """
    print_header("K-FOLD CROSS-VALIDATION RESULTS")

    # Create table with fold metrics
    table = Table()
    table.add_column("Fold", style="cyan")
    table.add_column("F1", style="green")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Precision", style="blue")
    table.add_column("Recall", style="magenta")

    # Add row for each fold
    for i, result in enumerate(fold_results):
        agg = result["aggregate"]
        table.add_row(
            str(i),
            f"{agg['f1']:.4f}",
            f"{agg['accuracy']:.4f}",
            f"{agg['precision']:.4f}",
            f"{agg['recall']:.4f}",
        )

    # Calculate mean metrics across all folds
    import numpy as np
    mean_f1 = np.mean([r["aggregate"]["f1"] for r in fold_results])
    mean_acc = np.mean([r["aggregate"]["accuracy"] for r in fold_results])
    mean_prec = np.mean([r["aggregate"]["precision"] for r in fold_results])
    mean_recall = np.mean([r["aggregate"]["recall"] for r in fold_results])

    # Add mean row (bold for emphasis)
    table.add_row(
        "[bold]Mean[/bold]",
        f"[bold]{mean_f1:.4f}[/bold]",
        f"[bold]{mean_acc:.4f}[/bold]",
        f"[bold]{mean_prec:.4f}[/bold]",
        f"[bold]{mean_recall:.4f}[/bold]",
    )

    console.print(table)
    console.print()

    # Return mean metrics for MLflow logging
    return {
        "mean_f1": mean_f1,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_prec,
        "mean_recall": mean_recall,
    }
