"""Visualization utilities for terminal output.

Provides Rich-based terminal formatting for headers, tables, and summaries.
Used throughout the CLI to create visually appealing training logs.
"""

from typing import Optional

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

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
        Model             microsoft/deberta-v3-base
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
    table.add_row("TF32", str(config.reproducibility.tf32))

    console.print(table)
    console.print()


def print_fold_summary(fold_results):
    """Print K-fold cross-validation results summary.

    Displays per-fold and mean metrics in a formatted table.
    Called after all K folds complete to show final results.

    Args:
        fold_results: List of dicts, each containing:
            - "aggregate": dict with keys: f1, accuracy, precision, recall, auc
            - "per_criterion": dict with per-criterion metrics (unused here)

    Returns:
        dict: Mean metrics across all folds:
            - mean_f1: Average F1 score
            - mean_accuracy: Average accuracy
            - mean_precision: Average precision
            - mean_recall: Average recall
            - mean_auc: Average ROC-AUC

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
    table.add_column("AUC", style="red")

    # Add row for each fold
    for i, result in enumerate(fold_results):
        agg = result["aggregate"]
        table.add_row(
            str(i),
            f"{agg['f1']:.4f}",
            f"{agg['accuracy']:.4f}",
            f"{agg['precision']:.4f}",
            f"{agg['recall']:.4f}",
            f"{agg['auc']:.4f}",
        )

    # Calculate mean metrics across all folds
    import numpy as np

    mean_f1 = np.mean([r["aggregate"]["f1"] for r in fold_results])
    mean_acc = np.mean([r["aggregate"]["accuracy"] for r in fold_results])
    mean_prec = np.mean([r["aggregate"]["precision"] for r in fold_results])
    mean_recall = np.mean([r["aggregate"]["recall"] for r in fold_results])
    mean_auc = np.mean([r["aggregate"]["auc"] for r in fold_results])

    # Add mean row (bold for emphasis)
    table.add_row(
        "[bold]Mean[/bold]",
        f"[bold]{mean_f1:.4f}[/bold]",
        f"[bold]{mean_acc:.4f}[/bold]",
        f"[bold]{mean_prec:.4f}[/bold]",
        f"[bold]{mean_recall:.4f}[/bold]",
        f"[bold]{mean_auc:.4f}[/bold]",
    )

    console.print(table)
    console.print()

    # Return mean metrics for MLflow logging
    return {
        "mean_f1": mean_f1,
        "mean_accuracy": mean_acc,
        "mean_precision": mean_prec,
        "mean_recall": mean_recall,
        "mean_auc": mean_auc,
    }


class TrainingProgressDisplay:
    """Live Rich dashboard for epoch/step progress and key metrics."""

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        fold: int = 0,
        total_folds: int = 1,
        patience: Optional[int] = None,
        refresh_per_second: int = 4,
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.fold = fold
        self.total_folds = total_folds
        self.patience = patience
        self.refresh_per_second = max(1, int(refresh_per_second or 4))

        self.metrics = {
            "epoch": 0,
            "train_loss": None,
            "val_f1": None,
            "best_f1": None,
            "best_epoch": None,
            "lr": None,
            "patience_left": patience,
        }
        self.status = "Waiting for first epoch..."
        self._running_loss = 0.0
        self._step_seen = 0

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self.epoch_task = self.progress.add_task(
            f"Fold {fold + 1}/{total_folds} epochs", total=total_epochs
        )
        self.step_task = self.progress.add_task(
            "Step progress",
            total=self.steps_per_epoch,
        )
        self.live = Live(
            self._render(), console=console, refresh_per_second=self.refresh_per_second
        )

        self.progress.start_task(self.epoch_task)
        self.progress.start_task(self.step_task)

    def start(self):
        """Start rendering the live dashboard."""
        self.progress.start()
        self.live.start()
        self._refresh()

    def start_epoch(self, epoch: int, steps: int):
        """Reset step progress for a new epoch."""
        self.metrics["epoch"] = epoch
        self._running_loss = 0.0
        self._step_seen = 0

        self.steps_per_epoch = max(1, steps)
        self.progress.reset(
            self.step_task,
            total=self.steps_per_epoch,
            completed=0,
        )
        self.progress.update(
            self.step_task,
            description=f"Epoch {epoch}/{self.total_epochs}",
        )
        self.progress.start_task(self.step_task)
        self._refresh()

    def advance_step(self, loss: float):
        """Advance step progress and accumulate loss."""
        self._running_loss += loss
        self._step_seen += 1
        self.progress.advance(self.step_task)

        # Light touch refresh during the epoch to avoid excessive redraws
        if self._step_seen % 5 == 0 or self._step_seen == self.steps_per_epoch:
            self._refresh()

    def finish_epoch(
        self,
        val_f1: float,
        best_f1: float,
        best_epoch: int,
        current_lr: float,
        patience_left: Optional[int],
    ):
        """Update metrics after validation."""
        avg_loss = self._running_loss / max(1, self._step_seen)
        self.metrics.update(
            {
                "train_loss": avg_loss,
                "val_f1": val_f1,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "lr": current_lr,
                "patience_left": patience_left,
            }
        )
        self.progress.update(self.epoch_task, completed=self.metrics["epoch"])
        self.progress.update(self.step_task, completed=self.steps_per_epoch)
        self._refresh()

    def update_status(self, message: str, style: str = "cyan"):
        """Display a short status message beneath the dashboard."""
        self.status = f"[{style}]{message}[/{style}]"
        self._refresh()

    def stop(self):
        """Stop live rendering."""
        self._refresh()
        self.live.stop()
        self.progress.stop()

    def _refresh(self):
        if self.live.is_started:
            self.live.update(self._render(), refresh=True)

    def _render(self):
        header = Align.center(
            Text(
                f"Fold {self.fold + 1}/{self.total_folds} Training",
                style="bold white",
            )
        )
        status_line = Align.left(Text.from_markup(self.status))

        body = Group(
            header,
            self.progress,
            self._metrics_table(),
            status_line,
        )
        return Panel(body, border_style="cyan", padding=(1, 2))

    def _metrics_table(self):
        table = Table.grid(expand=True)
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        table.add_row(
            self._metric_cell(
                "Epoch",
                f"{self.metrics['epoch']}/{self.total_epochs}",
                "cyan",
            ),
            self._metric_cell(
                "Train Loss",
                self._fmt(self.metrics["train_loss"]),
                "yellow",
            ),
            self._metric_cell(
                "Val F1",
                self._fmt(self.metrics["val_f1"]),
                "green",
            ),
        )
        table.add_row(
            self._metric_cell("Best F1", self._format_best(), "bright_green"),
            self._metric_cell("LR", self._format_lr(), "magenta"),
            self._metric_cell("Patience", self._format_patience(), "blue"),
        )
        return table

    def _metric_cell(self, label: str, value: str, color: str):
        return Panel(
            Align.center(Text(value, justify="center")),
            title=label,
            border_style=color,
            padding=(0, 1),
        )

    @staticmethod
    def _fmt(value: Optional[float], precision: int = 4) -> str:
        if value is None:
            return "—"
        return f"{value:.{precision}f}"

    def _format_best(self) -> str:
        if self.metrics["best_f1"] is None:
            return "—"
        best_epoch = self.metrics.get("best_epoch") or 0
        return f"{self.metrics['best_f1']:.4f} @ {best_epoch}"

    def _format_lr(self) -> str:
        if self.metrics["lr"] is None:
            return "—"
        return f"{self.metrics['lr']:.2e}"

    def _format_patience(self) -> str:
        if self.patience is None:
            return "∞"
        if self.metrics["patience_left"] is None:
            return str(self.patience)
        return str(max(0, int(self.metrics["patience_left"])))
