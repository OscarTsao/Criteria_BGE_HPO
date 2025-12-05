"""Evaluator for model evaluation."""

import math
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from criteria_bge_hpo.evaluation.metrics import compute_all_metrics

console = Console()


def _format_metric(value):
    """Format metric for table display, handling NaN gracefully."""
    if value is None:
        return "N/A"

    try:
        if math.isnan(value):
            return "N/A"
    except TypeError:
        return "N/A"

    return f"{value:.4f}"


class Evaluator:
    """Evaluator for DSM-5 NLI model."""

    def __init__(
        self,
        model,
        device,
        use_bf16: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
        positive_threshold: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_bf16 = use_bf16 and torch.cuda.is_available()
        if self.amp_dtype is None and self.use_bf16:
            self.amp_dtype = torch.bfloat16
        self.use_amp = self.amp_dtype is not None and torch.cuda.is_available()
        self.non_blocking = non_blocking and torch.cuda.is_available()
        self.positive_threshold = positive_threshold
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, eval_loader, data):
        """Evaluate model."""
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {
                k: v.to(self.device, non_blocking=self.non_blocking) for k, v in batch.items()
            }

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.bfloat16,
            ):
                outputs = self.model(**batch)

            logits = outputs["logits"].float()
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs >= self.positive_threshold).long()
                prob_vector = probs
            else:
                probs = torch.softmax(logits, dim=-1).float()
                preds = torch.argmax(logits, dim=-1)
                prob_vector = probs[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(prob_vector.cpu().numpy())

        # Compute aggregate metrics using new metrics module
        all_metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        # Build metrics dict with backward compatibility
        metrics = {
            "accuracy": all_metrics["accuracy"],
            "binary_f1": all_metrics["f1"],
            "macro_f1": all_metrics["f1"],  # For binary, same as binary_f1
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "sensitivity": all_metrics["recall"],  # Alias
            "f1": all_metrics["f1"],  # Backward compat
            # New metrics
            "mcc": all_metrics["mcc"],
            "balanced_accuracy": all_metrics["balanced_accuracy"],
            "specificity": all_metrics["specificity"],
            "ppv": all_metrics["ppv"],
            "npv": all_metrics["npv"],
        }

        # Add AUC with fallback
        try:
            if "roc_auc" in all_metrics and not np.isnan(all_metrics["roc_auc"]):
                metrics["auc"] = all_metrics["roc_auc"]
                metrics["roc_auc"] = all_metrics["roc_auc"]
            else:
                metrics["auc"] = roc_auc_score(all_labels, all_probs)
                metrics["roc_auc"] = metrics["auc"]
        except ValueError:
            metrics["auc"] = float("nan")
            metrics["roc_auc"] = float("nan")

        # Add AUPRC if available
        if "auprc" in all_metrics:
            metrics["auprc"] = all_metrics["auprc"]

        # Compute per-criterion metrics
        per_criterion = evaluate_per_criterion(
            all_preds, all_labels, all_probs, data["criterion_id"].values
        )

        return {
            "aggregate": metrics,
            "per_criterion": per_criterion,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    def save_predictions(self, data, predictions, probabilities, output_path):
        """Save predictions to CSV."""

        pred_df = data.copy()
        pred_df["prediction"] = predictions
        pred_df["probability"] = probabilities
        pred_df["groundtruth"] = data["label"]

        output_df = pred_df[
            [
                "post_id",
                "post",
                "criterion_id",
                "criterion",
                "prediction",
                "groundtruth",
                "probability",
            ]
        ]

        output_df.to_csv(output_path, index=False)
        console.print(f"[green]✓[/green] Saved predictions to {output_path}")


def evaluate_per_criterion(predictions, labels, probabilities, criterion_ids):
    """Compute per-criterion metrics."""
    import numpy as np

    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    criterion_ids = np.array(criterion_ids)

    per_criterion = {}

    for criterion_id in np.unique(criterion_ids):
        mask = criterion_ids == criterion_id

        criterion_preds = predictions[mask]
        criterion_labels = labels[mask]
        criterion_probs = probabilities[mask]

        # Compute all metrics using new metrics module
        all_metrics = compute_all_metrics(criterion_labels, criterion_preds, criterion_probs)

        # Compute AUC with fallback
        try:
            auc = roc_auc_score(criterion_labels, criterion_probs)
        except ValueError:
            auc = float("nan")

        per_criterion[criterion_id] = {
            "f1": all_metrics["f1"],
            "accuracy": all_metrics["accuracy"],
            "precision": all_metrics["precision"],
            "recall": all_metrics["recall"],
            "auc": all_metrics.get("roc_auc", auc),  # Use fallback if needed
            "n_samples": mask.sum(),
            # New metrics
            "mcc": all_metrics["mcc"],
            "balanced_accuracy": all_metrics["balanced_accuracy"],
            "specificity": all_metrics["specificity"],
            "ppv": all_metrics["ppv"],
            "npv": all_metrics["npv"],
        }

    return per_criterion


def display_per_criterion_results(per_criterion):
    """Display per-criterion results in table."""
    table = Table(title="Per-Criterion Metrics")

    table.add_column("Criterion", style="cyan")
    table.add_column("Samples", style="yellow")
    table.add_column("MCC", style="bright_green")
    table.add_column("Bal Acc", style="bright_green")
    table.add_column("F1", style="green")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="blue")
    table.add_column("Recall", style="magenta")
    table.add_column("AUC", style="red")

    for criterion_id, metrics in sorted(per_criterion.items()):
        table.add_row(
            criterion_id,
            str(metrics["n_samples"]),
            _format_metric(metrics.get("mcc")),
            _format_metric(metrics.get("balanced_accuracy")),
            f"{metrics['f1']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            _format_metric(metrics.get("auc")),
        )

    console.print(table)
