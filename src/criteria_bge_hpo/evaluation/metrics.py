"""Custom metrics for imbalanced classification.

This module provides comprehensive metrics for evaluating binary classifiers
on imbalanced datasets, with emphasis on Matthews Correlation Coefficient (MCC)
as the primary metric.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def matthews_correlation_coefficient(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
) -> float:
    """Compute Matthews Correlation Coefficient (MCC).

    MCC is a balanced metric for binary classification that works well
    with imbalanced datasets. It takes into account all four categories
    of the confusion matrix (TP, TN, FP, FN).

    Range: [-1, 1] where:
    - 1: perfect prediction
    - 0: random prediction
    - -1: perfect inverse prediction

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        MCC score, or nan if undefined

    Example:
        >>> y_true = [0, 0, 1, 1]
        >>> y_pred = [0, 0, 1, 1]
        >>> matthews_correlation_coefficient(y_true, y_pred)
        1.0
    """
    # Convert torch tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge cases
    if len(y_true) == 0:
        return float("nan")

    # Check if all predictions are same class
    if len(np.unique(y_pred)) == 1 or len(np.unique(y_true)) == 1:
        if np.array_equal(y_true, y_pred):
            return 1.0
        return float("nan")

    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")


def balanced_accuracy(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
) -> float:
    """Compute balanced accuracy.

    Balanced accuracy is the average of recall obtained on each class.
    For binary classification: (Sensitivity + Specificity) / 2

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score

    Example:
        >>> y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        >>> y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
        >>> balanced_accuracy(y_true, y_pred)
        0.75
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute sensitivity and specificity
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)

    return (sens + spec) / 2.0


def sensitivity(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
) -> float:
    """Compute sensitivity (recall, true positive rate).

    Sensitivity = TP / (TP + FN)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Sensitivity score

    Example:
        >>> y_true = [1, 1, 1, 0]
        >>> y_pred = [1, 1, 0, 0]
        >>> sensitivity(y_true, y_pred)
        0.6666666666666666
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return float(recall_score(y_true, y_pred, zero_division=0))


def specificity(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
) -> float:
    """Compute specificity (true negative rate).

    Specificity = TN / (TN + FP)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Specificity score

    Example:
        >>> y_true = [0, 0, 0, 1]
        >>> y_pred = [0, 0, 1, 1]
        >>> specificity(y_true, y_pred)
        0.6666666666666666
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute TN and FP
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tn + fp == 0:
        return 0.0

    return float(tn / (tn + fp))


def confusion_matrix_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
) -> Dict[str, float]:
    """Compute metrics derived from confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with ppv, npv, sensitivity, specificity

    Example:
        >>> y_true = [0, 0, 1, 1]
        >>> y_pred = [0, 1, 1, 0]
        >>> metrics = confusion_matrix_metrics(y_true, y_pred)
        >>> metrics['sensitivity']
        0.5
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Positive Predictive Value (Precision)
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")

    # Negative Predictive Value
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")

    # Sensitivity (Recall)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "ppv": ppv,
        "npv": npv,
        "sensitivity": sens,
        "specificity": spec,
    }


def compute_all_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    y_prob: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute comprehensive metric suite for imbalanced classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC metrics)

    Returns:
        Dictionary with all metrics

    Example:
        >>> y_true = [0, 0, 1, 1]
        >>> y_pred = [0, 0, 1, 1]
        >>> y_prob = [0.1, 0.2, 0.8, 0.9]
        >>> metrics = compute_all_metrics(y_true, y_pred, y_prob)
        >>> metrics['mcc']
        1.0
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Initialize metrics dictionary
    metrics = {}

    # Core metrics
    metrics["mcc"] = matthews_correlation_coefficient(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy(y_true, y_pred)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    # Confusion matrix derived metrics
    cm_metrics = confusion_matrix_metrics(y_true, y_pred)
    metrics.update(cm_metrics)

    # Probability-based metrics (if provided)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            if len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                metrics["auprc"] = float(auc(recall_curve, precision_curve))
            else:
                metrics["roc_auc"] = float("nan")
                metrics["auprc"] = float("nan")
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["auprc"] = float("nan")

    return metrics
