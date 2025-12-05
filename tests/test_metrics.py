"""Unit tests for evaluation metrics.

Tests all functions in criteria_bge_hpo.evaluation.metrics including:
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- Sensitivity and Specificity
- Confusion Matrix Metrics
- Comprehensive metric computation
"""

import numpy as np
import pytest
import torch

from criteria_bge_hpo.evaluation.metrics import (
    balanced_accuracy,
    compute_all_metrics,
    confusion_matrix_metrics,
    matthews_correlation_coefficient,
    sensitivity,
    specificity,
)


class TestMatthewsCorrelationCoefficient:
    """Test Matthews Correlation Coefficient computation."""

    def test_mcc_perfect_classifier(self) -> None:
        """Perfect predictions should give MCC = 1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert mcc == pytest.approx(1.0), "Perfect predictions should yield MCC = 1.0"

    def test_mcc_random_classifier(self) -> None:
        """Random predictions should give MCC approximately 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=1000)
        y_pred = np.random.randint(0, 2, size=1000)
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert abs(mcc) < 0.15, "Random predictions should yield MCC close to 0"

    def test_mcc_inverse_classifier(self) -> None:
        """Inverse predictions should give MCC approximately -1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert mcc == pytest.approx(-1.0), "Inverse predictions should yield MCC = -1.0"

    def test_mcc_imbalanced_data(self) -> None:
        """Test with imbalanced class distribution."""
        # 90% negative, 10% positive
        y_true = np.array([0] * 90 + [1] * 10)
        # Classifier gets 80% of positives, 95% of negatives
        y_pred = np.array([0] * 85 + [1] * 8 + [0] * 2 + [1] * 5)
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        # MCC should be positive but less than 1.0
        assert 0 < mcc < 1.0, "MCC should reflect good but imperfect performance"
        assert isinstance(mcc, float), "MCC should return a float"

    def test_mcc_all_one_class_prediction(self) -> None:
        """Should return nan for single-class predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All predict negative
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert np.isnan(mcc), "MCC should be nan when predictions are all one class"

    def test_mcc_empty_input(self) -> None:
        """Should return nan for empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert np.isnan(mcc), "MCC should be nan for empty input"

    def test_mcc_with_torch_tensors(self) -> None:
        """Test with PyTorch tensors."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_pred = torch.tensor([0, 0, 1, 1, 0, 1])
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert mcc == pytest.approx(1.0), "Should work with PyTorch tensors"

    def test_mcc_with_numpy_arrays(self) -> None:
        """Test with numpy arrays."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert mcc == pytest.approx(1.0), "Should work with numpy arrays"

    def test_mcc_with_lists(self) -> None:
        """Test with Python lists."""
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 1, 1, 0]
        mcc = matthews_correlation_coefficient(y_true, y_pred)
        assert mcc == pytest.approx(1.0), "Should work with Python lists"


class TestBalancedAccuracy:
    """Test balanced accuracy computation."""

    def test_balanced_accuracy_perfect(self) -> None:
        """Perfect predictions should give balanced accuracy = 1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        ba = balanced_accuracy(y_true, y_pred)
        assert ba == pytest.approx(1.0), "Perfect predictions should yield BA = 1.0"

    def test_balanced_accuracy_imbalanced_data(self) -> None:
        """Correctly handles imbalanced data."""
        # 90% negative, 10% positive
        y_true = np.array([0] * 90 + [1] * 10)
        # Naive classifier: predict all negative (90% accuracy but bad BA)
        y_pred = np.array([0] * 100)
        ba = balanced_accuracy(y_true, y_pred)
        # BA = (1.0 + 0.0) / 2 = 0.5 (perfect on negatives, terrible on positives)
        assert ba == pytest.approx(0.5), "BA should be 0.5 for all-negative predictions"

    def test_balanced_accuracy_all_positive(self) -> None:
        """All predictions positive."""
        y_true = np.array([0, 0, 1, 1, 0, 1])  # 3 positive, 3 negative
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # All predict positive
        ba = balanced_accuracy(y_true, y_pred)
        # Sensitivity = 3/3 = 1.0, Specificity = 0/3 = 0.0
        # BA = (1.0 + 0.0) / 2 = 0.5
        assert ba == pytest.approx(0.5), "BA should be 0.5 for all-positive predictions"

    def test_balanced_accuracy_all_negative(self) -> None:
        """All predictions negative."""
        y_true = np.array([0, 0, 1, 1, 0, 1])  # 3 positive, 3 negative
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All predict negative
        ba = balanced_accuracy(y_true, y_pred)
        # Sensitivity = 0/3 = 0.0, Specificity = 3/3 = 1.0
        # BA = (0.0 + 1.0) / 2 = 0.5
        assert ba == pytest.approx(0.5), "BA should be 0.5 for all-negative predictions"

    def test_balanced_accuracy_realistic(self) -> None:
        """Realistic scenario with partial correctness."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        # TP=3, TN=3, FP=1, FN=1
        # Sensitivity = 3/4 = 0.75
        # Specificity = 3/4 = 0.75
        # BA = (0.75 + 0.75) / 2 = 0.75
        ba = balanced_accuracy(y_true, y_pred)
        assert ba == pytest.approx(0.75), "BA should be 0.75"


class TestSensitivitySpecificity:
    """Test sensitivity and specificity computation."""

    def test_sensitivity_perfect(self) -> None:
        """Perfect recall should give sensitivity = 1.0."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 1])  # All true positives captured
        sens = sensitivity(y_true, y_pred)
        assert sens == pytest.approx(1.0), "Perfect recall should yield sensitivity = 1.0"

    def test_sensitivity_zero(self) -> None:
        """No true positives should give sensitivity = 0.0."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])  # All negatives
        sens = sensitivity(y_true, y_pred)
        assert sens == pytest.approx(0.0), "No true positives should yield sensitivity = 0.0"

    def test_sensitivity_partial(self) -> None:
        """Partial true positives."""
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0])  # 2 TP, 2 FN
        sens = sensitivity(y_true, y_pred)
        assert sens == pytest.approx(0.5), "Sensitivity should be 0.5"

    def test_specificity_perfect(self) -> None:
        """Perfect specificity should give 1.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0])  # All true negatives captured
        spec = specificity(y_true, y_pred)
        assert spec == pytest.approx(1.0), "Perfect specificity should yield 1.0"

    def test_specificity_zero(self) -> None:
        """No true negatives should give specificity = 0.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])  # All positives
        spec = specificity(y_true, y_pred)
        assert spec == pytest.approx(0.0), "No true negatives should yield specificity = 0.0"

    def test_specificity_partial(self) -> None:
        """Partial true negatives."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])  # 2 TN, 2 FP
        spec = specificity(y_true, y_pred)
        assert spec == pytest.approx(0.5), "Specificity should be 0.5"


class TestConfusionMatrixMetrics:
    """Test confusion matrix derived metrics."""

    def test_confusion_matrix_metrics_keys(self) -> None:
        """Returns all expected keys."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = confusion_matrix_metrics(y_true, y_pred)
        expected_keys = {"ppv", "npv", "sensitivity", "specificity"}
        assert set(metrics.keys()) == expected_keys, "Should return ppv, npv, sensitivity, specificity"

    def test_confusion_matrix_metrics_values(self) -> None:
        """Correct calculation of ppv, npv, sensitivity, specificity."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        # TP=3, TN=3, FP=1, FN=1
        metrics = confusion_matrix_metrics(y_true, y_pred)

        # PPV = TP / (TP + FP) = 3 / 4 = 0.75
        assert metrics["ppv"] == pytest.approx(0.75), "PPV should be 0.75"

        # NPV = TN / (TN + FN) = 3 / 4 = 0.75
        assert metrics["npv"] == pytest.approx(0.75), "NPV should be 0.75"

        # Sensitivity = TP / (TP + FN) = 3 / 4 = 0.75
        assert metrics["sensitivity"] == pytest.approx(0.75), "Sensitivity should be 0.75"

        # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
        assert metrics["specificity"] == pytest.approx(0.75), "Specificity should be 0.75"

    def test_confusion_matrix_metrics_perfect(self) -> None:
        """Perfect classifier."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        metrics = confusion_matrix_metrics(y_true, y_pred)

        assert metrics["ppv"] == pytest.approx(1.0), "Perfect PPV"
        assert metrics["npv"] == pytest.approx(1.0), "Perfect NPV"
        assert metrics["sensitivity"] == pytest.approx(1.0), "Perfect sensitivity"
        assert metrics["specificity"] == pytest.approx(1.0), "Perfect specificity"

    def test_confusion_matrix_metrics_all_negative_predictions(self) -> None:
        """All predictions negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        metrics = confusion_matrix_metrics(y_true, y_pred)

        # PPV undefined (no positive predictions), should be nan
        assert np.isnan(metrics["ppv"]), "PPV should be nan when no positive predictions"

        # NPV = 2/4 = 0.5 (2 true negatives, 2 false negatives)
        assert metrics["npv"] == pytest.approx(0.5), "NPV should be 0.5"

        # Sensitivity = 0 (no TP)
        assert metrics["sensitivity"] == pytest.approx(0.0), "Sensitivity should be 0"

        # Specificity = 1.0 (perfect on negatives)
        assert metrics["specificity"] == pytest.approx(1.0), "Specificity should be 1.0"


class TestComputeAllMetrics:
    """Test the comprehensive metrics function."""

    def test_compute_all_metrics_keys(self) -> None:
        """Returns all expected metric keys."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_all_metrics(y_true, y_pred)

        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "balanced_accuracy",
            "ppv",
            "npv",
            "sensitivity",
            "specificity",
        }
        assert expected_keys.issubset(set(metrics.keys())), f"Missing keys: {expected_keys - set(metrics.keys())}"

    def test_compute_all_metrics_with_probabilities(self) -> None:
        """Includes roc_auc and auprc when y_prob provided."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])

        metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob)

        assert "roc_auc" in metrics, "Should include roc_auc when y_prob provided"
        assert "auprc" in metrics, "Should include auprc when y_prob provided"
        assert 0 <= metrics["roc_auc"] <= 1, "ROC AUC should be in [0, 1]"
        assert 0 <= metrics["auprc"] <= 1, "AUPRC should be in [0, 1]"

    def test_compute_all_metrics_without_probabilities(self) -> None:
        """Works without y_prob."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metrics = compute_all_metrics(y_true, y_pred)

        assert "roc_auc" not in metrics or np.isnan(metrics.get("roc_auc", np.nan)), "Should not include valid roc_auc without y_prob"
        assert "accuracy" in metrics, "Should still compute standard metrics"

    def test_compute_all_metrics_realistic_imbalanced(self) -> None:
        """Realistic imbalanced scenario."""
        # 80% negative, 20% positive
        np.random.seed(42)
        y_true = np.array([0] * 80 + [1] * 20)

        # Classifier with 85% accuracy on negatives, 70% on positives
        y_pred = np.array(
            [0] * 68 + [1] * 12 + [1] * 14 + [0] * 6
        )

        y_prob = np.concatenate([
            np.random.uniform(0.0, 0.3, 68),  # True negatives
            np.random.uniform(0.6, 1.0, 12),  # False positives
            np.random.uniform(0.6, 1.0, 14),  # True positives
            np.random.uniform(0.0, 0.3, 6),   # False negatives
        ])

        metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob)

        # Sanity checks
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy in valid range"
        assert 0 <= metrics["f1"] <= 1, "F1 in valid range"
        assert -1 <= metrics["mcc"] <= 1, "MCC in valid range"
        assert 0 <= metrics["balanced_accuracy"] <= 1, "Balanced accuracy in valid range"

        # With imbalanced data, balanced_accuracy should differ from accuracy
        assert metrics["balanced_accuracy"] != pytest.approx(metrics["accuracy"]), \
            "Balanced accuracy should differ from accuracy in imbalanced data"

    def test_compute_all_metrics_perfect(self) -> None:
        """Perfect classifier should give perfect scores."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

        metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob)

        assert metrics["accuracy"] == pytest.approx(1.0), "Perfect accuracy"
        assert metrics["precision"] == pytest.approx(1.0), "Perfect precision"
        assert metrics["recall"] == pytest.approx(1.0), "Perfect recall"
        assert metrics["f1"] == pytest.approx(1.0), "Perfect F1"
        assert metrics["mcc"] == pytest.approx(1.0), "Perfect MCC"
        assert metrics["balanced_accuracy"] == pytest.approx(1.0), "Perfect balanced accuracy"
        assert metrics["roc_auc"] == pytest.approx(1.0), "Perfect ROC AUC"

    def test_compute_all_metrics_with_torch_tensors(self) -> None:
        """Works with PyTorch tensors."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_pred = torch.tensor([0, 0, 1, 1, 0, 1])
        y_prob = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])

        metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob)

        assert "accuracy" in metrics, "Should work with torch tensors"
        assert "roc_auc" in metrics, "Should compute probabilistic metrics"
        assert metrics["accuracy"] == pytest.approx(1.0), "Correct accuracy"
