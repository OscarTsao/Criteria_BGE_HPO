"""Unit tests for custom loss functions."""

import pytest
import torch
import torch.nn as nn
from criteria_bge_hpo.training.losses import (
    FocalLoss,
    WeightedBCELoss,
    compute_class_weights,
    compute_focal_alpha,
)


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization with valid parameters."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == 'mean'

    def test_focal_loss_invalid_alpha(self):
        """Test FocalLoss raises error for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(alpha=-0.1)

    def test_focal_loss_invalid_gamma(self):
        """Test FocalLoss raises error for invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            FocalLoss(gamma=-1.0)

    def test_focal_loss_invalid_reduction(self):
        """Test FocalLoss raises error for invalid reduction."""
        with pytest.raises(ValueError, match="reduction must be"):
            FocalLoss(reduction='invalid')

    def test_focal_loss_forward_balanced(self):
        """Test FocalLoss forward pass with balanced data."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Balanced data: 4 positive, 4 negative
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_focal_loss_forward_imbalanced(self):
        """Test FocalLoss forward pass with imbalanced data."""
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

        # Imbalanced: 7 negative, 1 positive
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_focal_loss_reduction_modes(self):
        """Test FocalLoss with different reduction modes."""
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,)).float()

        # Mean reduction
        loss_mean = FocalLoss(gamma=2.0, reduction='mean')(logits, targets)
        assert loss_mean.ndim == 0

        # Sum reduction
        loss_sum = FocalLoss(gamma=2.0, reduction='sum')(logits, targets)
        assert loss_sum.ndim == 0
        assert loss_sum.item() >= loss_mean.item()

        # None reduction
        loss_none = FocalLoss(gamma=2.0, reduction='none')(logits, targets)
        assert loss_none.ndim == 1
        assert loss_none.shape[0] == 8

    def test_focal_loss_backward(self):
        """Test FocalLoss gradient flow."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        logits = torch.randn(4, 1, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1]).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_focal_loss_gamma_zero_equals_bce(self):
        """Test that gamma=0 reduces to standard BCE."""
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16,)).float()

        focal_loss = FocalLoss(alpha=None, gamma=0.0)(logits, targets)
        bce_loss = nn.BCEWithLogitsLoss()(logits.view(-1), targets)

        # Should be approximately equal (within floating point tolerance)
        assert torch.allclose(focal_loss, bce_loss, atol=1e-6)


class TestWeightedBCELoss:
    """Tests for WeightedBCELoss."""

    def test_weighted_bce_initialization(self):
        """Test WeightedBCELoss initialization."""
        loss_fn = WeightedBCELoss(pos_weight=2.0)
        assert loss_fn.pos_weight == 2.0
        assert loss_fn.reduction == 'mean'

    def test_weighted_bce_forward(self):
        """Test WeightedBCELoss forward pass."""
        loss_fn = WeightedBCELoss(pos_weight=2.0)

        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8,)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_weighted_bce_without_weight(self):
        """Test WeightedBCELoss without pos_weight (defaults to BCE)."""
        loss_fn = WeightedBCELoss(pos_weight=None)

        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,)).float()

        loss_weighted = loss_fn(logits, targets)
        loss_bce = nn.BCEWithLogitsLoss()(logits.view(-1), targets)

        assert torch.allclose(loss_weighted, loss_bce, atol=1e-6)


class TestClassWeightFunctions:
    """Tests for class weight computation functions."""

    def test_compute_class_weights_balanced(self):
        """Test compute_class_weights with balanced data."""
        targets = torch.tensor([0, 0, 1, 1])  # 50-50 split

        neg_w, pos_w = compute_class_weights(targets)

        assert isinstance(neg_w, float)
        assert isinstance(pos_w, float)
        assert neg_w > 0
        assert pos_w > 0
        # For balanced data, weights should be approximately equal
        assert abs(neg_w - pos_w) < 0.1

    def test_compute_class_weights_imbalanced(self):
        """Test compute_class_weights with imbalanced data."""
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])  # 7:2 ratio

        neg_w, pos_w = compute_class_weights(targets)

        # Positive class should have higher weight
        assert pos_w > neg_w
        assert pos_w / neg_w > 2.0  # Significant imbalance

    def test_compute_class_weights_methods(self):
        """Test both weighting methods."""
        targets = torch.tensor([0, 0, 0, 1])

        # Inverse frequency
        neg_w1, pos_w1 = compute_class_weights(targets, method='inverse')
        assert neg_w1 > 0 and pos_w1 > 0

        # Effective number
        neg_w2, pos_w2 = compute_class_weights(targets, method='effective')
        assert neg_w2 > 0 and pos_w2 > 0

        # Methods should give different weights
        assert abs(neg_w1 - neg_w2) > 1e-6 or abs(pos_w1 - pos_w2) > 1e-6

    def test_compute_class_weights_invalid_method(self):
        """Test compute_class_weights with invalid method."""
        targets = torch.tensor([0, 1, 0, 1])

        with pytest.raises(ValueError, match="Unknown method"):
            compute_class_weights(targets, method='invalid')

    def test_compute_focal_alpha_balanced(self):
        """Test compute_focal_alpha with balanced data."""
        targets = torch.tensor([0, 0, 1, 1])

        alpha = compute_focal_alpha(targets)

        assert isinstance(alpha, float)
        assert 0.0 <= alpha <= 1.0
        # For balanced data, alpha should be ~0.5
        assert abs(alpha - 0.5) < 0.1

    def test_compute_focal_alpha_imbalanced(self):
        """Test compute_focal_alpha with imbalanced data."""
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])  # 22% positive

        alpha = compute_focal_alpha(targets)

        # Alpha = n_neg / n_total = 7/9 = 0.778
        assert 0.75 < alpha < 0.80

    def test_compute_focal_alpha_extreme_imbalance(self):
        """Test compute_focal_alpha with extreme imbalance."""
        targets = torch.tensor([0] * 99 + [1])  # 1% positive

        alpha = compute_focal_alpha(targets)

        # Alpha should be close to 0.99
        assert alpha > 0.95

    def test_compute_focal_alpha_empty(self):
        """Test compute_focal_alpha with empty tensor."""
        targets = torch.tensor([])

        alpha = compute_focal_alpha(targets)

        # Should return default 0.5
        assert alpha == 0.5


class TestLossIntegration:
    """Integration tests for loss functions."""

    def test_focal_loss_with_perfect_predictions(self):
        """Test FocalLoss with perfect predictions."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Perfect predictions (very confident)
        logits = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(logits, targets)

        # Loss should be very small for perfect predictions
        assert loss.item() < 0.01

    def test_focal_loss_with_wrong_predictions(self):
        """Test FocalLoss with completely wrong predictions."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Wrong predictions (very confident but wrong)
        logits = torch.tensor([[-10.0], [10.0], [-10.0], [10.0]])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(logits, targets)

        # Loss should be large for wrong predictions
        assert loss.item() > 1.0

    def test_loss_gradient_stability(self):
        """Test that loss gradients are stable (no NaN or Inf)."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Random predictions including edge cases
        logits = torch.randn(100, 1, requires_grad=True)
        targets = torch.randint(0, 2, (100,)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        # Gradients should be finite
        assert torch.isfinite(logits.grad).all()
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()
