"""Custom loss functions for imbalanced classification."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    The focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard misclassified examples:

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        p_t = p if y=1, else 1-p (predicted probability of ground truth class)
        α_t = alpha if y=1, else 1-alpha (class weighting factor)
        γ = gamma (focusing parameter, typically 2.0)

    Args:
        alpha: Weighting factor in [0, 1] for class 1 (positive class).
               If None, no class weighting is applied.
        gamma: Focusing parameter γ ≥ 0. When γ=0, FL is equivalent to CE.
               Typically set to 2.0.
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(8, 1, requires_grad=True)
        >>> targets = torch.randint(0, 2, (8,)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model predictions of shape (N,) or (N, 1) for binary classification
            targets: Ground truth labels of shape (N,) with values in {0, 1}

        Returns:
            Focal loss value (scalar if reduction='mean'/'sum', tensor if reduction='none')
        """
        # Ensure logits and targets have compatible shapes
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities using sigmoid
        probs = torch.sigmoid(logits)

        # Compute p_t: probability of the ground truth class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute base cross-entropy loss: -log(p_t)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class weighting if alpha is specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Binary Cross-Entropy with class weighting.

    Standard BCE loss with optional per-class weighting to handle imbalance:

        L = -α_t * [y * log(p) + (1-y) * log(1-p)]

    Args:
        pos_weight: Weight for positive class. If None, computed from class distribution.
        reduction: Specifies reduction: 'none' | 'mean' | 'sum'

    Example:
        >>> loss_fn = WeightedBCELoss(pos_weight=2.0)
        >>> logits = torch.randn(8, 1, requires_grad=True)
        >>> targets = torch.randint(0, 2, (8,)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Model predictions of shape (N,) or (N, 1)
            targets: Ground truth labels of shape (N,) with values in {0, 1}

        Returns:
            Loss value
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                logits, targets, reduction=self.reduction
            )


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int = 2,
    method: str = 'inverse'
) -> Tuple[float, float]:
    """Compute class weights from target distribution.

    Args:
        targets: Binary labels tensor of shape (N,)
        num_classes: Number of classes (must be 2 for binary)
        method: Weighting method:
            - 'inverse': weight_i = n_samples / (n_classes * n_samples_i)
            - 'effective': Effective number of samples (Cui et al. 2019)

    Returns:
        Tuple of (neg_weight, pos_weight)

    Example:
        >>> targets = torch.tensor([0, 0, 0, 1])  # 75% negative, 25% positive
        >>> neg_w, pos_w = compute_class_weights(targets)
        >>> print(f"Weights: neg={neg_w:.2f}, pos={pos_w:.2f}")
        Weights: neg=0.67, pos=2.00
    """
    assert num_classes == 2, "Only binary classification supported"

    targets_np = targets.cpu().numpy()
    n_samples = len(targets_np)
    n_pos = targets_np.sum()
    n_neg = n_samples - n_pos

    if method == 'inverse':
        # Standard inverse frequency weighting
        if n_neg == 0:
            neg_weight = 1.0
        else:
            neg_weight = n_samples / (num_classes * n_neg)

        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = n_samples / (num_classes * n_pos)

    elif method == 'effective':
        # Effective number of samples (Cui et al. "Class-Balanced Loss", CVPR 2019)
        beta = 0.9999
        if n_neg == 0:
            neg_weight = 1.0
        else:
            effective_neg = (1 - beta**n_neg) / (1 - beta)
            neg_weight = (1 - beta) / effective_neg

        if n_pos == 0:
            pos_weight = 1.0
        else:
            effective_pos = (1 - beta**n_pos) / (1 - beta)
            pos_weight = (1 - beta) / effective_pos
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(neg_weight), float(pos_weight)


def compute_focal_alpha(targets: torch.Tensor) -> float:
    """Compute focal loss alpha from class distribution.

    Sets alpha to balance the contribution of positive and negative examples:
        alpha = n_neg / (n_pos + n_neg)

    This ensures the weighted loss contribution is balanced when combined
    with the focal modulating factor.

    Args:
        targets: Binary labels tensor of shape (N,)

    Returns:
        Alpha value in [0, 1] for positive class weighting

    Example:
        >>> targets = torch.tensor([0, 0, 0, 1])  # 25% positive
        >>> alpha = compute_focal_alpha(targets)
        >>> print(f"Alpha: {alpha:.3f}")
        Alpha: 0.750
    """
    n_samples = len(targets)
    n_pos = targets.sum().item()
    n_neg = n_samples - n_pos

    if n_samples == 0:
        return 0.5

    # Alpha for positive class = negative class proportion
    alpha = n_neg / n_samples
    return alpha
