"""Classification head architectures for sequence classification models.

Provides a registry of different pooling and classification head strategies,
from simple linear layers to attention-based pooling mechanisms.
"""

from typing import Optional

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Simple linear classification head on [CLS] token.

    Architecture: [CLS] → Linear(hidden_size, num_labels)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), unused for CLS pooling

        Returns:
            logits: (batch_size, num_labels)
        """
        # Take [CLS] token representation
        cls_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class PoolerLinearHead(nn.Module):
    """Linear head with pooler layer (BERT-style).

    Architecture: [CLS] → Linear(h,h) → Tanh → Dropout → Linear(h, num_labels)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), unused

        Returns:
            logits: (batch_size, num_labels)
        """
        # Take [CLS] token
        cls_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        # Pooler transformation
        pooled_output = self.dense(cls_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Classify
        logits = self.classifier(pooled_output)
        return logits


class MLP1Head(nn.Module):
    """Single-layer MLP classification head with LayerNorm.

    Architecture: [CLS] → Linear(h,h) → LayerNorm → GELU → Dropout → Linear(h, num_labels)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), unused

        Returns:
            logits: (batch_size, num_labels)
        """
        # Take [CLS] token
        cls_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        # MLP transformation
        x = self.dense(cls_output)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Classify
        logits = self.classifier(x)
        return logits


class MLP2Head(nn.Module):
    """Two-layer MLP classification head with configurable intermediate dimension.

    Architecture:
        [CLS] → Linear(h, intermediate_dim) → LayerNorm → GELU → Dropout
              → Linear(intermediate_dim, h) → LayerNorm → GELU → Dropout
              → Linear(h, num_labels)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        intermediate_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_size * 2

        # First layer
        self.dense1 = nn.Linear(hidden_size, intermediate_dim)
        self.layer_norm1 = nn.LayerNorm(intermediate_dim)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second layer
        self.dense2 = nn.Linear(intermediate_dim, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), unused

        Returns:
            logits: (batch_size, num_labels)
        """
        # Take [CLS] token
        cls_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        # First MLP layer
        x = self.dense1(cls_output)
        x = self.layer_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        # Second MLP layer
        x = self.dense2(x)
        x = self.layer_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        # Classify
        logits = self.classifier(x)
        return logits


class MeanPoolingHead(nn.Module):
    """Mean pooling over sequence with attention mask weighting.

    Architecture:
        Mean(hidden_states * attention_mask) → Linear(h, num_labels)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with masked mean pooling.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding

        Returns:
            logits: (batch_size, num_labels)
        """
        if attention_mask is None:
            # No mask, simple mean
            pooled = hidden_states.mean(dim=1)
        else:
            # Expand mask to match hidden dimension
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())

            # Sum hidden states where mask=1
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)

            # Count valid tokens per sample
            sum_mask = mask_expanded.sum(dim=1)

            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            # Mean pooling
            pooled = sum_hidden / sum_mask

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class MaxPoolingHead(nn.Module):
    """Max pooling over sequence with attention mask.

    Architecture:
        Max(hidden_states * attention_mask) -> Dropout -> Linear(h, num_labels)

    Takes the maximum value across the sequence dimension for each feature,
    properly handling padding tokens via attention mask.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with max pooling.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding

        Returns:
            logits: (batch_size, num_labels)
        """
        if attention_mask is None:
            # No mask provided, max pool over all positions
            pooled, _ = hidden_states.max(dim=1)
        else:
            # Expand mask to match hidden_states shape
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())

            # Set padding positions to -inf so they don't affect max
            masked_hidden = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))

            # Max pool over sequence dimension
            pooled, _ = masked_hidden.max(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class AttentionPoolingHead(nn.Module):
    """Multi-head attention pooling over sequence.

    Architecture:
        Query vector → MultiHeadAttention(query, hidden_states, hidden_states)
                    → Linear(h, num_labels)

    Uses a learnable query vector to attend over the sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        num_attention_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with attention pooling.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding

        Returns:
            logits: (batch_size, num_labels)
        """
        batch_size = hidden_states.size(0)

        # Expand query to batch
        query = self.query.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)

        # Create key_padding_mask (True for positions to ignore)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (batch_size, seq_len)
        else:
            key_padding_mask = None

        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )

        # Squeeze query dimension
        pooled = attn_output.squeeze(1)  # (batch_size, hidden_size)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# Registry mapping names to head classes
HEAD_REGISTRY = {
    "linear": LinearHead,
    "pooler_linear": PoolerLinearHead,
    "mlp1": MLP1Head,
    "mlp2": MLP2Head,
    "mean_pooling": MeanPoolingHead,
    "max_pooling": MaxPoolingHead,
    "attention_pooling": AttentionPoolingHead,
}


class ClassifierHeadFactory:
    """Factory for creating classification heads from config."""

    @staticmethod
    def create(
        head_type: str,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ) -> nn.Module:
        """Create a classification head.

        Args:
            head_type: Name of head architecture (e.g., "linear", "mlp1")
            hidden_size: Transformer hidden dimension
            num_labels: Number of output classes
            dropout: Dropout probability
            **kwargs: Additional head-specific arguments

        Returns:
            Classification head module

        Raises:
            ValueError: If head_type not in registry
        """
        if head_type not in HEAD_REGISTRY:
            available = ", ".join(HEAD_REGISTRY.keys())
            raise ValueError(
                f"Unknown head type '{head_type}'. Available: {available}"
            )

        head_class = HEAD_REGISTRY[head_type]
        return head_class(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            **kwargs
        )
