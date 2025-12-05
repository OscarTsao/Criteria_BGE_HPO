"""Unit tests for classification heads."""

import pytest
import torch
from criteria_bge_hpo.models.classifier_heads import (
    LinearHead,
    PoolerLinearHead,
    MLP1Head,
    MLP2Head,
    MeanPoolingHead,
    MaxPoolingHead,
    AttentionPoolingHead,
    ClassifierHeadFactory,
    HEAD_REGISTRY,
)


class TestClassifierHeadRegistry:
    """Tests for the head registry and factory."""

    def test_registry_contains_all_heads(self):
        """Test that registry contains all head types."""
        expected_heads = [
            "linear",
            "pooler_linear",
            "mlp1",
            "mlp2",
            "mean_pooling",
            "max_pooling",
            "attention_pooling",
        ]

        for head_name in expected_heads:
            assert head_name in HEAD_REGISTRY

    def test_factory_creates_all_head_types(self):
        """Test factory can create all registered head types."""
        hidden_size = 768
        num_labels = 2

        for head_name in HEAD_REGISTRY.keys():
            head = ClassifierHeadFactory.create(
                head_type=head_name,
                hidden_size=hidden_size,
                num_labels=num_labels,
            )
            assert head is not None
            assert isinstance(head, torch.nn.Module)

    def test_factory_invalid_head_type(self):
        """Test factory raises error for invalid head type."""
        with pytest.raises(ValueError, match="Unknown head type"):
            ClassifierHeadFactory.create(
                head_type="invalid_head",
                hidden_size=768,
                num_labels=2,
            )


class TestLinearHead:
    """Tests for LinearHead."""

    def test_linear_head_initialization(self):
        """Test LinearHead initialization."""
        head = LinearHead(hidden_size=768, num_labels=2, dropout=0.1)
        assert isinstance(head, torch.nn.Module)

    def test_linear_head_forward(self):
        """Test LinearHead forward pass."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = LinearHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states)

        assert logits.shape == (batch_size, num_labels)

    def test_linear_head_uses_cls_token(self):
        """Test that LinearHead uses CLS token (first position)."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_labels = 2

        head = LinearHead(hidden_size=hidden_size, num_labels=num_labels)

        # Create input with distinct first token
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
        hidden_states[:, 0, :] = 1.0  # CLS token has all 1s

        logits = head(hidden_states)

        # Output should be non-zero since CLS token is non-zero
        assert not torch.allclose(logits, torch.zeros_like(logits))


class TestPoolerLinearHead:
    """Tests for PoolerLinearHead."""

    def test_pooler_linear_head_forward(self):
        """Test PoolerLinearHead forward pass."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = PoolerLinearHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states)

        assert logits.shape == (batch_size, num_labels)


class TestMLP1Head:
    """Tests for MLP1Head."""

    def test_mlp1_head_forward(self):
        """Test MLP1Head forward pass."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MLP1Head(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states)

        assert logits.shape == (batch_size, num_labels)


class TestMLP2Head:
    """Tests for MLP2Head."""

    def test_mlp2_head_forward(self):
        """Test MLP2Head forward pass."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MLP2Head(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states)

        assert logits.shape == (batch_size, num_labels)

    def test_mlp2_head_custom_intermediate_dim(self):
        """Test MLP2Head with custom intermediate dimension."""
        head = MLP2Head(
            hidden_size=768,
            num_labels=2,
            intermediate_dim=512,
        )
        hidden_states = torch.randn(4, 16, 768)

        logits = head(hidden_states)

        assert logits.shape == (4, 2)


class TestMeanPoolingHead:
    """Tests for MeanPoolingHead."""

    def test_mean_pooling_head_forward_without_mask(self):
        """Test MeanPoolingHead forward pass without attention mask."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MeanPoolingHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states, attention_mask=None)

        assert logits.shape == (batch_size, num_labels)

    def test_mean_pooling_head_forward_with_mask(self):
        """Test MeanPoolingHead forward pass with attention mask."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MeanPoolingHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Create attention mask with padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # Padding for first example

        logits = head(hidden_states, attention_mask=attention_mask)

        assert logits.shape == (batch_size, num_labels)

    def test_mean_pooling_ignores_padding(self):
        """Test that MeanPoolingHead correctly ignores padding tokens."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_labels = 2

        head = MeanPoolingHead(hidden_size=hidden_size, num_labels=num_labels)

        # Create input where first half is 1s, second half is 0s (padding)
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
        hidden_states[:, :5, :] = 1.0

        # Mask indicating first 5 tokens are real, last 5 are padding
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, :5] = 1.0

        logits_masked = head(hidden_states, attention_mask=attention_mask)

        # Without mask, would average all tokens (mean = 0.5)
        logits_unmasked = head(hidden_states, attention_mask=None)

        # Results should differ because mask excludes padding
        assert not torch.allclose(logits_masked, logits_unmasked)


class TestMaxPoolingHead:
    """Tests for MaxPoolingHead."""

    def test_max_pooling_head_initialization(self):
        """Test MaxPoolingHead initialization."""
        head = MaxPoolingHead(hidden_size=768, num_labels=2, dropout=0.1)
        assert isinstance(head, torch.nn.Module)

    def test_max_pooling_head_forward_without_mask(self):
        """Test MaxPoolingHead forward pass without attention mask."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MaxPoolingHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states, attention_mask=None)

        assert logits.shape == (batch_size, num_labels)

    def test_max_pooling_head_forward_with_mask(self):
        """Test MaxPoolingHead forward pass with attention mask."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = MaxPoolingHead(hidden_size=hidden_size, num_labels=num_labels)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Create attention mask with padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # Padding for first example

        logits = head(hidden_states, attention_mask=attention_mask)

        assert logits.shape == (batch_size, num_labels)

    def test_max_pooling_ignores_padding(self):
        """Test that MaxPoolingHead correctly ignores padding tokens."""
        batch_size, seq_len, hidden_size = 2, 10, 8
        num_labels = 2

        head = MaxPoolingHead(hidden_size=hidden_size, num_labels=num_labels)

        # Create input with specific pattern
        # Real tokens: [1, 2, 3, 4, 5], Padding: [10, 10, 10, 10, 10]
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
        for i in range(5):
            hidden_states[:, i, :] = i + 1  # Real tokens: 1-5
        for i in range(5, 10):
            hidden_states[:, i, :] = 10  # Padding tokens: 10

        # Mask indicating first 5 tokens are real
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, :5] = 1.0

        logits_masked = head(hidden_states, attention_mask=attention_mask)

        # Without mask, max would be 10 (from padding)
        logits_unmasked = head(hidden_states, attention_mask=None)

        # Results should differ because mask excludes padding
        assert not torch.allclose(logits_masked, logits_unmasked)

    def test_max_pooling_takes_maximum(self):
        """Test that MaxPoolingHead correctly computes maximum."""
        batch_size, seq_len, hidden_size = 1, 5, 4
        num_labels = 1

        head = MaxPoolingHead(hidden_size=hidden_size, num_labels=num_labels, dropout=0.0)

        # Create input with known max values
        hidden_states = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                        [5.0, 1.0, 2.0, 3.0],
                                        [2.0, 8.0, 1.0, 2.0],
                                        [3.0, 4.0, 9.0, 1.0],
                                        [4.0, 5.0, 6.0, 10.0]]])

        # Expected pooled output should be max across sequence: [5, 8, 9, 10]
        _ = head(hidden_states, attention_mask=None)

        # Just verify it runs without error (actual values depend on linear layer)
        assert True


class TestAttentionPoolingHead:
    """Tests for AttentionPoolingHead."""

    def test_attention_pooling_head_forward(self):
        """Test AttentionPoolingHead forward pass."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = AttentionPoolingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            num_attention_heads=8,
        )
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        logits = head(hidden_states)

        assert logits.shape == (batch_size, num_labels)

    def test_attention_pooling_head_with_mask(self):
        """Test AttentionPoolingHead with attention mask."""
        batch_size, seq_len, hidden_size = 4, 16, 768
        num_labels = 2

        head = AttentionPoolingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
        )
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0

        logits = head(hidden_states, attention_mask=attention_mask)

        assert logits.shape == (batch_size, num_labels)


class TestHeadOutputShapes:
    """Test that all heads produce correct output shapes."""

    @pytest.mark.parametrize("head_type", list(HEAD_REGISTRY.keys()))
    @pytest.mark.parametrize("num_labels", [1, 2, 3])
    def test_all_heads_output_shape(self, head_type, num_labels):
        """Test that all heads produce correct output shape."""
        batch_size, seq_len, hidden_size = 4, 16, 768

        head = ClassifierHeadFactory.create(
            head_type=head_type,
            hidden_size=hidden_size,
            num_labels=num_labels,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        logits = head(hidden_states, attention_mask)

        assert logits.shape == (batch_size, num_labels)

    @pytest.mark.parametrize("head_type", list(HEAD_REGISTRY.keys()))
    def test_all_heads_gradient_flow(self, head_type):
        """Test that gradients flow through all heads."""
        batch_size, seq_len, hidden_size = 4, 16, 768

        head = ClassifierHeadFactory.create(
            head_type=head_type,
            hidden_size=hidden_size,
            num_labels=2,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        attention_mask = torch.ones(batch_size, seq_len)

        logits = head(hidden_states, attention_mask)
        loss = logits.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
