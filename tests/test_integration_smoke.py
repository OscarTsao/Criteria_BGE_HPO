"""Integration smoke test for nested CV system.

This test verifies that the entire pipeline works end-to-end with minimal data.
"""

import pytest
import torch
from transformers import AutoTokenizer

from criteria_bge_hpo.models import DeBERTaClassifier
from criteria_bge_hpo.training.losses import FocalLoss


class TestIntegrationSmoke:
    """Smoke tests for end-to-end integration."""

    def test_deberta_nli_model_instantiation(self):
        """Test that DeBERTa NLI model can be instantiated."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            classifier_head="linear",
            classifier_dropout=0.1,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            loss_type="focal",
            focal_gamma=2.0,
            focal_alpha=0.25,
        )

        assert model is not None
        assert model.loss_type == "focal"
        assert isinstance(model.loss_fn, FocalLoss)

    def test_deberta_forward_pass(self):
        """Test DeBERTa forward pass with synthetic data."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            classifier_head="max_pooling",
            loss_type="focal",
            focal_gamma=2.0,
        )

        # Create synthetic batch
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)

        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["loss"] is not None
        assert outputs["loss"].item() >= 0

    def test_all_classifier_heads_work(self):
        """Test that all classifier heads can be used with DeBERTa."""
        heads = ["linear", "mean_pooling", "max_pooling", "attention_pooling", "mlp1"]

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,)).float()

        for head_type in heads:
            model = DeBERTaClassifier(
                model_name="microsoft/deberta-v3-base",
                num_labels=1,
                classifier_head=head_type,
            )

            outputs = model(input_ids, attention_mask, labels=labels)
            assert outputs["logits"].shape == (batch_size, 1)
            assert outputs["loss"] is not None

    def test_all_loss_types_work(self):
        """Test that all loss types can be used with DeBERTa."""
        loss_types = ["bce", "focal", "weighted_bce"]

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,)).float()

        for loss_type in loss_types:
            model = DeBERTaClassifier(
                model_name="microsoft/deberta-v3-base",
                num_labels=1,
                loss_type=loss_type,
                focal_gamma=2.0 if loss_type == "focal" else 2.0,
                pos_weight=2.0 if loss_type == "weighted_bce" else None,
            )

            outputs = model(input_ids, attention_mask, labels=labels)
            assert outputs["loss"] is not None
            assert outputs["loss"].item() >= 0

    def test_dropout_injection(self):
        """Test that dropout parameters are correctly injected into config."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            hidden_dropout=0.15,
            attention_dropout=0.2,
        )

        # Check that dropouts were injected
        assert model.config.hidden_dropout_prob == 0.15
        assert model.config.attention_probs_dropout_prob == 0.2

    def test_gradient_flow_through_model(self):
        """Test that gradients flow correctly through the entire model."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            classifier_head="max_pooling",
            loss_type="focal",
        )

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Check that classifier head has gradients
        for param in model.classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_model_inference_mode(self):
        """Test model can run in inference mode (no labels)."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            classifier_head="linear",
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 1)
        assert outputs["loss"] is None

    def test_config_files_exist(self):
        """Test that all required config files exist."""
        import os

        config_dir = "configs"
        required_configs = [
            "configs/model/deberta_nli.yaml",
            "configs/hpo/nested_cv.yaml",
            "configs/model/deberta_v3_base.yaml",
            "configs/training/deberta.yaml",
            "configs/hpo/deberta.yaml",
        ]

        for config_path in required_configs:
            assert os.path.exists(config_path), f"Missing config file: {config_path}"

    def test_imports_work(self):
        """Test that all new components can be imported."""
        from criteria_bge_hpo.models import (
            DeBERTaClassifier,
            MaxPoolingHead,
            HEAD_REGISTRY,
        )
        from criteria_bge_hpo.training import (
            FocalLoss,
            WeightedBCELoss,
            compute_class_weights,
            compute_focal_alpha,
        )
        from criteria_bge_hpo.evaluation import Evaluator

        # Verify imports worked
        assert DeBERTaClassifier is not None
        assert MaxPoolingHead is not None
        assert "max_pooling" in HEAD_REGISTRY
        assert FocalLoss is not None
        assert WeightedBCELoss is not None
        assert Evaluator is not None

    @pytest.mark.slow
    def test_single_training_step(self):
        """Test that a single training step completes without errors."""
        model = DeBERTaClassifier(
            model_name="microsoft/deberta-v3-base",
            num_labels=1,
            classifier_head="max_pooling",
            loss_type="focal",
            focal_gamma=2.0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Create synthetic batch
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        # Verify loss decreased (or at least computed)
        assert loss.item() >= 0
