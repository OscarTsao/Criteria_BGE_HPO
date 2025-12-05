"""DeBERTa-v3 classifier with configurable classification heads."""

import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from criteria_bge_hpo.models.classifier_heads import ClassifierHeadFactory
from criteria_bge_hpo.training.losses import FocalLoss, WeightedBCELoss


class DeBERTaClassifier(nn.Module):
    """DeBERTa-v3-base encoder with configurable classification head."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        classifier_head: str = "linear",
        classifier_dropout: float = 0.1,
        hidden_dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        intermediate_dim: int = 384,
        num_attention_heads: int = 4,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        pos_weight: Optional[float] = None,
        config: Optional[AutoConfig] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.classifier_head_type = classifier_head
        self.loss_type = loss_type

        # Load configuration
        self.config = config or AutoConfig.from_pretrained(model_name)
        if attn_implementation is not None:
            self.config.attn_implementation = attn_implementation
        if gradient_checkpointing:
            self.config.gradient_checkpointing = True
            if hasattr(self.config, "use_cache"):
                self.config.use_cache = False

        # Inject dropout parameters into config if provided
        if hidden_dropout is not None:
            self.config.hidden_dropout_prob = hidden_dropout
        if attention_dropout is not None:
            self.config.attention_probs_dropout_prob = attention_dropout

        # Load encoder (base model without classification head)
        encoder_kwargs = {"config": self.config}
        if torch_dtype is not None:
            encoder_kwargs["dtype"] = torch_dtype  # Use 'dtype' instead of deprecated 'torch_dtype'
        self.encoder = AutoModel.from_pretrained(model_name, **encoder_kwargs)

        # Initialize loss function
        self._init_loss_function(loss_type, focal_gamma, focal_alpha, pos_weight)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Create classification head
        hidden_size = self.config.hidden_size
        self.classifier = ClassifierHeadFactory.create(
            head_type=classifier_head,
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=classifier_dropout,
            intermediate_dim=intermediate_dim,
            num_attention_heads=num_attention_heads,
        )

    def _init_loss_function(
        self,
        loss_type: str,
        focal_gamma: float,
        focal_alpha: Optional[float],
        pos_weight: Optional[float],
    ):
        """Initialize the loss function based on config.

        Args:
            loss_type: Type of loss ('focal', 'bce', 'weighted_bce')
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss (auto-computed if None)
            pos_weight: Positive class weight for weighted BCE
        """
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        elif loss_type == "weighted_bce":
            self.loss_fn = WeightedBCELoss(pos_weight=pos_weight, reduction='mean')
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of: focal, bce, weighted_bce")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Build encoder inputs
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Get encoder outputs
        encoder_outputs = self.encoder(**encoder_inputs)
        sequence_output = encoder_outputs.last_hidden_state

        # Apply classification head
        logits = self.classifier(sequence_output, attention_mask)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Binary classification with sigmoid
                labels = labels.float()
                loss = self.loss_fn(logits.view(-1), labels.view(-1))
            else:
                # Multi-class classification with softmax
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"logits": logits, "loss": loss}

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.encoder.save_pretrained(save_directory)
        classifier_path = os.path.join(save_directory, "classifier_head.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        head_config = {
            "classifier_head": self.classifier_head_type,
            "num_labels": self.num_labels,
        }
        config_path = os.path.join(save_directory, "head_config.json")
        with open(config_path, "w") as f:
            json.dump(head_config, f)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        classifier_dropout: float = 0.1,
        hidden_dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        intermediate_dim: int = 384,
        num_attention_heads: int = 4,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        pos_weight: Optional[float] = None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "DeBERTaClassifier":
        config_path = os.path.join(load_directory, "head_config.json")
        with open(config_path, "r") as f:
            head_config = json.load(f)

        config = AutoConfig.from_pretrained(load_directory)
        model = cls(
            model_name=load_directory,
            num_labels=head_config["num_labels"],
            classifier_head=head_config["classifier_head"],
            classifier_dropout=classifier_dropout,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            intermediate_dim=intermediate_dim,
            num_attention_heads=num_attention_heads,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            pos_weight=pos_weight,
            config=config,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
        )

        classifier_path = os.path.join(load_directory, "classifier_head.pt")
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

        return model
