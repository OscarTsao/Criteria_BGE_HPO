"""Sequence classification model wrapper for Hugging Face encoders."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class BERTClassifier(nn.Module):
    """Wrap AutoModelForSequenceClassification for generic classifiers."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        freeze_backbone: bool = False,
        config: Optional[AutoConfig] = None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the sequence classification model.

        Args:
            model_name: Pretrained model identifier or local path.
            num_labels: Number of output labels. Keep at 1 to align with the pretrained head.
            freeze_backbone: Whether to freeze the encoder parameters.
            config: Optional pre-loaded configuration to reuse.
            attn_implementation: Attention backend to request (e.g., sdpa, flash_attention_2).
            gradient_checkpointing: Enable gradient checkpointing to save memory.
            torch_dtype: Optional dtype override when loading weights.
        """
        super().__init__()

        self.model_name = model_name
        self.config = config or AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels or self.config.num_labels
        if attn_implementation is not None:
            self.config.attn_implementation = attn_implementation
        if gradient_checkpointing:
            self.config.gradient_checkpointing = True
            if hasattr(self.config, "use_cache"):
                self.config.use_cache = False

        model_kwargs = {"config": self.config}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype  # Use 'dtype' instead of deprecated 'torch_dtype'

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.num_labels = self.model.config.num_labels

        # Detect model capabilities
        self.uses_token_type_ids = (
            hasattr(self.model.config, "type_vocab_size")
            and self.model.config.type_vocab_size is not None
            and self.model.config.type_vocab_size > 1
        )

        if freeze_backbone:
            backbone = getattr(self.model, getattr(self.model, "base_model_prefix", ""), None)
            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """Forward pass."""
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.uses_token_type_ids and token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        loss = None
        if labels is not None:
            if logits.shape[-1] == 1:
                labels = labels.float()
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"logits": logits, "loss": loss}

    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Load model from directory."""
        config = AutoConfig.from_pretrained(load_directory)
        return cls(
            model_name=load_directory,
            num_labels=config.num_labels,
            config=config,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
        )
