"""Tests for the generic classifier configuration."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from criteria_bge_hpo.data.dataset import DSM5NLIDataset  # noqa: E402
from criteria_bge_hpo.evaluation.evaluator import Evaluator  # noqa: E402
from criteria_bge_hpo.models.bert_classifier import BERTClassifier  # noqa: E402


@pytest.fixture
def config_dir():
    return str(repo_root / "configs")


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "post": [
                "I feel sad and hopeless every day",
                "I lost interest in activities I used to enjoy",
                "My sleep has been terrible lately",
            ],
            "criterion": [
                "Depressed mood most of the day",
                "Diminished interest or pleasure in activities",
                "Insomnia or hypersomnia nearly every day",
            ],
            "label": [1, 1, 1],
        }
    )


def test_default_model_config(config_dir):
    """Hydra loads the generic model configuration."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    assert cfg.model.model_name == "microsoft/deberta-v3-base"
    assert cfg.model.num_labels == 2
    assert cfg.model.freeze_backbone is False
    assert cfg.model.positive_threshold == pytest.approx(0.5)


def test_dataset_with_token_type_ids(sample_data):
    """Dataset includes token_type_ids for BERT-style tokenizers."""
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    dataset = DSM5NLIDataset(sample_data, tokenizer, max_length=128, verify_format=True)

    sample = dataset[0]
    assert "token_type_ids" in sample
    assert sample["input_ids"].shape == (128,)
    assert sample["attention_mask"].shape == (128,)
    assert sample["token_type_ids"].shape == (128,)


class _DummyModel(torch.nn.Module):
    """Tiny stand-in for AutoModelForSequenceClassification with two logits."""

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config
        self.base_model_prefix = "encoder"
        self.encoder = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Linear(8, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        batch_size = input_ids.size(0)
        hidden = self.encoder(torch.zeros(batch_size, 8, device=input_ids.device))
        logits = self.classifier(hidden)
        return {"logits": logits}


def test_classifier_forward_two_logits(monkeypatch):
    """Model wrapper uses CrossEntropyLoss for two-logit heads."""
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.num_labels = 2

    monkeypatch.setattr(
        "criteria_bge_hpo.models.bert_classifier.AutoModelForSequenceClassification.from_pretrained",
        lambda model_name, config: _DummyModel(config),
    )

    model = BERTClassifier(
        model_name="prajjwal1/bert-tiny",
        num_labels=2,
        config=config,
    )

    batch_size, seq_len = 2, 5
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([1, 0])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert outputs["logits"].shape == (batch_size, 2)
    assert outputs["loss"] is not None
    outputs["loss"].backward()
    assert model.model.classifier.weight.grad is not None


class _FixedLogitModel(torch.nn.Module):
    """Return predetermined logits for evaluation logic tests."""

    def __init__(self, logits):
        super().__init__()
        self.config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        self.config.num_labels = 1
        self._logits = torch.tensor(logits, dtype=torch.float)
        self._cursor = 0

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        batch_size = input_ids.size(0) if input_ids is not None else self._logits.size(0)
        start = self._cursor
        end = start + batch_size
        logits = self._logits[start:end]
        self._cursor = end % self._logits.size(0)
        return {"logits": logits}


class _TinyDataset(Dataset):
    def __init__(self):
        self.labels = [1, 0, 1, 0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return {
            "input_ids": torch.ones(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def test_evaluator_thresholding_for_single_logit():
    """Evaluator thresholds sigmoid scores correctly."""
    dataset = _TinyDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = _FixedLogitModel([[5.0], [-5.0], [4.0], [-4.0]])
    evaluator = Evaluator(model=model, device="cpu", positive_threshold=0.5)

    data = pd.DataFrame(
        {
            "criterion_id": ["A.1", "A.1", "A.2", "A.2"],
            "label": dataset.labels,
        }
    )

    results = evaluator.evaluate(loader, data)
    assert results["aggregate"]["accuracy"] == 1.0
    assert results["aggregate"]["f1"] == 1.0
