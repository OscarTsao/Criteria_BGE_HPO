"""Tests for preprocessing and dataset utilities."""

from types import SimpleNamespace

import pandas as pd
import pytest
from transformers import AutoTokenizer

from criteria_bge_hpo.data.dataset import DSM5NLIDataset, create_dataloaders
from criteria_bge_hpo.data.preprocessing import load_and_preprocess_data


def _build_config(tmp_path, sample_size=None):
    """Create a lightweight config namespace for preprocessing."""
    groundtruth_path = tmp_path / "groundtruth.csv"
    criteria_path = tmp_path / "criteria.json"

    pd.DataFrame(
        {
            "post_id": ["p1", "p1"],
            "post": ["sad post", "sad post"],
            "DSM5_symptom": ["A.1", "A.1"],
            "groundtruth": [1, 0],
        }
    ).to_csv(groundtruth_path, index=False)

    criteria_path.write_text(
        '{"criteria": [{"id": "A.1", "text": "Depressed mood most of the day."}]}',
        encoding="utf-8",
    )

    data_cfg = SimpleNamespace(
        groundtruth_csv=str(groundtruth_path),
        criteria_json=str(criteria_path),
        sample_size=sample_size,
        sample_seed=123,
        max_length=32,
    )
    return SimpleNamespace(data=data_cfg, seed=123)


def _make_dataset(tokenizer):
    sample_df = pd.DataFrame(
        {
            "post": ["I feel down", "Sleep is impossible"],
            "criterion": [
                "Depressed mood most of the day",
                "Insomnia or hypersomnia nearly every day",
            ],
            "label": [1, 0],
        }
    )
    return DSM5NLIDataset(sample_df, tokenizer, max_length=32, verify_format=True)


def test_load_and_preprocess_data_returns_expected_columns(tmp_path):
    """End-to-end preprocessing builds labeled pairs with criterion text."""
    cfg = _build_config(tmp_path)
    df = load_and_preprocess_data(cfg)

    assert list(df.columns) == ["post_id", "post", "criterion_id", "criterion", "label", "evidence_text"]
    assert df["criterion"].iloc[0] == "Depressed mood most of the day."
    # Sanity-check that label column is coerced to integers
    assert df["label"].dtype == "int64"


def test_load_and_preprocess_data_supports_sampling(tmp_path):
    """sample_size trims the dataset deterministically for smoke tests."""
    cfg = _build_config(tmp_path, sample_size=1)
    df = load_and_preprocess_data(cfg)

    assert len(df) == 1
    assert df.iloc[0]["post_id"] == "p1"


@pytest.mark.filterwarnings("ignore:Multiprocessing dataloaders")
def test_create_dataloaders_falls_back_when_multiprocessing_unavailable(monkeypatch):
    """create_dataloaders should drop to num_workers=0 if multiprocessing fails."""
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    dataset = _make_dataset(tokenizer)

    monkeypatch.setattr(
        "criteria_bge_hpo.data.dataset._supports_multiprocessing",
        lambda: False,
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=False,
    )

    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
