"""PyTorch Dataset for DSM-5 NLI."""

import multiprocessing as mp
import numpy as np
import random
import warnings
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from criteria_bge_hpo.data.augmentation import AugmentationFactory
from criteria_bge_hpo.data.nli_templates import NLITemplateGenerator


class DSM5NLIDataset(Dataset):
    """Dataset for DSM-5 Natural Language Inference binary classification.

    Each sample is a (premise, hypothesis) pair with binary label:
    - Premise: Reddit sentence/post (evidence text)
    - Hypothesis: DSM-5 criterion (or template-generated hypothesis)
    - Label: 1 if premise entails hypothesis (symptom present), 0 otherwise

    Input format: tokenizer(premise, hypothesis) -> [CLS] premise [SEP] hypothesis [SEP]

    Supports two modes:
    - Legacy: data has (post, criterion) columns
    - NLI: data has (sentence_text, criterion_id) + nli_generator provided
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        verify_format: bool = False,
        model_name: Optional[str] = None,
        augment_config=None,
        nli_generator: Optional[NLITemplateGenerator] = None,
    ):
        """Initialize dataset.

        Args:
            data: DataFrame with columns:
                - post/sentence_text: Premise (evidence text)
                - criterion/criterion_id: Hypothesis source
                - label: Binary label (1=entailment, 0=no-entailment)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            verify_format: Whether to validate column presence (debug helper)
            model_name: Model name for detecting token_type_ids support (optional)
            augment_config: Optional augmentation config namespace
            nli_generator: Optional NLI template generator for hypothesis generation.
                If provided, uses criterion_id to generate hypothesis templates.
                If None, uses criterion column directly (backward compatibility).
        """
        if verify_format:
            required_columns = {"post", "criterion", "label"}
            if augment_config is not None:
                required_columns.add("evidence_text")
            missing = required_columns - set(data.columns)
            if missing:
                raise ValueError(f"DSM5NLIDataset missing required columns: {sorted(missing)}")

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_config = augment_config
        self.augmenter = AugmentationFactory.get_augmenter(augment_config)
        self.augment_prob = float(getattr(augment_config, "prob", 0.0) or 0.0)
        self.augment_mode = str(
            getattr(augment_config, "imbalance_mode", "minority_only") or "minority_only"
        ).lower()
        self.nli_generator = nli_generator
        # Detect if tokenizer produces token_type_ids
        test_encoding = self.tokenizer("test", "test", return_tensors="pt")
        self.has_token_type_ids = "token_type_ids" in test_encoding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample as (premise, hypothesis, label) tuple."""
        row = self.data.iloc[idx]
        label_value = int(row["label"])

        # Get premise (evidence text)
        premise = str(row.get("sentence_text", row.get("post", "")))

        # Get hypothesis (criterion template)
        if self.nli_generator and "criterion_id" in row:
            # NLI mode: Generate hypothesis from criterion_id using template
            hypothesis = self.nli_generator.generate_hypothesis(row["criterion_id"])
        else:
            # Legacy mode: Use criterion column directly
            hypothesis = str(row.get("criterion", ""))

        # For backward compatibility, use post_text and criterion_text aliases
        post_text = premise
        criterion_text = hypothesis

        if self.augmenter:
            evidence_text = str(row.get("evidence_text", "") or "").strip()
            prob = self.augment_prob
            label_eligible = label_value == 1 or self.augment_mode == "all"
            should_augment = (
                label_eligible and bool(evidence_text) and random.random() < prob
            )
            if should_augment:
                try:
                    augmented_span = self.augmenter(evidence_text)
                    if isinstance(augmented_span, list):
                        augmented_span = augmented_span[0]
                    if evidence_text and evidence_text in post_text:
                        post_text = post_text.replace(evidence_text, str(augmented_span), 1)
                except Exception:
                    # If augmentation fails for any reason, fall back to original text
                    pass

        encoding = self.tokenizer(
            post_text,
            criterion_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_value, dtype=torch.long),
        }
        if self.has_token_type_ids and "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        # Add criterion_id if available (useful for per-criterion analysis)
        if "criterion_id" in row:
            item["criterion_id"] = row["criterion_id"]

        return item


def _supports_multiprocessing() -> bool:
    """Return True if Python multiprocessing primitives are usable."""
    try:
        ctx = mp.get_context()
        queue = ctx.Queue(maxsize=1)
        queue.put_nowait(None)
        queue.close()
        queue.join_thread()
        return True
    except (OSError, PermissionError):
        return False


def seed_worker(worker_id: int):
    """Seed NumPy and random for deterministic augmentation within workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BucketBatchSampler(Sampler[List[int]]):
    """Bucketed batch sampler that groups sequences of similar length."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Sort indices by length, then create buckets
        sorted_indices = sorted(range(len(lengths)), key=lambda idx: lengths[idx])
        bucket_span = max(batch_size * bucket_size, batch_size)
        buckets = [
            sorted_indices[i : i + bucket_span] for i in range(0, len(sorted_indices), bucket_span)
        ]

        # Shuffle within buckets and assemble batches
        batches: List[List[int]] = []
        for bucket in buckets:
            if shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i : i + batch_size]
                if batch:
                    batches.append(batch)

        if shuffle:
            random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def _approximate_lengths(dataset: DSM5NLIDataset) -> List[int]:
    """Approximate token lengths for bucketing without full tokenization."""
    lengths: List[int] = []
    for row in dataset.data.itertuples(index=False):
        post_text = getattr(row, "post", None) or getattr(row, "sentence_text", "") or ""
        criterion_text = getattr(row, "criterion", "") or ""
        lengths.append(len(str(post_text)) + len(str(criterion_text)))
    return lengths


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    bucket_by_length: bool = False,
    bucket_size: int = 50,
    seed: Optional[int] = None,
):
    """Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive across epochs
        bucket_by_length: Use length-aware bucketed batching for training loader
        bucket_size: Number of samples per bucket (multiplied by batch_size)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    use_pin_memory = pin_memory and torch.cuda.is_available()
    generator = None
    worker_init = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init = seed_worker

    # Some sandboxes disable semaphores/shared memory; fall back to single-process loading
    worker_count = num_workers
    if worker_count > 0 and not _supports_multiprocessing():
        warnings.warn(
            "Multiprocessing dataloaders are not available; falling back to num_workers=0.",
            RuntimeWarning,
        )
        worker_count = 0

    persistent = persistent_workers and worker_count > 0

    if bucket_by_length:
        lengths = _approximate_lengths(train_dataset)
        batch_sampler = BucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            bucket_size=bucket_size,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=worker_count,
            pin_memory=use_pin_memory,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=use_pin_memory,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
            generator=generator,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init,
        generator=generator,
    )

    return train_loader, val_loader
