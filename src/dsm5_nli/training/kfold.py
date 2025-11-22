"""K-fold cross-validation with grouped splitting.

Uses StratifiedGroupKFold to prevent data leakage in post-criterion pairs.

CRITICAL DATA LEAKAGE PREVENTION
=================================
Problem:
    A single post may have multiple criterion annotations.
    Example: Post_123 matched against Criterion_A, Criterion_B, etc.

Leakage Risk:
    If Post_123+Criterion_A is in training set and Post_123+Criterion_B is in
    validation set, the model sees Post_123's text during training, creating
    an unfair advantage when evaluating on validation.

Solution:
    StratifiedGroupKFold groups all pairs from the same post_id together.
    All pairs from Post_123 go to EITHER train OR validation, never split.

Stratification:
    Additionally maintains class balance (positive/negative label ratios)
    across folds for stable validation metrics.

Example:
    Post_A + Criterion_1 (label=1) -> Fold 0 (train)
    Post_A + Criterion_2 (label=0) -> Fold 0 (train)  # Same fold due to grouping
    Post_B + Criterion_1 (label=1) -> Fold 1 (val)
    Post_B + Criterion_3 (label=0) -> Fold 1 (val)    # Same fold due to grouping
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from rich.console import Console
from rich.table import Table

console = Console()


def create_kfold_splits(data: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """Create K-fold splits with grouped stratification.

    Prevents data leakage by ensuring all pairs from the same post stay together.

    Args:
        data: DataFrame with columns: post_id, label, post, criterion
        n_splits: Number of folds (default: 5 for standard CV)
        random_state: Random seed for reproducible splits

    Yields:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df) for each fold

    Example:
        >>> for fold, (train_df, val_df) in enumerate(create_kfold_splits(data)):
        ...     print(f"Fold {fold}: {len(train_df)} train, {len(val_df)} val")
    """
    # Create dummy X (indices) - sklearn requires it but we only need y and groups
    X = np.arange(len(data))
    y = data["label"].values  # Binary labels for stratification
    groups = data["post_id"].values  # Group by post to prevent leakage

    # StratifiedGroupKFold parameters:
    # - groups: Ensures all samples with same post_id stay together
    # - y: Maintains class balance across folds (stratification)
    # - shuffle=True: Randomizes fold assignment (controlled by random_state)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)

        # Critical assertion: Verify no post appears in both train and validation
        # This prevents the model from seeing the same post text during training
        # when evaluating on validation set (data leakage)
        train_posts = set(train_df["post_id"].unique())
        val_posts = set(val_df["post_id"].unique())
        assert len(train_posts & val_posts) == 0, f"Fold {fold_idx}: Data leakage detected!"

        yield train_df, val_df


def get_fold_statistics(data: pd.DataFrame, splits):
    """Compute statistics for each fold.

    Useful for verifying:
    - Similar fold sizes (balanced splitting)
    - Consistent class distributions (stratification working)
    - Post grouping effectiveness

    Args:
        data: Full dataset DataFrame
        splits: Iterator of (train_df, val_df) tuples from create_kfold_splits()

    Returns:
        pd.DataFrame: Statistics table with columns:
            - Fold: Fold index
            - Train Size: Number of training samples
            - Val Size: Number of validation samples
            - Train Posts: Unique posts in training set
            - Val Posts: Unique posts in validation set
            - Train Pos%: Percentage of positive labels in train
            - Val Pos%: Percentage of positive labels in val
    """
    stats = []
    for fold_idx, (train_df, val_df) in enumerate(splits):
        stats.append({
            "Fold": fold_idx,
            "Train Size": len(train_df),
            "Val Size": len(val_df),
            "Train Posts": train_df["post_id"].nunique(),
            "Val Posts": val_df["post_id"].nunique(),
            # Calculate positive label percentage for stratification verification
            "Train Pos%": f"{(train_df['label']==1).mean()*100:.1f}%",
            "Val Pos%": f"{(val_df['label']==1).mean()*100:.1f}%",
        })
    return pd.DataFrame(stats)


def display_fold_statistics(stats_df: pd.DataFrame):
    """Display fold statistics in formatted Rich table.

    Creates visual table showing fold-by-fold statistics for quick verification
    of split quality (balance, stratification, grouping).

    Args:
        stats_df: DataFrame from get_fold_statistics()
    """
    table = Table(title="K-Fold Statistics", show_header=True)

    # Add all columns from stats DataFrame
    for col in stats_df.columns:
        table.add_column(col, style="cyan")

    # Add each fold as a row
    for _, row in stats_df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)
