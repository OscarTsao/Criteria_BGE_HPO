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
        Tuple[np.ndarray, np.ndarray]: (train_idx, val_idx) indices for each fold

    Example:
        >>> for fold, (train_idx, val_idx) in enumerate(create_kfold_splits(data)):
        ...     print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
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
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]

        # Critical assertion: Verify no post appears in both train and validation
        # This prevents the model from seeing the same post text during training
        # when evaluating on validation set (data leakage)
        train_posts = set(train_df["post_id"].unique())
        val_posts = set(val_df["post_id"].unique())
        assert len(train_posts & val_posts) == 0, f"Fold {fold_idx}: Data leakage detected!"

        yield train_idx, val_idx


def get_fold_statistics(data: pd.DataFrame, splits):
    """Compute statistics for each fold.

    Useful for verifying:
    - Similar fold sizes (balanced splitting)
    - Consistent class distributions (stratification working)
    - Post grouping effectiveness

    Args:
        data: Full dataset DataFrame
        splits: Iterable of (train_idx, val_idx) tuples from create_kfold_splits()

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
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]
        stats.append(
            {
                "Fold": fold_idx,
                "Train Size": len(train_df),
                "Val Size": len(val_df),
                "Train Posts": train_df["post_id"].nunique(),
                "Val Posts": val_df["post_id"].nunique(),
                # Calculate positive label percentage for stratification verification
                "Train Pos%": f"{(train_df['label']==1).mean()*100:.1f}%",
                "Val Pos%": f"{(val_df['label']==1).mean()*100:.1f}%",
            }
        )
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


def create_nested_kfold_splits(
    data: pd.DataFrame,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int = 42,
):
    """Create nested K-fold splits for hyperparameter optimization.

    Implements true nested cross-validation with:
    - Outer CV loop (5 folds): For unbiased performance estimation
    - Inner CV loop (3-5 folds): For hyperparameter optimization per outer fold

    CRITICAL: Both outer and inner splits use GroupKFold by post_id to prevent
    data leakage. This ensures no post appears in multiple splits.

    Nested CV Structure:
        For each outer fold i:
            - Outer test set: Fold i data (held out for final evaluation)
            - Outer train set: All other folds
                For each inner fold j within outer train:
                    - Inner val set: Fold j data (for HPO validation)
                    - Inner train set: Remaining outer train data
                    - Run trial with these inner splits
                Best HP from inner CV used to train on full outer train
            Evaluate on outer test set

    Args:
        data: DataFrame with columns: post_id, label, (plus post/criterion or sentence_text/criterion_id)
        n_outer_splits: Number of outer folds (default: 5)
        n_inner_splits: Number of inner folds per outer fold (default: 3)
        random_state: Random seed for reproducible splits

    Yields:
        Tuple containing:
            - outer_fold_idx: Outer fold index (0 to n_outer_splits-1)
            - outer_train_idx: Indices for outer training set
            - outer_test_idx: Indices for outer test set
            - inner_splits: List of (inner_train_idx, inner_val_idx) tuples for HPO

    Example:
        >>> for outer_fold, outer_train, outer_test, inner_splits in create_nested_kfold_splits(data):
        ...     print(f"Outer Fold {outer_fold}")
        ...     print(f"  Outer Train: {len(outer_train)}, Outer Test: {len(outer_test)}")
        ...     print(f"  Inner CV: {len(inner_splits)} folds for HPO")
        ...
        ...     # Hyperparameter optimization on inner splits
        ...     for inner_fold, (inner_train, inner_val) in enumerate(inner_splits):
        ...         print(f"    Inner Fold {inner_fold}: Train={len(inner_train)}, Val={len(inner_val)}")

    Computational Cost:
        Total fold-epochs = n_outer × n_trials × n_inner × avg_epochs
        Example: 5 × 100 × 3 × 20 = 30,000 fold-epochs
    """
    # Outer CV loop: Performance estimation
    outer_splitter = StratifiedGroupKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )

    X = np.arange(len(data))
    y = data["label"].values
    groups = data["post_id"].values

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(X, y, groups)
    ):
        # Verify no data leakage between outer train and test
        outer_train_df = data.iloc[outer_train_idx]
        outer_test_df = data.iloc[outer_test_idx]

        outer_train_posts = set(outer_train_df["post_id"].unique())
        outer_test_posts = set(outer_test_df["post_id"].unique())

        assert (
            len(outer_train_posts & outer_test_posts) == 0
        ), f"Outer fold {outer_fold_idx}: Data leakage detected between train and test!"

        # Inner CV loop: Hyperparameter optimization on outer train set
        inner_splits = []

        # Create inner splits ONLY from outer training data
        inner_splitter = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state + outer_fold_idx,  # Different seed per outer fold
        )

        # Map indices: inner splitter works on outer_train_df indices [0, 1, 2, ...]
        # but we need to map back to original data indices
        X_inner = np.arange(len(outer_train_df))
        y_inner = outer_train_df["label"].values
        groups_inner = outer_train_df["post_id"].values

        for inner_train_rel, inner_val_rel in inner_splitter.split(X_inner, y_inner, groups_inner):
            # Map relative indices back to absolute indices in original data
            inner_train_abs = outer_train_idx[inner_train_rel]
            inner_val_abs = outer_train_idx[inner_val_rel]

            # Verify no data leakage between inner train and val
            inner_train_df = data.iloc[inner_train_abs]
            inner_val_df = data.iloc[inner_val_abs]

            inner_train_posts = set(inner_train_df["post_id"].unique())
            inner_val_posts = set(inner_val_df["post_id"].unique())

            assert (
                len(inner_train_posts & inner_val_posts) == 0
            ), f"Outer fold {outer_fold_idx}: Inner split has data leakage!"

            # Also verify inner splits don't leak into outer test
            assert (
                len(inner_train_posts & outer_test_posts) == 0
            ), f"Outer fold {outer_fold_idx}: Inner train leaks into outer test!"
            assert (
                len(inner_val_posts & outer_test_posts) == 0
            ), f"Outer fold {outer_fold_idx}: Inner val leaks into outer test!"

            inner_splits.append((inner_train_abs, inner_val_abs))

        yield outer_fold_idx, outer_train_idx, outer_test_idx, inner_splits


def display_nested_fold_statistics(data: pd.DataFrame, nested_splits):
    """Display statistics for nested K-fold splits.

    Shows both outer and inner fold information for verification.

    Args:
        data: Full dataset DataFrame
        nested_splits: Iterator from create_nested_kfold_splits()
    """
    console.print("\n[bold cyan]Nested K-Fold Statistics[/bold cyan]\n")

    for outer_fold, outer_train, outer_test, inner_splits in nested_splits:
        outer_train_df = data.iloc[outer_train]
        outer_test_df = data.iloc[outer_test]

        # Outer fold info
        table = Table(title=f"Outer Fold {outer_fold}", show_header=True)
        table.add_column("Split", style="cyan")
        table.add_column("Samples", style="yellow")
        table.add_column("Posts", style="green")
        table.add_column("Pos%", style="magenta")

        table.add_row(
            "Train",
            str(len(outer_train_df)),
            str(outer_train_df["post_id"].nunique()),
            f"{(outer_train_df['label']==1).mean()*100:.1f}%",
        )
        table.add_row(
            "Test",
            str(len(outer_test_df)),
            str(outer_test_df["post_id"].nunique()),
            f"{(outer_test_df['label']==1).mean()*100:.1f}%",
        )

        console.print(table)

        # Inner fold info
        inner_table = Table(title=f"  Inner CV for Outer Fold {outer_fold}", show_header=True)
        inner_table.add_column("Inner Fold", style="cyan")
        inner_table.add_column("Train", style="yellow")
        inner_table.add_column("Val", style="yellow")
        inner_table.add_column("Train Posts", style="green")
        inner_table.add_column("Val Posts", style="green")

        for inner_idx, (inner_train, inner_val) in enumerate(inner_splits):
            inner_train_df = data.iloc[inner_train]
            inner_val_df = data.iloc[inner_val]

            inner_table.add_row(
                str(inner_idx),
                str(len(inner_train_df)),
                str(len(inner_val_df)),
                str(inner_train_df["post_id"].nunique()),
                str(inner_val_df["post_id"].nunique()),
            )

        console.print(inner_table)
        console.print()  # Blank line between outer folds
