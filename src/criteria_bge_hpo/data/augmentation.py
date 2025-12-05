"""Factory for text augmentation utilities."""

from __future__ import annotations

from typing import Callable, Optional

import torch

# TextAttack is an optional, research-only dependency. Keep it behind a flag to
# avoid import failures unless explicitly enabled.
ENABLE_TEXTATTACK_LIBRARY = False


class AugmentationFactory:
    """Create augmentation callables based on configuration."""

    @staticmethod
    def get_augmenter(config) -> Optional[Callable[[str], str]]:
        """Return a callable augmenter or None if disabled."""
        if config is None or not getattr(config, "enable", False):
            return None

        lib = str(getattr(config, "lib", "nlpaug")).lower()
        aug_type = str(getattr(config, "type", "")).lower()

        if lib == "nlpaug":
            try:
                import nlpaug.augmenter.word as naw
            except ImportError as exc:
                raise ImportError(
                    "nlpaug is required for augmentation. Install via `pip install nlpaug`."
                ) from exc

            # Strip "nlpaug_" prefix if present (for compatibility with search space)
            aug_type_clean = aug_type.replace("nlpaug_", "") if aug_type.startswith("nlpaug_") else aug_type

            if aug_type_clean == "contextual":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                augmenter = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased",
                    action="substitute",
                    device=device,
                )
            elif aug_type_clean == "synonym":
                augmenter = naw.SynonymAug(aug_src="wordnet")
            else:
                raise ValueError(f"Unsupported nlpaug augmentation type: {aug_type}")

        elif lib == "textattack" or aug_type.startswith("textattack"):
            if not ENABLE_TEXTATTACK_LIBRARY:
                raise ValueError(
                    "TextAttack augmentation requested but ENABLE_TEXTATTACK_LIBRARY is False. "
                    "Set ENABLE_TEXTATTACK_LIBRARY=True to allow TextAttack-based augmenters."
                )

            try:  # pragma: no cover - optional dependency
                from textattack.augmentation import EasyDataAugmenter, EmbeddingAugmenter
            except Exception as exc:
                raise ImportError(
                    "TextAttack is required for this augmentation. Install via `pip install textattack`."
                ) from exc

            if aug_type in {"textattack_eda", "eda"}:
                augmenter = EasyDataAugmenter()
            elif aug_type in {"textattack_embedding", "embedding"}:
                augmenter = EmbeddingAugmenter()
            else:
                raise ValueError(f"Unsupported TextAttack augmentation type: {aug_type}")

        else:
            raise ValueError(f"Unsupported augmentation library: {lib}")

        return lambda text: augmenter.augment(text)
