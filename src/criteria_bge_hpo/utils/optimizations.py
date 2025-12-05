"""Utilities for applying performance-oriented training settings."""

from __future__ import annotations

import importlib
from typing import Optional

import torch
from rich.console import Console

console = Console()


def select_amp_dtype(use_bf16: bool, enable_fp16: bool = True) -> Optional[torch.dtype]:
    """Choose an autocast dtype based on hardware support and config.

    Returns torch.bfloat16 when BF16 is both requested and supported,
    otherwise falls back to torch.float16 when FP16 is allowed. If CUDA
    is unavailable, returns None so training runs in FP32.
    """
    if not torch.cuda.is_available():
        return None

    if use_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    if enable_fp16:
        return torch.float16

    return None


def should_use_grad_scaler(amp_dtype: Optional[torch.dtype]) -> bool:
    """Enable GradScaler only when running FP16 autocast."""
    return amp_dtype == torch.float16 and torch.cuda.is_available()


def resolve_attention_backend(requested_backend: Optional[str]) -> str:
    """Resolve the requested attention backend with safe fallbacks."""
    backend = (requested_backend or "sdpa").lower()

    if backend == "flash_attention_2":
        try:
            importlib.import_module("flash_attn")
        except Exception as exc:  # pragma: no cover - import guard only
            console.print(
                f"[yellow]⚠ FlashAttention 2 unavailable ({exc}); using SDPA instead[/yellow]"
            )
            backend = "sdpa"

    return backend


def configure_sdp_backend(attn_backend: str) -> None:
    """Enable fused SDPA kernels when using SDPA-style attention."""
    if not torch.cuda.is_available():
        return

    if attn_backend in {"sdpa", "flash_attention_2"}:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            # Older PyTorch builds may not expose SDP toggles; ignore silently.
            pass
