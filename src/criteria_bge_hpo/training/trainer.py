"""Training loop with GPU optimizations."""

import math
import warnings
from typing import Optional

import torch
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

from criteria_bge_hpo.utils.visualization import TrainingProgressDisplay, console


def create_optimizer_and_scheduler(
    model,
    train_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    use_fused,
    optimizer_type="adamw_torch_fused",
    gradient_accumulation_steps=1,
):
    """Create optimizer and scheduler.

    Args:
        model: Model to optimize
        train_loader: Training dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        use_fused: Use fused AdamW
        optimizer_type: Which optimizer variant to use
        gradient_accumulation_steps: Steps to accumulate before optimizer step

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate parameters: no weight decay for bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Create optimizer
    optimizer_choice = (optimizer_type or "adamw").lower()
    optimizer = None

    if optimizer_choice == "adamw_bnb_8bit":
        try:  # pragma: no cover - optional dependency
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        except Exception as exc:
            warnings.warn(
                f"adamw_bnb_8bit requested but unavailable ({exc}); "
                f"falling back to torch.AdamW (fused={use_fused})."
            )
            optimizer_choice = "adamw_torch_fused" if use_fused else "adamw"

    if optimizer is None:
        fused_flag = use_fused and torch.cuda.is_available() and optimizer_choice != "adamw"
        if fused_flag:
            try:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                    fused=True,
                )
            except RuntimeError:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
    )

    # Create scheduler
    effective_batch_per_epoch = max(
        1, math.ceil(len(train_loader) / gradient_accumulation_steps)
    )
    num_training_steps = effective_batch_per_epoch * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


class Trainer:
    """Trainer with GPU optimizations."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        use_bf16=False,
        amp_dtype=None,
        use_grad_scaler=False,
        use_compile=False,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        non_blocking: bool = True,
        mlflow_enabled=True,
        early_stopping_patience=None,
        checkpoint_dir=None,
        positive_threshold: float = 0.5,
        enable_progress: bool = True,
        total_folds: int = 1,
        progress_refresh_per_second: int = 4,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_bf16 = use_bf16 and torch.cuda.is_available()
        if self.amp_dtype is None and self.use_bf16:
            self.amp_dtype = torch.bfloat16
        self.use_amp = self.amp_dtype is not None and torch.cuda.is_available()
        self.use_grad_scaler = (
            bool(use_grad_scaler) and self.use_amp and self.amp_dtype == torch.float16
        )
        self.scaler = GradScaler(enabled=self.use_grad_scaler)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.non_blocking = non_blocking and torch.cuda.is_available()
        self.mlflow_enabled = mlflow_enabled
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.best_epoch = 0
        self.best_state_dict = None
        self.positive_threshold = positive_threshold
        self.enable_progress = enable_progress
        self.total_folds = max(1, total_folds)
        self.progress_refresh_per_second = progress_refresh_per_second

        # Apply torch.compile if requested
        if use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except RuntimeError:
                pass

        self.model.to(self.device)
        self.best_val_f1 = float("-inf")

    def train(self, num_epochs, fold):
        """Train for num_epochs."""
        epochs_without_improvement = 0
        patience = self.early_stopping_patience
        progress_display: Optional[TrainingProgressDisplay] = None
        stop_message: Optional[str] = None

        if self.enable_progress:
            progress_display = TrainingProgressDisplay(
                total_epochs=num_epochs,
                steps_per_epoch=len(self.train_loader),
                fold=fold,
                total_folds=self.total_folds,
                patience=patience,
                refresh_per_second=self.progress_refresh_per_second,
            )
            progress_display.start()

        try:
            for epoch in range(1, num_epochs + 1):
                # Train epoch
                self.model.train()

                if progress_display:
                    progress_display.start_epoch(epoch, len(self.train_loader))

                epoch_loss = 0.0
                epoch_steps = 0

                for step, batch in enumerate(self.train_loader):
                    # Move only tensor values to device, skip metadata fields like criterion_id
                    batch = {
                        k: v.to(self.device, non_blocking=self.non_blocking)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }

                    # Forward
                    with torch.amp.autocast(
                        "cuda",
                        enabled=self.use_amp,
                        dtype=self.amp_dtype or torch.bfloat16,
                    ):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / self.gradient_accumulation_steps

                    # Backward
                    if self.use_grad_scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    scaled_loss = loss.item() * self.gradient_accumulation_steps

                    epoch_loss += scaled_loss
                    epoch_steps += 1

                    # Update weights
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_grad_scaler:
                            self.scaler.unscale_(self.optimizer)
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                        if self.use_grad_scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    if progress_display:
                        progress_display.advance_step(scaled_loss)

                # Evaluate
                val_f1 = self._evaluate()
                new_best = val_f1 > self.best_val_f1
                if new_best:
                    self.best_val_f1 = val_f1
                    self.best_epoch = epoch
                    epochs_without_improvement = 0

                    model_to_save = self._get_unwrapped_model()
                    self._store_best_state(model_to_save)

                    # Save best model to disk if requested
                    if self.checkpoint_dir:
                        import os

                        model_save_dir = os.path.join(self.checkpoint_dir, f"fold_{fold}")
                        model_to_save.save_pretrained(model_save_dir)
                        status_message = (
                            f"Saved new best model to {model_save_dir} (F1: {val_f1:.4f})"
                        )
                        if self.mlflow_enabled:
                            console.log(status_message)
                        else:
                            console.print(status_message)
                        if progress_display:
                            progress_display.update_status(status_message, style="green")
                else:
                    epochs_without_improvement += 1

                avg_loss = epoch_loss / max(1, epoch_steps)
                patience_left = (
                    patience - epochs_without_improvement if patience is not None else None
                )
                if progress_display:
                    progress_display.finish_epoch(
                        val_f1=val_f1,
                        best_f1=self.best_val_f1,
                        best_epoch=self.best_epoch,
                        current_lr=self.optimizer.param_groups[0]["lr"],
                        patience_left=patience_left,
                    )
                    if new_best and not self.checkpoint_dir:
                        progress_display.update_status(
                            f"New best F1 {val_f1:.4f} at epoch {epoch}",
                            style="green",
                        )
                else:
                    console.print(
                        f"Epoch {epoch}/{num_epochs} "
                        f"- loss: {avg_loss:.4f}, val_f1: {val_f1:.4f}"
                    )

                if (
                    patience is not None
                    and patience > 0
                    and epochs_without_improvement >= patience
                ):
                    stop_message = (
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best F1 {self.best_val_f1:.4f} at epoch {self.best_epoch})"
                    )
                    if progress_display:
                        progress_display.update_status(stop_message, style="yellow")
                    break
        finally:
            if progress_display:
                progress_display.stop()

        if stop_message:
            console.print(stop_message)

        self._restore_best_state()

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            # Move only tensor values to device, skip metadata fields like criterion_id
            batch = {
                k: v.to(self.device, non_blocking=self.non_blocking)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.bfloat16,
            ):
                outputs = self.model(**batch)

            logits = outputs["logits"]
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs >= self.positive_threshold).long()
            else:
                preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average="binary")
        return f1

    def _get_unwrapped_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _store_best_state(self, reference_model):
        self.best_state_dict = {
            k: v.detach().cpu().clone() for k, v in reference_model.state_dict().items()
        }

    def _restore_best_state(self):
        if not self.best_state_dict:
            return
        reference_model = self._get_unwrapped_model()
        reference_model.load_state_dict(self.best_state_dict)
