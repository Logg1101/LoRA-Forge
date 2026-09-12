"""
Training Plan Preview Widget that calculates and displays comprehensive metrics
(images, repeats, effective images, steps/epoch, total steps, learning rates, etc.)
before the user commits to training.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig


class TrainingPlanWidget(QFrame):
    """Calculates and renders a structured training plan summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "group-box")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title = QLabel("TRAINING PLAN PREVIEW")
        title.setProperty("class", "section-title")
        header.addWidget(title)
        header.addStretch()
        self.lbl_arch_badge = QLabel("SDXL")
        self.lbl_arch_badge.setProperty("class", "meta-badge")
        header.addWidget(self.lbl_arch_badge)
        layout.addLayout(header)

        # Summary Grid
        self.grid = QGridLayout()
        self.grid.setSpacing(8)

        # Metric Labels
        self.val_dataset = self._add_row("Dataset:", "0 images × 1 repeat", 0)
        self.val_effective = self._add_row("Effective Size:", "0 images / epoch", 1)
        self.val_batch_epoch = self._add_row("Batch & Epochs:", "Batch 1, 10 Epochs", 2)
        self.val_steps = self._add_row("Total Steps:", "0 steps (0 / epoch)", 3)
        self.val_network = self._add_row("Network Setup:", "LoRA (Rank 16, Alpha 8.0)", 4)
        self.val_rates = self._add_row("Learning Rates:", "UNet: 1e-4 | TE: 5e-5", 5)
        self.val_opt = self._add_row("Opt & Schedule:", "AdamW8bit | cosine", 6)
        self.val_mem = self._add_row("Memory Strategy:", "1024px | bf16 | Latent Caching ON", 7)

        layout.addLayout(self.grid)

    def _add_row(self, title: str, default_val: str, row: int) -> QLabel:
        lbl_title = QLabel(title)
        lbl_title.setProperty("class", "field-label")
        lbl_title.setStyleSheet("color: #8b949e; min-width: 110px;")

        lbl_val = QLabel(default_val)
        lbl_val.setStyleSheet("color: #f0f6fc; font-weight: 600; font-family: monospace;")

        self.grid.addWidget(lbl_title, row, 0)
        self.grid.addWidget(lbl_val, row, 1)
        return lbl_val

    def update_plan(self, config: TrainingConfig, image_count: int = 0):
        """Recompute and update the plan values."""
        plan = config.training_plan(image_count=image_count)

        self.lbl_arch_badge.setText(config.model_type)

        # Dataset row
        repeats = config.dataset_repeats
        self.val_dataset.setText(f"{image_count} images × {repeats} repeat{'s' if repeats > 1 else ''}")

        # Effective size
        eff = plan["effective_images"]
        self.val_effective.setText(f"{eff} effective images / epoch")

        # Batch & Epochs
        self.val_batch_epoch.setText(f"Batch {config.batch_size} (Grad Accum: {config.grad_accum_steps}) | {config.epochs} Epochs")

        # Steps
        steps_epoch = plan["steps_per_epoch"]
        tot_steps = plan["total_steps"]
        max_s_note = f" (capped from max_train_steps)" if config.max_train_steps > 0 else ""
        self.val_steps.setText(f"{tot_steps:,} total steps ({steps_epoch:,} / epoch){max_s_note}")

        # Network
        net_desc = f"{config.network_type.upper()} (Rank {config.rank}, Alpha {config.alpha})"
        if config.conv_rank > 0:
            net_desc += f" [Conv Rank {config.conv_rank}]"
        self.val_network.setText(net_desc)

        # Learning rates
        te_note = f" | TE: {config.text_encoder_lr:.2e}" if config.text_encoder_training else " | TE: Frozen"
        self.val_rates.setText(f"UNet: {config.learning_rate:.2e}{te_note}")

        # Opt & Scheduler
        self.val_opt.setText(f"{config.optimizer} | {config.lr_scheduler}")

        # Memory Strategy
        quant_str = f" | Quant: {config.quantization.upper()}" if config.quantization != "none" else ""
        cache_str = "Cache ON" if config.cache_latents else "Cache OFF"
        ckpt_str = "GradCkpt ON" if config.gradient_checkpointing else "GradCkpt OFF"
        self.val_mem.setText(f"{config.resolution}px | {config.mixed_precision} | {cache_str} | {ckpt_str}{quant_str}")
