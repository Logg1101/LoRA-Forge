"""
Training Hyperparameters & Optimization Page.
Controls: Batch size, Epochs, Max steps, Grad accumulation, Grad clipping,
UNet LR, Text Encoder LR, Text Encoder training toggle, Optimizer, Scheduler, Warmup, Weight Decay.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig


class TrainingPage(QWidget):
    """Page 4: Training hyperparameters, optimization algorithms, and learning rate scheduling."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("TRAINING HYPERPARAMETERS & OPTIMIZATION")
        title.setProperty("class", "section-title")
        desc = QLabel("Set iteration length, batch accumulation, separate UNet/Text-Encoder learning rates, optimizers, and schedulers.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Epochs & Batches ──
        grp_iter = QFrame()
        grp_iter.setProperty("class", "group-box")
        g1 = QGridLayout(grp_iter)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("ITERATION BUDGET & BATCH ACCUMULATION")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Batch size
        lbl_batch = QLabel("Batch Size:")
        lbl_batch.setProperty("class", "field-label")
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        self.spin_batch.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_batch, 1, 0)
        g1.addWidget(self.spin_batch, 1, 1)

        # Grad accum
        lbl_accum = QLabel("Gradient Accumulation:")
        lbl_accum.setProperty("class", "field-label")
        self.spin_accum = QSpinBox()
        self.spin_accum.setRange(1, 128)
        self.spin_accum.setValue(1)
        self.spin_accum.setToolTip("Accumulate gradients over N steps before updating weights. Emulates larger effective batch size.")
        self.spin_accum.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_accum, 1, 2)
        g1.addWidget(self.spin_accum, 1, 3)

        # Epochs
        lbl_ep = QLabel("Epochs:")
        lbl_ep.setProperty("class", "field-label")
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(10)
        self.spin_epochs.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_ep, 2, 0)
        g1.addWidget(self.spin_epochs, 2, 1)

        # Max Steps
        lbl_steps = QLabel("Max Steps (0 = by Epochs):")
        lbl_steps.setProperty("class", "field-label")
        self.spin_max_steps = QSpinBox()
        self.spin_max_steps.setRange(0, 1000000)
        self.spin_max_steps.setSingleStep(100)
        self.spin_max_steps.setValue(0)
        self.spin_max_steps.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_steps, 2, 2)
        g1.addWidget(self.spin_max_steps, 2, 3)

        layout.addWidget(grp_iter)

        # ── Group 2: Learning Rates & Encoders ──
        grp_lr = QFrame()
        grp_lr.setProperty("class", "group-box")
        g2 = QGridLayout(grp_lr)
        g2.setSpacing(10)
        g2.setContentsMargins(14, 12, 14, 12)

        g2_title = QLabel("LEARNING RATES & ENCODER TRAINING")
        g2_title.setProperty("class", "section-title")
        g2.addWidget(g2_title, 0, 0, 1, 4)

        # UNet LR
        lbl_unet_lr = QLabel("UNet / DiT Learning Rate:")
        lbl_unet_lr.setProperty("class", "field-label")
        self.spin_unet_lr = QDoubleSpinBox()
        self.spin_unet_lr.setRange(1e-7, 1.0)
        self.spin_unet_lr.setDecimals(7)
        self.spin_unet_lr.setSingleStep(1e-5)
        self.spin_unet_lr.setValue(1e-4)
        self.spin_unet_lr.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_unet_lr, 1, 0)
        g2.addWidget(self.spin_unet_lr, 1, 1)

        # TE LR
        lbl_te_lr = QLabel("Text Encoder Learning Rate:")
        lbl_te_lr.setProperty("class", "field-label")
        self.spin_te_lr = QDoubleSpinBox()
        self.spin_te_lr.setRange(1e-7, 1.0)
        self.spin_te_lr.setDecimals(7)
        self.spin_te_lr.setSingleStep(1e-5)
        self.spin_te_lr.setValue(5e-5)
        self.spin_te_lr.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_te_lr, 1, 2)
        g2.addWidget(self.spin_te_lr, 1, 3)

        # Train TE Toggle
        self.chk_train_te = QCheckBox("Train Text Encoder(s)")
        self.chk_train_te.setToolTip("Fine-tunes text encoder layers alongside LoRA. Requires 'Cache Latents' to be OFF.")
        self.chk_train_te.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_train_te, 2, 0, 1, 2)

        # Gradient clipping
        lbl_clip = QLabel("Gradient Clip Norm (0 = Off):")
        lbl_clip.setProperty("class", "field-label")
        self.spin_clip = QDoubleSpinBox()
        self.spin_clip.setRange(0.0, 10.0)
        self.spin_clip.setSingleStep(0.1)
        self.spin_clip.setValue(1.0)
        self.spin_clip.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_clip, 2, 2)
        g2.addWidget(self.spin_clip, 2, 3)

        layout.addWidget(grp_lr)

        # ── Group 3: Optimizer & Scheduler ──
        grp_opt = QFrame()
        grp_opt.setProperty("class", "group-box")
        g3 = QGridLayout(grp_opt)
        g3.setSpacing(10)
        g3.setContentsMargins(14, 12, 14, 12)

        g3_title = QLabel("OPTIMIZER & LEARNING RATE SCHEDULE")
        g3_title.setProperty("class", "section-title")
        g3.addWidget(g3_title, 0, 0, 1, 4)

        # Optimizer
        lbl_opt = QLabel("Optimizer:")
        lbl_opt.setProperty("class", "field-label")
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(["AdamW8bit", "AdamW", "Prodigy"])
        self.combo_opt.currentTextChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_opt, 1, 0)
        g3.addWidget(self.combo_opt, 1, 1)

        # Scheduler
        lbl_sched = QLabel("LR Scheduler:")
        lbl_sched.setProperty("class", "field-label")
        self.combo_sched = QComboBox()
        self.combo_sched.addItems(["cosine", "linear", "constant", "cosine_with_warmup", "constant_with_warmup"])
        self.combo_sched.currentTextChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_sched, 1, 2)
        g3.addWidget(self.combo_sched, 1, 3)

        # Warmup Ratio
        lbl_warm = QLabel("Warmup Ratio:")
        lbl_warm.setProperty("class", "field-label")
        self.spin_warmup_ratio = QDoubleSpinBox()
        self.spin_warmup_ratio.setRange(0.0, 1.0)
        self.spin_warmup_ratio.setSingleStep(0.01)
        self.spin_warmup_ratio.setValue(0.0)
        self.spin_warmup_ratio.setToolTip("Fraction of total steps dedicated to learning rate warmup.")
        self.spin_warmup_ratio.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_warm, 2, 0)
        g3.addWidget(self.spin_warmup_ratio, 2, 1)

        # Weight Decay
        lbl_wd = QLabel("Weight Decay:")
        lbl_wd.setProperty("class", "field-label")
        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(0.0, 1.0)
        self.spin_wd.setSingleStep(0.001)
        self.spin_wd.setDecimals(4)
        self.spin_wd.setValue(0.01)
        self.spin_wd.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_wd, 2, 2)
        g3.addWidget(self.spin_wd, 2, 3)

        # Prodigy D-Coef
        lbl_d_coef = QLabel("Prodigy D-Coef:")
        lbl_d_coef.setProperty("class", "field-label")
        self.spin_prodigy_d_coef = QDoubleSpinBox()
        self.spin_prodigy_d_coef.setRange(0.1, 10.0)
        self.spin_prodigy_d_coef.setSingleStep(0.1)
        self.spin_prodigy_d_coef.setValue(1.0)
        self.spin_prodigy_d_coef.setToolTip("Coefficient for Prodigy D estimate (default 1.0; 0.5-2.0 typical).")
        self.spin_prodigy_d_coef.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_d_coef, 3, 0)
        g3.addWidget(self.spin_prodigy_d_coef, 3, 1)

        # Prodigy Bias Correction
        self.chk_prodigy_bias = QCheckBox("Prodigy Bias Correction")
        self.chk_prodigy_bias.setChecked(True)
        self.chk_prodigy_bias.setToolTip("Enables bias correction in the Prodigy adaptive optimizer.")
        self.chk_prodigy_bias.stateChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(self.chk_prodigy_bias, 3, 2, 1, 2)

        layout.addWidget(grp_opt)
        layout.addStretch()

    def load_config(self, config: TrainingConfig):
        self.spin_batch.setValue(config.batch_size)
        self.spin_accum.setValue(config.grad_accum_steps)
        self.spin_epochs.setValue(config.epochs)
        self.spin_max_steps.setValue(config.max_train_steps)
        self.spin_unet_lr.setValue(config.learning_rate)
        self.spin_te_lr.setValue(config.text_encoder_lr)
        self.chk_train_te.setChecked(config.text_encoder_training)
        self.spin_clip.setValue(config.gradient_clip_norm)
        self.combo_opt.setCurrentText(config.optimizer)
        self.combo_sched.setCurrentText(config.lr_scheduler)
        self.spin_warmup_ratio.setValue(config.warmup_ratio)
        self.spin_wd.setValue(config.weight_decay)
        self.spin_prodigy_d_coef.setValue(config.prodigy_d_coef)
        self.chk_prodigy_bias.setChecked(config.prodigy_use_bias_correction)

    def save_config(self, config: TrainingConfig):
        config.batch_size = self.spin_batch.value()
        config.grad_accum_steps = self.spin_accum.value()
        config.epochs = self.spin_epochs.value()
        config.max_train_steps = self.spin_max_steps.value()
        config.learning_rate = self.spin_unet_lr.value()
        config.text_encoder_lr = self.spin_te_lr.value()
        config.text_encoder_training = self.chk_train_te.isChecked()
        config.gradient_clip_norm = self.spin_clip.value()
        config.optimizer = self.combo_opt.currentText()
        config.lr_scheduler = self.combo_sched.currentText()
        config.warmup_ratio = self.spin_warmup_ratio.value()
        config.weight_decay = self.spin_wd.value()
        config.prodigy_d_coef = self.spin_prodigy_d_coef.value()
        config.prodigy_use_bias_correction = self.chk_prodigy_bias.isChecked()
