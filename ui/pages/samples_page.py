"""
Automatic In-Training Sample Generation Page.
Controls: Enable Samples, Intervals (steps/epochs), Multiple Prompt Editor,
Negative Prompt, Steps, CFG, Seed, Sampler, and Sample Resolution.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QPlainTextEdit, QLineEdit, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig


class SamplesPage(QWidget):
    """Page 7: Automated sample generation during training checkpoints."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("AUTOMATED SAMPLE GENERATION")
        title.setProperty("class", "section-title")
        desc = QLabel("Generate test images at configured step or epoch intervals using fixed seeds to visually track LoRA progression.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: General Sampling Rules ──
        grp_rules = QFrame()
        grp_rules.setProperty("class", "group-box")
        g1 = QGridLayout(grp_rules)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("SAMPLING INTERVALS & INFERENCE CONTROLS")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Enable checkbox
        self.chk_enable = QCheckBox("Enable Sample Generation During Training")
        self.chk_enable.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_enable, 1, 0, 1, 2)

        # Sample every N epochs
        lbl_ep = QLabel("Sample Every N Epochs:")
        lbl_ep.setProperty("class", "field-label")
        self.spin_every_epochs = QSpinBox()
        self.spin_every_epochs.setRange(0, 100)
        self.spin_every_epochs.setValue(1)
        self.spin_every_epochs.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_ep, 2, 0)
        g1.addWidget(self.spin_every_epochs, 2, 1)

        # Sample every N steps
        lbl_st = QLabel("Sample Every N Steps:")
        lbl_st.setProperty("class", "field-label")
        self.spin_every_steps = QSpinBox()
        self.spin_every_steps.setRange(0, 50000)
        self.spin_every_steps.setSingleStep(100)
        self.spin_every_steps.setValue(0)
        self.spin_every_steps.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_st, 2, 2)
        g1.addWidget(self.spin_every_steps, 2, 3)

        # Sampler Steps & CFG
        lbl_inf_steps = QLabel("Sampling Steps:")
        lbl_inf_steps.setProperty("class", "field-label")
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 100)
        self.spin_steps.setValue(28)
        self.spin_steps.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_inf_steps, 3, 0)
        g1.addWidget(self.spin_steps, 3, 1)

        lbl_cfg = QLabel("CFG Scale:")
        lbl_cfg.setProperty("class", "field-label")
        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(1.0, 30.0)
        self.spin_cfg.setSingleStep(0.5)
        self.spin_cfg.setValue(7.0)
        self.spin_cfg.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_cfg, 3, 2)
        g1.addWidget(self.spin_cfg, 3, 3)

        # Seed & Resolution
        lbl_seed = QLabel("Fixed Seed:")
        lbl_seed.setProperty("class", "field-label")
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(42)
        self.spin_seed.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_seed, 4, 0)
        g1.addWidget(self.spin_seed, 4, 1)

        # Sampler selection
        lbl_sampler = QLabel("Inference Sampler:")
        lbl_sampler.setProperty("class", "field-label")
        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems([
            "euler", "euler_a", "dpm++ 2m karras", "dpm++ 2m sde karras",
            "ddim", "lms", "heun", "uni_pc",
        ])
        self.combo_sampler.setToolTip("Sampler/scheduler used for generating preview images during training.")
        self.combo_sampler.currentTextChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_sampler, 4, 2)
        g1.addWidget(self.combo_sampler, 4, 3)

        lbl_sres = QLabel("Sample Resolution:")
        lbl_sres.setProperty("class", "field-label")
        self.spin_res = QSpinBox()
        self.spin_res.setRange(256, 2048)
        self.spin_res.setSingleStep(64)
        self.spin_res.setValue(512)
        self.spin_res.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_sres, 5, 0)
        g1.addWidget(self.spin_res, 5, 1)

        layout.addWidget(grp_rules)

        # ── Group 2: Prompts Editor ──
        grp_prompts = QFrame()
        grp_prompts.setProperty("class", "group-box")
        g2 = QVBoxLayout(grp_prompts)
        g2.setSpacing(8)
        g2.setContentsMargins(14, 12, 14, 12)

        p_title = QLabel("SAMPLE PROMPTS (One prompt per line)")
        p_title.setProperty("class", "section-title")
        g2.addWidget(p_title)

        self.txt_prompts = QPlainTextEdit()
        self.txt_prompts.setPlaceholderText("Enter one prompt per line. Example:\nportrait of <trigger>, cinematic lighting, detailed\nfull body photo of <trigger>, standing in nature")
        self.txt_prompts.textChanged.connect(lambda: self.config_changed.emit())
        self.txt_prompts.setMaximumHeight(120)
        g2.addWidget(self.txt_prompts)

        lbl_neg = QLabel("Negative Prompt:")
        lbl_neg.setProperty("class", "field-label")
        g2.addWidget(lbl_neg)

        self.edit_negative = QLineEdit("ugly, blurry, low quality, distorted, extra limbs")
        self.edit_negative.textChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.edit_negative)

        layout.addWidget(grp_prompts)
        layout.addStretch()

    def load_config(self, config: TrainingConfig):
        self.chk_enable.setChecked(config.enable_samples)
        self.spin_every_epochs.setValue(config.sample_every_n_epochs)
        self.spin_every_steps.setValue(config.sample_every_n_steps)
        self.spin_steps.setValue(config.sample_steps)
        self.spin_cfg.setValue(config.sample_cfg)
        self.spin_seed.setValue(config.sample_seed)
        self.spin_res.setValue(config.sample_resolution)
        self.combo_sampler.setCurrentText(config.sample_sampler)
        self.txt_prompts.setPlainText("\n".join(config.sample_prompts))
        self.edit_negative.setText(config.sample_negative_prompt)

    def save_config(self, config: TrainingConfig):
        config.enable_samples = self.chk_enable.isChecked()
        config.sample_every_n_epochs = self.spin_every_epochs.value()
        config.sample_every_n_steps = self.spin_every_steps.value()
        config.sample_steps = self.spin_steps.value()
        config.sample_cfg = self.spin_cfg.value()
        config.sample_seed = self.spin_seed.value()
        config.sample_resolution = self.spin_res.value()
        config.sample_sampler = self.combo_sampler.currentText()
        lines = [line.strip() for line in self.txt_prompts.toPlainText().splitlines() if line.strip()]
        config.sample_prompts = lines if lines else ["portrait, cinematic lighting"]
        config.sample_negative_prompt = self.edit_negative.text().strip()
