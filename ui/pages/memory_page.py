"""
Memory & Hardware Performance Optimization Page.
Controls: Resolution, Mixed Precision, Gradient Checkpointing, Latent Caching,
Base Model Quantization (NF4 / INT8), DataLoader workers, Pin memory, and live GPU monitor.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QComboBox,
    QCheckBox, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig
from ui.widgets.gpu_monitor_widget import GPUMonitorWidget


class MemoryPage(QWidget):
    """Page 5: VRAM management, caching strategies, model quantization, and device optimization."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("MEMORY & HARDWARE PERFORMANCE")
        title.setProperty("class", "section-title")
        desc = QLabel("Configure VRAM conservation techniques: base model quantization (NF4), latent RAM pre-caching, and gradient checkpointing.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # Live GPU Card
        self.gpu_widget = GPUMonitorWidget()
        layout.addWidget(self.gpu_widget)

        # ── Group 1: VRAM Reduction Options ──
        grp_vram = QFrame()
        grp_vram.setProperty("class", "group-box")
        g1 = QGridLayout(grp_vram)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("VRAM OPTIMIZATION & PRECISION")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Resolution
        lbl_res = QLabel("Max Training Resolution:")
        lbl_res.setProperty("class", "field-label")
        self.spin_res = QSpinBox()
        self.spin_res.setRange(128, 2048)
        self.spin_res.setSingleStep(64)
        self.spin_res.setValue(1024)
        self.spin_res.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_res, 1, 0)
        g1.addWidget(self.spin_res, 1, 1)

        # Precision
        lbl_prec = QLabel("Mixed Precision:")
        lbl_prec.setProperty("class", "field-label")
        self.combo_prec = QComboBox()
        self.combo_prec.addItems(["bf16", "fp16", "fp8", "no"])
        self.combo_prec.currentTextChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_prec, 1, 2)
        g1.addWidget(self.combo_prec, 1, 3)

        # Quantization
        lbl_quant = QLabel("Base Model Quantization:")
        lbl_quant.setProperty("class", "field-label")
        self.combo_quant = QComboBox()
        self.combo_quant.addItems(["none", "nf4", "int8"])
        self.combo_quant.setToolTip("NF4 cuts transformer VRAM from ~12GB to ~3GB (critical for 12GB GPUs like RTX 5070).")
        self.combo_quant.currentTextChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_quant, 2, 0)
        g1.addWidget(self.combo_quant, 2, 1)

        # Gradient Checkpointing
        self.chk_grad_ckpt = QCheckBox("Gradient Checkpointing")
        self.chk_grad_ckpt.setChecked(True)
        self.chk_grad_ckpt.setToolTip("Reduces activation memory by ~60% at a ~30% compute time cost.")
        self.chk_grad_ckpt.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_grad_ckpt, 2, 2, 1, 2)

        # Cache Latents
        self.chk_cache = QCheckBox("Cache Latents to RAM")
        self.chk_cache.setChecked(True)
        self.chk_cache.setToolTip("Encodes all images and captions once prior to training, offloading VAE/TE from GPU.")
        self.chk_cache.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_cache, 3, 0, 1, 2)

        layout.addWidget(grp_vram)

        # ── Group 2: DataLoader & Host Pipeline ──
        grp_dl = QFrame()
        grp_dl.setProperty("class", "group-box")
        g2 = QGridLayout(grp_dl)
        g2.setSpacing(10)
        g2.setContentsMargins(14, 12, 14, 12)

        g2_title = QLabel("DATALOADER & PIPELINE WORKERS")
        g2_title.setProperty("class", "section-title")
        g2.addWidget(g2_title, 0, 0, 1, 4)

        # Workers
        lbl_workers = QLabel("DataLoader Workers:")
        lbl_workers.setProperty("class", "field-label")
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(0)
        self.spin_workers.setToolTip("Keep 0 when Latent Caching is active. Set 2-4 for disk-based on-the-fly loading.")
        self.spin_workers.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_workers, 1, 0)
        g2.addWidget(self.spin_workers, 1, 1)

        # Pin Memory
        self.chk_pin = QCheckBox("Pin Memory")
        self.chk_pin.setChecked(True)
        self.chk_pin.setToolTip("Pins host RAM for faster GPU tensor transfers.")
        self.chk_pin.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_pin, 1, 2, 1, 2)

        layout.addWidget(grp_dl)
        layout.addStretch()

    def load_config(self, config: TrainingConfig):
        self.spin_res.setValue(config.resolution)
        self.combo_prec.setCurrentText(config.mixed_precision)
        self.combo_quant.setCurrentText(config.quantization)
        self.chk_grad_ckpt.setChecked(config.gradient_checkpointing)
        self.chk_cache.setChecked(config.cache_latents)
        self.spin_workers.setValue(config.dataloader_workers)
        self.chk_pin.setChecked(config.pin_memory)

    def save_config(self, config: TrainingConfig):
        config.resolution = self.spin_res.value()
        config.mixed_precision = self.combo_prec.currentText()
        config.quantization = self.combo_quant.currentText()
        config.gradient_checkpointing = self.chk_grad_ckpt.isChecked()
        config.cache_latents = self.chk_cache.isChecked()
        config.dataloader_workers = self.spin_workers.value()
        config.pin_memory = self.chk_pin.isChecked()
