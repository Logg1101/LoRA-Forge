"""
Loss Functions, Diffusion Schedule & Real-Time Loss Graph Page.
Controls: Loss function (MSE / Huber), Huber C parameter, Min-SNR Gamma weighting,
Noise Offset, V-Prediction toggle, and live Loss Curve plotter.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig
from ui.widgets.loss_graph_widget import LossGraphWidget


class LossPage(QWidget):
    """Page 6: Diffusion loss formulation, SNR balancing, and real-time loss tracking."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("LOSS FORMULATION & NOISE DYNAMICS")
        title.setProperty("class", "section-title")
        desc = QLabel("Select loss metrics, enable Min-SNR gamma weighting, inject noise offset for dark/light contrast, and monitor loss convergence.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Loss Configuration ──
        grp_loss = QFrame()
        grp_loss.setProperty("class", "group-box")
        g1 = QGridLayout(grp_loss)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("DIFFUSION OBJECTIVE SETTINGS")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Loss Function
        lbl_fn = QLabel("Loss Function:")
        lbl_fn.setProperty("class", "field-label")
        self.combo_fn = QComboBox()
        self.combo_fn.addItems(["mse", "huber"])
        self.combo_fn.currentTextChanged.connect(self._on_fn_changed)
        g1.addWidget(lbl_fn, 1, 0)
        g1.addWidget(self.combo_fn, 1, 1)

        # Huber C
        self.lbl_huber = QLabel("Huber Beta/C Parameter:")
        self.lbl_huber.setProperty("class", "field-label")
        self.spin_huber = QDoubleSpinBox()
        self.spin_huber.setRange(0.001, 10.0)
        self.spin_huber.setSingleStep(0.01)
        self.spin_huber.setValue(0.1)
        self.spin_huber.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.lbl_huber, 1, 2)
        g1.addWidget(self.spin_huber, 1, 3)

        # Min-SNR Gamma
        lbl_snr = QLabel("Min-SNR Gamma (0 = Off):")
        lbl_snr.setProperty("class", "field-label")
        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(0.0, 20.0)
        self.spin_snr.setSingleStep(1.0)
        self.spin_snr.setValue(0.0)
        self.spin_snr.setToolTip("Clamps loss weighting by SNR to accelerate convergence (typically 5.0 for SDXL).")
        self.spin_snr.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_snr, 2, 0)
        g1.addWidget(self.spin_snr, 2, 1)

        # Noise Offset
        lbl_offset = QLabel("Noise Offset (0 = Off):")
        lbl_offset.setProperty("class", "field-label")
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(0.0, 0.5)
        self.spin_offset.setSingleStep(0.01)
        self.spin_offset.setValue(0.0)
        self.spin_offset.setToolTip("Adds mean offset to noise latents, allowing generation of very dark or very bright scenes (recommended: 0.03-0.05).")
        self.spin_offset.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_offset, 2, 2)
        g1.addWidget(self.spin_offset, 2, 3)

        # V-Prediction
        self.chk_vpred = QCheckBox("V-Prediction Target")
        self.chk_vpred.setToolTip("Enable only if training on v-prediction base models (e.g. SD2.1-v).")
        self.chk_vpred.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_vpred, 3, 0, 1, 2)

        layout.addWidget(grp_loss)

        # ── Group 2: Real-Time Loss Graph ──
        self.loss_graph = LossGraphWidget()
        layout.addWidget(self.loss_graph, stretch=1)

        self._on_fn_changed(self.combo_fn.currentText())

    def _on_fn_changed(self, fn: str):
        is_huber = fn == "huber"
        self.lbl_huber.setEnabled(is_huber)
        self.spin_huber.setEnabled(is_huber)
        self.config_changed.emit()

    def update_loss_point(self, step: int, loss: float):
        self.loss_graph.update_loss(step, loss)

    def load_config(self, config: TrainingConfig):
        self.combo_fn.setCurrentText(config.loss_function)
        self.spin_huber.setValue(config.huber_c)
        self.spin_snr.setValue(config.min_snr_gamma)
        self.spin_offset.setValue(config.noise_offset)
        self.chk_vpred.setChecked(config.v_prediction)
        self._on_fn_changed(config.loss_function)

    def save_config(self, config: TrainingConfig):
        config.loss_function = self.combo_fn.currentText()
        config.huber_c = self.spin_huber.value()
        config.min_snr_gamma = self.spin_snr.value()
        config.noise_offset = self.spin_offset.value()
        config.v_prediction = self.chk_vpred.isChecked()
