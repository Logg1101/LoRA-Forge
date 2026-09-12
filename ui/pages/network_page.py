"""
Network & Architecture Tuning Page.
Controls: Adapter type (LoRA, DoRA, LoHA, LoCon), Linear Rank (dim), Alpha,
Dropout parameters, and Convolutional Rank/Alpha controls.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFrame, QGridLayout
)

from core.config_schema import TrainingConfig


class NetworkPage(QWidget):
    """Page 3: Adapter algorithms, ranks, alpha scaling, and regularizations."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("NETWORK ARCHITECTURE & ADAPTERS")
        title.setProperty("class", "section-title")
        desc = QLabel("Configure parameter-efficient fine-tuning matrix dimensions, scaling factors, and regularizations.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Core Network Settings ──
        grp_core = QFrame()
        grp_core.setProperty("class", "group-box")
        g1 = QGridLayout(grp_core)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("ADAPTER MATRIX SPECS")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Network Type
        lbl_type = QLabel("Network Type:")
        lbl_type.setProperty("class", "field-label")
        self.combo_type = QComboBox()
        self.combo_type.addItems(["lora", "dora", "loha", "locon"])
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        g1.addWidget(lbl_type, 1, 0)
        g1.addWidget(self.combo_type, 1, 1)

        # Rank
        lbl_rank = QLabel("Linear Rank (Dim):")
        lbl_rank.setProperty("class", "field-label")
        self.spin_rank = QSpinBox()
        self.spin_rank.setRange(1, 512)
        self.spin_rank.setValue(16)
        self.spin_rank.valueChanged.connect(self._on_rank_changed)
        g1.addWidget(lbl_rank, 1, 2)
        g1.addWidget(self.spin_rank, 1, 3)

        # Alpha
        lbl_alpha = QLabel("Alpha (Scale):")
        lbl_alpha.setProperty("class", "field-label")
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.01, 512.0)
        self.spin_alpha.setValue(8.0)
        self.spin_alpha.setToolTip("Scaling factor. Standard practice is alpha = rank / 2 or alpha = rank.")
        self.spin_alpha.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_alpha, 2, 0)
        g1.addWidget(self.spin_alpha, 2, 1)

        layout.addWidget(grp_core)

        # ── Group 2: Conv Layers (Conditional) ──
        self.grp_conv = QFrame()
        self.grp_conv.setProperty("class", "group-box")
        g2 = QGridLayout(self.grp_conv)
        g2.setSpacing(10)
        g2.setContentsMargins(14, 12, 14, 12)

        g2_title = QLabel("CONVOLUTIONAL LAYER ADAPTERS (LOCON / LOHA)")
        g2_title.setProperty("class", "section-title")
        g2.addWidget(g2_title, 0, 0, 1, 4)

        lbl_c_rank = QLabel("Conv Rank:")
        lbl_c_rank.setProperty("class", "field-label")
        self.spin_conv_rank = QSpinBox()
        self.spin_conv_rank.setRange(0, 512)
        self.spin_conv_rank.setValue(0)
        self.spin_conv_rank.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_c_rank, 1, 0)
        g2.addWidget(self.spin_conv_rank, 1, 1)

        lbl_c_alpha = QLabel("Conv Alpha:")
        lbl_c_alpha.setProperty("class", "field-label")
        self.spin_conv_alpha = QDoubleSpinBox()
        self.spin_conv_alpha.setRange(0.0, 512.0)
        self.spin_conv_alpha.setValue(0.0)
        self.spin_conv_alpha.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_c_alpha, 1, 2)
        g2.addWidget(self.spin_conv_alpha, 1, 3)

        layout.addWidget(self.grp_conv)

        # ── Group 3: Regularization / Dropout ──
        grp_drop = QFrame()
        grp_drop.setProperty("class", "group-box")
        g3 = QGridLayout(grp_drop)
        g3.setSpacing(10)
        g3.setContentsMargins(14, 12, 14, 12)

        g3_title = QLabel("REGULARIZATION & DROPOUT")
        g3_title.setProperty("class", "section-title")
        g3.addWidget(g3_title, 0, 0, 1, 4)

        # Network Dropout
        lbl_nd = QLabel("Network Dropout:")
        lbl_nd.setProperty("class", "field-label")
        self.spin_net_drop = QDoubleSpinBox()
        self.spin_net_drop.setRange(0.0, 1.0)
        self.spin_net_drop.setSingleStep(0.05)
        self.spin_net_drop.setValue(0.0)
        self.spin_net_drop.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_nd, 1, 0)
        g3.addWidget(self.spin_net_drop, 1, 1)

        # Rank Dropout
        lbl_rd = QLabel("Rank Dropout:")
        lbl_rd.setProperty("class", "field-label")
        self.spin_rank_drop = QDoubleSpinBox()
        self.spin_rank_drop.setRange(0.0, 1.0)
        self.spin_rank_drop.setSingleStep(0.05)
        self.spin_rank_drop.setValue(0.0)
        self.spin_rank_drop.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_rd, 1, 2)
        g3.addWidget(self.spin_rank_drop, 1, 3)

        # Module Dropout
        lbl_md = QLabel("Module Dropout:")
        lbl_md.setProperty("class", "field-label")
        self.spin_mod_drop = QDoubleSpinBox()
        self.spin_mod_drop.setRange(0.0, 1.0)
        self.spin_mod_drop.setSingleStep(0.05)
        self.spin_mod_drop.setValue(0.0)
        self.spin_mod_drop.valueChanged.connect(lambda: self.config_changed.emit())
        g3.addWidget(lbl_md, 2, 0)
        g3.addWidget(self.spin_mod_drop, 2, 1)

        layout.addWidget(grp_drop)

        # ── Group 4: UNet Block LR Multipliers ──
        grp_blocks = QFrame()
        grp_blocks.setProperty("class", "group-box")
        g4 = QGridLayout(grp_blocks)
        g4.setSpacing(10)
        g4.setContentsMargins(14, 12, 14, 12)

        g4_title = QLabel("UNET BLOCK LR MULTIPLIERS")
        g4_title.setProperty("class", "section-title")
        g4.addWidget(g4_title, 0, 0, 1, 4)

        lbl_down = QLabel("Down Blocks (Input):")
        lbl_down.setProperty("class", "field-label")
        self.spin_down_weight = QDoubleSpinBox()
        self.spin_down_weight.setRange(0.0, 5.0)
        self.spin_down_weight.setSingleStep(0.1)
        self.spin_down_weight.setValue(1.0)
        self.spin_down_weight.setToolTip("LR multiplier for UNet input blocks. Set to 0 to freeze input layers.")
        self.spin_down_weight.valueChanged.connect(lambda: self.config_changed.emit())
        g4.addWidget(lbl_down, 1, 0)
        g4.addWidget(self.spin_down_weight, 1, 1)

        lbl_mid = QLabel("Mid Block:")
        lbl_mid.setProperty("class", "field-label")
        self.spin_mid_weight = QDoubleSpinBox()
        self.spin_mid_weight.setRange(0.0, 5.0)
        self.spin_mid_weight.setSingleStep(0.1)
        self.spin_mid_weight.setValue(1.0)
        self.spin_mid_weight.setToolTip("LR multiplier for UNet middle block. Set to 0 to freeze middle layer.")
        self.spin_mid_weight.valueChanged.connect(lambda: self.config_changed.emit())
        g4.addWidget(lbl_mid, 1, 2)
        g4.addWidget(self.spin_mid_weight, 1, 3)

        lbl_up = QLabel("Up Blocks (Output):")
        lbl_up.setProperty("class", "field-label")
        self.spin_up_weight = QDoubleSpinBox()
        self.spin_up_weight.setRange(0.0, 5.0)
        self.spin_up_weight.setSingleStep(0.1)
        self.spin_up_weight.setValue(1.0)
        self.spin_up_weight.setToolTip("LR multiplier for UNet output blocks. Set to 0 to freeze output layers.")
        self.spin_up_weight.valueChanged.connect(lambda: self.config_changed.emit())
        g4.addWidget(lbl_up, 2, 0)
        g4.addWidget(self.spin_up_weight, 2, 1)

        layout.addWidget(grp_blocks)
        layout.addStretch()

        # Update initial visibility
        self._on_type_changed(self.combo_type.currentText())

    def _on_type_changed(self, net_type: str):
        is_conv_supported = net_type in ("locon", "loha")
        self.grp_conv.setVisible(is_conv_supported)
        self.config_changed.emit()

    def _on_rank_changed(self, rank: int):
        # Only auto-set alpha if it's at the initial zero value; respect user overrides
        if self.spin_alpha.value() == 0:
            self.spin_alpha.setValue(max(1.0, rank / 2.0))
        self.config_changed.emit()

    def load_config(self, config: TrainingConfig):
        self.combo_type.setCurrentText(config.network_type)
        self.spin_rank.setValue(config.rank)
        self.spin_alpha.setValue(config.alpha)
        self.spin_conv_rank.setValue(config.conv_rank)
        self.spin_conv_alpha.setValue(config.conv_alpha)
        self.spin_net_drop.setValue(config.network_dropout)
        self.spin_rank_drop.setValue(config.rank_dropout)
        self.spin_mod_drop.setValue(config.module_dropout)
        self.spin_down_weight.setValue(config.down_lr_weight)
        self.spin_mid_weight.setValue(config.mid_lr_weight)
        self.spin_up_weight.setValue(config.up_lr_weight)
        self._on_type_changed(config.network_type)

    def save_config(self, config: TrainingConfig):
        config.network_type = self.combo_type.currentText()
        config.rank = self.spin_rank.value()
        config.alpha = self.spin_alpha.value()
        config.conv_rank = self.spin_conv_rank.value()
        config.conv_alpha = self.spin_conv_alpha.value()
        config.network_dropout = self.spin_net_drop.value()
        config.rank_dropout = self.spin_rank_drop.value()
        config.module_dropout = self.spin_mod_drop.value()
        config.down_lr_weight = self.spin_down_weight.value()
        config.mid_lr_weight = self.spin_mid_weight.value()
        config.up_lr_weight = self.spin_up_weight.value()
