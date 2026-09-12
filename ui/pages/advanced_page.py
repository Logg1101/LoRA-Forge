"""
Advanced Configuration, Custom VAE, Reproducibility, Presets & Live System Console Page.
Controls: Seed, Shuffle Seed, Deterministic mode, Custom VAE picker, Preset Save/Load/Reset,
and embedded system log console.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox,
    QComboBox, QPushButton, QFrame, QGridLayout, QInputDialog, QMessageBox
)

from core.config_schema import TrainingConfig
from core.presets import get_preset_names, get_preset
from core.config_handler import save_preset, load_preset, list_user_presets
from ui.widgets.path_picker import PathPicker
from ui.widgets.log_console_widget import LogConsoleWidget


class AdvancedPage(QWidget):
    """Page 9: Seed reproducibility, custom VAE override, configuration preset manager, and console."""

    config_changed = Signal()
    preset_loaded = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("ADVANCED ARCHITECTURE, PRESETS & SYSTEM LOGS")
        title.setProperty("class", "section-title")
        desc = QLabel("Manage reproducibility seeds, external VAE overrides, configuration presets, and inspect real-time system logs.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Presets Manager ──
        grp_preset = QFrame()
        grp_preset.setProperty("class", "group-box")
        g1 = QHBoxLayout(grp_preset)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 10, 14, 10)

        p_lbl = QLabel("Training Preset:")
        p_lbl.setProperty("class", "field-label")
        g1.addWidget(p_lbl)

        self.combo_presets = QComboBox()
        self._refresh_presets_combo()
        g1.addWidget(self.combo_presets, stretch=1)

        self.btn_load_preset = QPushButton("Load Preset")
        self.btn_load_preset.setProperty("class", "browse-btn")
        self.btn_load_preset.clicked.connect(self._on_load_preset)
        g1.addWidget(self.btn_load_preset)

        self.btn_save_preset = QPushButton("Save Preset...")
        self.btn_save_preset.setProperty("class", "browse-btn")
        self.btn_save_preset.clicked.connect(self._on_save_preset)
        g1.addWidget(self.btn_save_preset)

        layout.addWidget(grp_preset)

        # ── Group 2: Seed & Reproducibility ──
        grp_seed = QFrame()
        grp_seed.setProperty("class", "group-box")
        g2 = QGridLayout(grp_seed)
        g2.setSpacing(10)
        g2.setContentsMargins(14, 12, 14, 12)

        g2_title = QLabel("REPRODUCIBILITY & RANDOM SEEDS")
        g2_title.setProperty("class", "section-title")
        g2.addWidget(g2_title, 0, 0, 1, 4)

        # Training seed
        lbl_s1 = QLabel("Training Seed (-1 = Random):")
        lbl_s1.setProperty("class", "field-label")
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        self.spin_seed.setValue(-1)
        self.spin_seed.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_s1, 1, 0)
        g2.addWidget(self.spin_seed, 1, 1)

        # Shuffle seed
        lbl_s2 = QLabel("Dataset Shuffle Seed:")
        lbl_s2.setProperty("class", "field-label")
        self.spin_shuffle_seed = QSpinBox()
        self.spin_shuffle_seed.setRange(-1, 2147483647)
        self.spin_shuffle_seed.setValue(-1)
        self.spin_shuffle_seed.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_s2, 1, 2)
        g2.addWidget(self.spin_shuffle_seed, 1, 3)

        # Deterministic
        self.chk_deterministic = QCheckBox("Deterministic PyTorch Kernels")
        self.chk_deterministic.setToolTip("Forces deterministic CUDA algorithms for 100% exact repeatability (may slightly reduce training speed).")
        self.chk_deterministic.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_deterministic, 2, 0, 1, 2)

        layout.addWidget(grp_seed)

        # ── Group 3: Custom VAE Override ──
        grp_vae = QFrame()
        grp_vae.setProperty("class", "group-box")
        g3 = QGridLayout(grp_vae)
        g3.setSpacing(10)
        g3.setContentsMargins(14, 12, 14, 12)

        g3_title = QLabel("CUSTOM VAE OVERRIDE")
        g3_title.setProperty("class", "section-title")
        g3.addWidget(g3_title, 0, 0, 1, 4)

        self.chk_custom_vae = QCheckBox("Use Custom VAE (Ignore Base Model VAE)")
        self.chk_custom_vae.stateChanged.connect(self._on_custom_vae_toggled)
        g3.addWidget(self.chk_custom_vae, 1, 0, 1, 4)

        self.lbl_vae_path = QLabel("VAE Checkpoint:")
        self.lbl_vae_path.setProperty("class", "field-label")
        self.picker_vae = PathPicker(mode="file", file_filter="VAE Models (*.safetensors *.pt);;All Files (*.*)")
        self.picker_vae.path_changed.connect(lambda: self.config_changed.emit())
        g3.addWidget(self.lbl_vae_path, 2, 0)
        g3.addWidget(self.picker_vae, 2, 1, 1, 3)

        layout.addWidget(grp_vae)

        # ── Group 4: Live Console ──
        self.console = LogConsoleWidget()
        layout.addWidget(self.console, stretch=1)

        self._on_custom_vae_toggled(False)

    def _refresh_presets_combo(self):
        self.combo_presets.clear()
        # Built-in presets
        for name in get_preset_names():
            self.combo_presets.addItem(f"📦 {name}", userData=("builtin", name))
        # User presets
        for name in list_user_presets():
            self.combo_presets.addItem(f"👤 {name}", userData=("user", name))

    def _on_load_preset(self):
        idx = self.combo_presets.currentIndex()
        if idx < 0:
            return
        p_type, name = self.combo_presets.currentData()
        if p_type == "builtin":
            p_dict = get_preset(name)
            self.preset_loaded.emit(p_dict)
            self.console.log(f"Loaded built-in preset '{name}'", "SUCCESS")
        else:
            p_cfg = load_preset(name)
            if p_cfg:
                self.preset_loaded.emit(p_cfg.model_dump())
                self.console.log(f"Loaded user preset '{name}'", "SUCCESS")

    def _on_save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter a name for this configuration preset:")
        if ok and name.strip():
            # Parent UI will handle config serialization and save
            self.save_preset_requested = name.strip()
            self.config_changed.emit()
            self._refresh_presets_combo()

    def _on_custom_vae_toggled(self, checked):
        self.lbl_vae_path.setEnabled(checked)
        self.picker_vae.setEnabled(checked)
        self.config_changed.emit()

    def log_message(self, msg: str, level: str = "INFO"):
        self.console.log(msg, level)

    def load_config(self, config: TrainingConfig):
        self.spin_seed.setValue(config.training_seed)
        self.spin_shuffle_seed.setValue(config.shuffle_seed)
        self.chk_deterministic.setChecked(config.deterministic)
        self.chk_custom_vae.setChecked(config.use_custom_vae)
        self.picker_vae.setText(config.custom_vae_path)
        self._on_custom_vae_toggled(config.use_custom_vae)

    def save_config(self, config: TrainingConfig):
        config.training_seed = self.spin_seed.value()
        config.shuffle_seed = self.spin_shuffle_seed.value()
        config.deterministic = self.chk_deterministic.isChecked()
        config.use_custom_vae = self.chk_custom_vae.isChecked()
        config.custom_vae_path = self.picker_vae.text()
