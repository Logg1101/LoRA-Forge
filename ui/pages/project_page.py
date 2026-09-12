"""
Project & Model Setup Page.
Controls: Project Name, Base Model Path, Architecture, Dataset Directory, Output Directory, and live metadata summary.
"""
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QFrame, QGridLayout
)

from core.config_schema import TrainingConfig
from ui.widgets.path_picker import PathPicker


class ProjectPage(QWidget):
    """Page 1: Project configuration, base model selection and path routing."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Section Header
        title = QLabel("PROJECT & BASE MODEL CONFIGURATION")
        title.setProperty("class", "section-title")
        desc = QLabel("Set your LoRA project identity, select the base diffusion checkpoint, and define filesystem paths.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # Main Setup Form Box
        form_box = QFrame()
        form_box.setProperty("class", "group-box")
        form_layout = QGridLayout(form_box)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(14, 12, 14, 12)

        # Row 0: Project Name
        lbl_proj = QLabel("Project Name:")
        lbl_proj.setProperty("class", "field-label")
        self.edit_project_name = QLineEdit()
        self.edit_project_name.setPlaceholderText("e.g. Belfast_v1_LoRA")
        self.edit_project_name.textChanged.connect(lambda: self.config_changed.emit())
        form_layout.addWidget(lbl_proj, 0, 0)
        form_layout.addWidget(self.edit_project_name, 0, 1, 1, 2)

        # Row 1: Model Architecture
        lbl_arch = QLabel("Architecture:")
        lbl_arch.setProperty("class", "field-label")
        self.combo_arch = QComboBox()
        self.combo_arch.addItems(["SDXL", "Flux Dev", "Flux Schnell", "Flux.1", "SD1.5"])
        self.combo_arch.currentTextChanged.connect(self._on_arch_changed)
        form_layout.addWidget(lbl_arch, 1, 0)
        form_layout.addWidget(self.combo_arch, 1, 1, 1, 2)

        # Row 2: Base Model Path
        lbl_model = QLabel("Base Model File:")
        lbl_model.setProperty("class", "field-label")
        self.picker_model = PathPicker(
            mode="file",
            file_filter="Model Checkpoints (*.safetensors *.bin *.pt);;All Files (*.*)",
            placeholder="Select a .safetensors checkpoint or diffusers directory..."
        )
        self.picker_model.path_changed.connect(self._on_model_path_changed)
        form_layout.addWidget(lbl_model, 2, 0)
        form_layout.addWidget(self.picker_model, 2, 1, 1, 2)

        # Row 3: Dataset Directory
        lbl_dataset = QLabel("Dataset Directory:")
        lbl_dataset.setProperty("class", "field-label")
        self.picker_dataset = PathPicker(mode="dir", placeholder="Folder containing training images and .txt captions...")
        self.picker_dataset.path_changed.connect(self._on_dataset_dir_changed)
        form_layout.addWidget(lbl_dataset, 3, 0)
        form_layout.addWidget(self.picker_dataset, 3, 1, 1, 2)

        # Row 4: Output Directory
        lbl_output = QLabel("Output Directory:")
        lbl_output.setProperty("class", "field-label")
        self.picker_output = PathPicker(mode="dir", placeholder="Folder where finished weights and backups are saved...")
        self.picker_output.path_changed.connect(lambda: self.config_changed.emit())
        form_layout.addWidget(lbl_output, 4, 0)
        form_layout.addWidget(self.picker_output, 4, 1, 1, 2)

        layout.addWidget(form_box)

        # Metadata Inspection Box
        meta_box = QFrame()
        meta_box.setProperty("class", "group-box")
        meta_layout = QGridLayout(meta_box)
        meta_layout.setSpacing(8)
        meta_layout.setContentsMargins(14, 12, 14, 12)

        meta_title = QLabel("MODEL & ASSET METADATA")
        meta_title.setProperty("class", "section-title")
        meta_layout.addWidget(meta_title, 0, 0, 1, 2)

        self.lbl_meta_model = QLabel("Model File: None selected")
        self.lbl_meta_model.setStyleSheet("color: #8b949e; font-family: monospace;")
        self.lbl_meta_size = QLabel("File Size: --")
        self.lbl_meta_size.setStyleSheet("color: #8b949e; font-family: monospace;")
        self.lbl_meta_dataset = QLabel("Dataset Status: Not loaded")
        self.lbl_meta_dataset.setStyleSheet("color: #8b949e; font-family: monospace;")

        meta_layout.addWidget(self.lbl_meta_model, 1, 0)
        meta_layout.addWidget(self.lbl_meta_size, 1, 1)
        meta_layout.addWidget(self.lbl_meta_dataset, 2, 0, 1, 2)

        layout.addWidget(meta_box)
        layout.addStretch()

    def _on_arch_changed(self, text: str):
        self.config_changed.emit()

    def _on_model_path_changed(self, path_str: str):
        p = Path(path_str)
        if p.exists() and p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            size_gb = size_mb / 1024
            self.lbl_meta_model.setText(f"Model: {p.name}")
            self.lbl_meta_size.setText(f"Size: {size_gb:.2f} GB ({size_mb:.0f} MB)")
            self.lbl_meta_model.setStyleSheet("color: #79c0ff; font-family: monospace;")
            self.lbl_meta_size.setStyleSheet("color: #79c0ff; font-family: monospace;")
        else:
            self.lbl_meta_model.setText("Model File: Path invalid or missing")
            self.lbl_meta_size.setText("File Size: --")
            self.lbl_meta_model.setStyleSheet("color: #f85149; font-family: monospace;")
        self.config_changed.emit()

    def _on_dataset_dir_changed(self, path_str: str):
        p = Path(path_str)
        if p.exists() and p.is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            images = [f for f in p.rglob("*") if f.suffix.lower() in valid_exts]
            captions = [f for f in p.rglob("*") if f.suffix.lower() in {".txt", ".caption"}]
            self.lbl_meta_dataset.setText(f"Dataset: {len(images)} images detected, {len(captions)} caption files")
            self.lbl_meta_dataset.setStyleSheet("color: #3fb950; font-family: monospace;")
        else:
            self.lbl_meta_dataset.setText("Dataset Status: Directory not found")
            self.lbl_meta_dataset.setStyleSheet("color: #8b949e; font-family: monospace;")
        self.config_changed.emit()

    def load_config(self, config: TrainingConfig):
        self.edit_project_name.setText(config.project_name)
        self.combo_arch.setCurrentText(config.model_type)
        self.picker_model.setText(config.base_model_path)
        self.picker_dataset.setText(config.dataset_dir)
        self.picker_output.setText(config.output_dir)

    def save_config(self, config: TrainingConfig):
        config.project_name = self.edit_project_name.text().strip() or "my_lora"
        config.model_type = self.combo_arch.currentText()
        config.base_model_path = self.picker_model.text()
        config.dataset_dir = self.picker_dataset.text()
        config.output_dir = self.picker_output.text()
