"""
Dataset Management, Aspect Ratio Bucketing & Augmentation Page.
Controls: Repeats, Caption extension, Tag shuffling, Keep tokens, Caption dropout,
Bucketing parameters (min/max/step), Augmentations (flip, color, crop), and Dataset Validation.
"""
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QPushButton, QFrame, QGridLayout, QTextEdit
)

from core.config_schema import TrainingConfig


class DatasetPage(QWidget):
    """Page 2: Dataset repetition, tag management, bucketing rules, and data augmentations."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("DATASET, BUCKETING & AUGMENTATION")
        title.setProperty("class", "section-title")
        desc = QLabel("Fine-tune dataset repeats, prompt tag tokenization, dynamic aspect ratio buckets, and non-destructive augmentations.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Dataset & Caption Handling ──
        grp_captions = QFrame()
        grp_captions.setProperty("class", "group-box")
        g1 = QGridLayout(grp_captions)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("DATASET REPEATS & CAPTION RULES")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Repeats
        lbl_rep = QLabel("Dataset Repeats:")
        lbl_rep.setProperty("class", "field-label")
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 100)
        self.spin_repeats.setValue(1)
        self.spin_repeats.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_rep, 1, 0)
        g1.addWidget(self.spin_repeats, 1, 1)

        # Caption Extension
        lbl_ext = QLabel("Caption Extension:")
        lbl_ext.setProperty("class", "field-label")
        self.edit_ext = QLineEdit(".txt")
        self.edit_ext.textChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_ext, 1, 2)
        g1.addWidget(self.edit_ext, 1, 3)

        # Shuffle Captions
        self.chk_shuffle = QCheckBox("Shuffle Comma-Separated Tags")
        self.chk_shuffle.setToolTip("Randomizes tag order in caption files each epoch for better concept generalization.")
        self.chk_shuffle.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_shuffle, 2, 0, 1, 2)

        # Keep Tokens
        lbl_keep = QLabel("Keep Tokens (protect prefix):")
        lbl_keep.setProperty("class", "field-label")
        self.spin_keep = QSpinBox()
        self.spin_keep.setRange(0, 100)
        self.spin_keep.setValue(0)
        self.spin_keep.setToolTip("Number of initial comma-separated tags protected from shuffling (e.g. your trigger word).")
        self.spin_keep.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_keep, 2, 2)
        g1.addWidget(self.spin_keep, 2, 3)

        # Caption Dropout
        lbl_drop = QLabel("Caption Dropout Rate:")
        lbl_drop.setProperty("class", "field-label")
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 1.0)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(0.0)
        self.spin_dropout.setToolTip("Probability of replacing caption with empty string to train unconditional generation.")
        self.spin_dropout.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_drop, 3, 0)
        g1.addWidget(self.spin_dropout, 3, 1)

        # Tag Dropout
        lbl_tag_drop = QLabel("Tag Dropout Rate:")
        lbl_tag_drop.setProperty("class", "field-label")
        self.spin_tag_dropout = QDoubleSpinBox()
        self.spin_tag_dropout.setRange(0.0, 1.0)
        self.spin_tag_dropout.setSingleStep(0.05)
        self.spin_tag_dropout.setValue(0.0)
        self.spin_tag_dropout.setToolTip("Probability of dropping individual comma-separated tags (prefix keep-tokens are protected).")
        self.spin_tag_dropout.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_tag_drop, 3, 2)
        g1.addWidget(self.spin_tag_dropout, 3, 3)

        layout.addWidget(grp_captions)

        # ── Group 2: Aspect Ratio Bucketing & Augmentation ──
        grp_buckets = QFrame()
        grp_buckets.setProperty("class", "group-box")
        g2 = QGridLayout(grp_buckets)
        g2.setSpacing(10)
        g2.setContentsMargins(14, 12, 14, 12)

        g2_title = QLabel("ASPECT RATIO BUCKETING & AUGMENTATION")
        g2_title.setProperty("class", "section-title")
        g2.addWidget(g2_title, 0, 0, 1, 4)

        # Min / Max Bucket
        lbl_min_b = QLabel("Min Bucket Res:")
        lbl_min_b.setProperty("class", "field-label")
        self.spin_min_b = QSpinBox()
        self.spin_min_b.setRange(128, 2048)
        self.spin_min_b.setSingleStep(64)
        self.spin_min_b.setValue(256)
        self.spin_min_b.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_min_b, 1, 0)
        g2.addWidget(self.spin_min_b, 1, 1)

        lbl_max_b = QLabel("Max Bucket Res:")
        lbl_max_b.setProperty("class", "field-label")
        self.spin_max_b = QSpinBox()
        self.spin_max_b.setRange(256, 4096)
        self.spin_max_b.setSingleStep(64)
        self.spin_max_b.setValue(2048)
        self.spin_max_b.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_max_b, 1, 2)
        g2.addWidget(self.spin_max_b, 1, 3)

        # Bucket Step
        lbl_step = QLabel("Bucket Step Size:")
        lbl_step.setProperty("class", "field-label")
        self.spin_step = QSpinBox()
        self.spin_step.setRange(8, 128)
        self.spin_step.setSingleStep(8)
        self.spin_step.setValue(64)
        self.spin_step.valueChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(lbl_step, 2, 0)
        g2.addWidget(self.spin_step, 2, 1)

        # Augmentations Row
        self.chk_flip = QCheckBox("Horizontal Flip")
        self.chk_flip.setToolTip("Randomly mirrors images horizontally (useful for styles/objects, avoid for asymmetric characters/text).")
        self.chk_flip.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_flip, 3, 0)

        self.chk_color = QCheckBox("Color Jitter Augmentation")
        self.chk_color.setToolTip("Subtle random hue and brightness perturbation.")
        self.chk_color.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_color, 3, 1)

        self.chk_crop = QCheckBox("Random Crop")
        self.chk_crop.setToolTip("Crops randomly instead of center crop when fitting bucket bounds.")
        self.chk_crop.stateChanged.connect(lambda: self.config_changed.emit())
        g2.addWidget(self.chk_crop, 3, 2, 1, 2)

        layout.addWidget(grp_buckets)

        # ── Group 3: Dataset Validator ──
        grp_val = QFrame()
        grp_val.setProperty("class", "group-box")
        g3 = QVBoxLayout(grp_val)
        g3.setSpacing(8)
        g3.setContentsMargins(14, 12, 14, 12)

        v_head = QHBoxLayout()
        v_title = QLabel("DATASET INTEGRITY AUDIT")
        v_title.setProperty("class", "section-title")
        v_head.addWidget(v_title)
        v_head.addStretch()

        self.btn_validate = QPushButton("Run Dataset Audit")
        self.btn_validate.setProperty("class", "browse-btn")
        self.btn_validate.clicked.connect(self._run_audit)
        v_head.addWidget(self.btn_validate)
        g3.addLayout(v_head)

        self.txt_audit = QTextEdit()
        self.txt_audit.setReadOnly(True)
        self.txt_audit.setMaximumHeight(90)
        self.txt_audit.setPlaceholderText("Click 'Run Dataset Audit' to scan dataset for missing captions, corrupt images, or invalid formats...")
        g3.addWidget(self.txt_audit)

        layout.addWidget(grp_val)
        layout.addStretch()

    def set_dataset_dir(self, directory: str):
        self.current_dataset_dir = directory

    def _run_audit(self):
        # Scan dataset dir if available
        d_dir = getattr(self, "current_dataset_dir", "")
        if not d_dir or not Path(d_dir).exists():
            self.txt_audit.setHtml("<span style='color: #f85149;'>⚠️ No valid dataset directory specified in Project settings.</span>")
            return

        from engine.dataset import ARBDataset
        warnings = ARBDataset.validate_dataset_dir(d_dir)

        if not warnings:
            self.txt_audit.setHtml("<span style='color: #3fb950;'>✅ Dataset Audit Passed! All images have valid dimensions, format, and accompanying captions.</span>")
        else:
            html = "<span style='color: #d29922; font-weight: bold;'>⚠️ Audit Warnings Found:</span><br>"
            for w in warnings:
                html += f"<span style='color: #f0f6fc;'>• {w}</span><br>"
            self.txt_audit.setHtml(html)

    def load_config(self, config: TrainingConfig):
        self.spin_repeats.setValue(config.dataset_repeats)
        self.edit_ext.setText(config.caption_extension)
        self.chk_shuffle.setChecked(config.shuffle_captions)
        self.spin_keep.setValue(config.keep_tokens)
        self.spin_dropout.setValue(config.caption_dropout_rate)
        self.spin_tag_dropout.setValue(config.tag_dropout_rate)
        self.spin_min_b.setValue(config.min_bucket_resolution)
        self.spin_max_b.setValue(config.max_bucket_resolution)
        self.spin_step.setValue(config.bucket_step)
        self.chk_flip.setChecked(config.horizontal_flip)
        self.chk_color.setChecked(config.color_augmentation)
        self.chk_crop.setChecked(config.random_crop)
        self.current_dataset_dir = config.dataset_dir

    def save_config(self, config: TrainingConfig):
        config.dataset_repeats = self.spin_repeats.value()
        config.caption_extension = self.edit_ext.text().strip() or ".txt"
        config.shuffle_captions = self.chk_shuffle.isChecked()
        config.keep_tokens = self.spin_keep.value()
        config.caption_dropout_rate = self.spin_dropout.value()
        config.tag_dropout_rate = self.spin_tag_dropout.value()
        config.min_bucket_resolution = self.spin_min_b.value()
        config.max_bucket_resolution = self.spin_max_b.value()
        config.bucket_step = self.spin_step.value()
        config.horizontal_flip = self.chk_flip.isChecked()
        config.color_augmentation = self.chk_color.isChecked()
        config.random_crop = self.chk_crop.isChecked()
