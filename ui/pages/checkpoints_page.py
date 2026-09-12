"""
Checkpoint Management & Resume Browser Page.
Controls: Save step interval, Save epoch interval, Save precision, Keep last N checkpoints,
Save optimizer state, and Checkpoint Browser table with file sizes and resume action.
"""
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QFrame, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView
)

from core.config_schema import TrainingConfig


class CheckpointsPage(QWidget):
    """Page 8: Checkpoint saving policies, snapshot cleanup, and checkpoint browser."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.output_dir = "checkpoints/final"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header
        title = QLabel("CHECKPOINTS & STATE RECOVERY")
        title.setProperty("class", "section-title")
        desc = QLabel("Automate intermediate checkpoint backups, configure snapshot pruning, and browse saved weights.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # ── Group 1: Backup Rules ──
        grp_rules = QFrame()
        grp_rules.setProperty("class", "group-box")
        g1 = QGridLayout(grp_rules)
        g1.setSpacing(10)
        g1.setContentsMargins(14, 12, 14, 12)

        g1_title = QLabel("CHECKPOINT SAVING POLICIES")
        g1_title.setProperty("class", "section-title")
        g1.addWidget(g1_title, 0, 0, 1, 4)

        # Save every N steps
        lbl_steps = QLabel("Save Every N Steps:")
        lbl_steps.setProperty("class", "field-label")
        self.spin_save_steps = QSpinBox()
        self.spin_save_steps.setRange(0, 50000)
        self.spin_save_steps.setSingleStep(50)
        self.spin_save_steps.setValue(100)
        self.spin_save_steps.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_steps, 1, 0)
        g1.addWidget(self.spin_save_steps, 1, 1)

        # Save every N epochs
        lbl_epochs = QLabel("Save Every N Epochs:")
        lbl_epochs.setProperty("class", "field-label")
        self.spin_save_epochs = QSpinBox()
        self.spin_save_epochs.setRange(0, 1000)
        self.spin_save_epochs.setValue(0)
        self.spin_save_epochs.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_epochs, 1, 2)
        g1.addWidget(self.spin_save_epochs, 1, 3)

        # Save precision
        lbl_prec = QLabel("Save Precision:")
        lbl_prec.setProperty("class", "field-label")
        self.combo_prec = QComboBox()
        self.combo_prec.addItems(["fp16", "bf16", "float32"])
        self.combo_prec.currentTextChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_prec, 2, 0)
        g1.addWidget(self.combo_prec, 2, 1)

        # Keep last N checkpoints
        lbl_keep = QLabel("Keep Last N Checkpoints (0 = All):")
        lbl_keep.setProperty("class", "field-label")
        self.spin_keep_n = QSpinBox()
        self.spin_keep_n.setRange(0, 100)
        self.spin_keep_n.setValue(0)
        self.spin_keep_n.valueChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(lbl_keep, 2, 2)
        g1.addWidget(self.spin_keep_n, 2, 3)

        # Save Optimizer state
        self.chk_opt_state = QCheckBox("Save Optimizer & Scheduler State")
        self.chk_opt_state.setToolTip("Enables exact seamless training resumption from checkpoint (increases disk usage).")
        self.chk_opt_state.stateChanged.connect(lambda: self.config_changed.emit())
        g1.addWidget(self.chk_opt_state, 3, 0, 1, 2)

        layout.addWidget(grp_rules)

        # ── Group 2: Checkpoint Browser ──
        grp_table = QFrame()
        grp_table.setProperty("class", "group-box")
        g2 = QVBoxLayout(grp_table)
        g2.setSpacing(8)
        g2.setContentsMargins(14, 12, 14, 12)

        t_head = QHBoxLayout()
        t_title = QLabel("CHECKPOINT BROWSER")
        t_title.setProperty("class", "section-title")
        t_head.addWidget(t_title)
        t_head.addStretch()

        self.lbl_resume_active = QLabel("Resume Target: None")
        self.lbl_resume_active.setProperty("class", "meta-badge")
        t_head.addWidget(self.lbl_resume_active)

        self.btn_select_resume = QPushButton("Set As Resume Target")
        self.btn_select_resume.setProperty("class", "browse-btn")
        self.btn_select_resume.clicked.connect(self._on_select_resume)
        t_head.addWidget(self.btn_select_resume)

        self.btn_clear_resume = QPushButton("Clear Resume")
        self.btn_clear_resume.setProperty("class", "browse-btn")
        self.btn_clear_resume.clicked.connect(self._on_clear_resume)
        t_head.addWidget(self.btn_clear_resume)

        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.setProperty("class", "browse-btn")
        self.btn_refresh.clicked.connect(self.refresh_checkpoints)
        t_head.addWidget(self.btn_refresh)
        g2.addLayout(t_head)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Filename", "Date Modified", "Size (MB)", "Path"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(200)
        g2.addWidget(self.table)

        layout.addWidget(grp_table)
        layout.addStretch()

    def set_output_dir(self, directory: str):
        self.output_dir = directory
        self.refresh_checkpoints()

    def refresh_checkpoints(self):
        """Scans both the output directory and backups folder for .safetensors."""
        paths_to_scan = [Path(self.output_dir), Path("checkpoints/backups")]
        files = []
        for p in paths_to_scan:
            if p.exists():
                for f in p.glob("*.safetensors"):
                    files.append(f)

        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        self.table.setRowCount(len(files))
        for row, f in enumerate(files):
            stat = f.stat()
            size_mb = f"{stat.st_size / (1024 * 1024):.1f} MB"
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

            item_name = QTableWidgetItem(f.name)
            item_date = QTableWidgetItem(mtime)
            item_size = QTableWidgetItem(size_mb)
            item_path = QTableWidgetItem(str(f))

            self.table.setItem(row, 0, item_name)
            self.table.setItem(row, 1, item_date)
            self.table.setItem(row, 2, item_size)
            self.table.setItem(row, 3, item_path)

    def _on_select_resume(self):
        row = self.table.currentRow()
        if row >= 0:
            path_item = self.table.item(row, 3)
            if path_item:
                self.resume_path = path_item.text().strip()
                name_item = self.table.item(row, 0)
                self.lbl_resume_active.setText(f"Resume Target: {name_item.text()}")
                self.lbl_resume_active.setStyleSheet("color: #79c0ff;")
                self.config_changed.emit()

    def _on_clear_resume(self):
        self.resume_path = ""
        self.lbl_resume_active.setText("Resume Target: None")
        self.lbl_resume_active.setStyleSheet("color: #8b949e;")
        self.config_changed.emit()

    def load_config(self, config: TrainingConfig):
        self.spin_save_steps.setValue(config.save_every_n_steps)
        self.spin_save_epochs.setValue(config.save_every_n_epochs)
        self.combo_prec.setCurrentText(config.save_precision)
        self.spin_keep_n.setValue(config.keep_last_n_checkpoints)
        self.chk_opt_state.setChecked(config.save_optimizer_state)
        self.resume_path = config.resume_from_checkpoint
        if self.resume_path and Path(self.resume_path).exists():
            self.lbl_resume_active.setText(f"Resume Target: {Path(self.resume_path).name}")
            self.lbl_resume_active.setStyleSheet("color: #79c0ff;")
        else:
            self.lbl_resume_active.setText("Resume Target: None")
            self.lbl_resume_active.setStyleSheet("color: #8b949e;")
        self.output_dir = config.output_dir
        self.refresh_checkpoints()

    def save_config(self, config: TrainingConfig):
        config.save_every_n_steps = self.spin_save_steps.value()
        config.save_every_n_epochs = self.spin_save_epochs.value()
        config.save_precision = self.combo_prec.currentText()
        config.keep_last_n_checkpoints = self.spin_keep_n.value()
        config.save_optimizer_state = self.chk_opt_state.isChecked()
        config.resume_from_checkpoint = getattr(self, "resume_path", "")
