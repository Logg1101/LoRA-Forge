"""
Top Bar Header for LoRA Forge.
Contains: Application title, Architecture badge, Training status indicator, and Start / Pause / Resume / Stop action buttons.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QFrame
)


class TopBar(QWidget):
    """Header bar with project identity and execution controls."""

    start_clicked = Signal()
    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        # App Identity
        self.lbl_app_name = QLabel("⚡ LoRA Forge")
        self.lbl_app_name.setObjectName("appNameLabel")
        layout.addWidget(self.lbl_app_name)

        self.lbl_version = QLabel("v2.0")
        self.lbl_version.setObjectName("appVersionLabel")
        layout.addWidget(self.lbl_version)

        # Architecture Badge
        self.lbl_arch = QLabel("SDXL")
        self.lbl_arch.setObjectName("archBadge")
        layout.addWidget(self.lbl_arch)

        layout.addSpacing(12)

        # Status Indicator Pill
        self.lbl_status = QLabel("● IDLE")
        self.lbl_status.setObjectName("statusIndicator")
        self.lbl_status.setProperty("status", "IDLE")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        # Action Buttons
        self.btn_start = QPushButton("▶  Start Training")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setObjectName("btnPause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_clicked.emit)
        layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("⏯  Resume")
        self.btn_resume.setObjectName("btnResume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self.resume_clicked.emit)
        layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("🛑  Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.btn_stop)

    def set_arch(self, arch: str):
        self.lbl_arch.setText(arch)

    def set_status(self, status: str):
        self.lbl_status.setText(f"● {status}")
        self.lbl_status.setProperty("status", status)
        # Refresh style
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)

        # Update button enable/disable states based on lifecycle
        is_training = status in ("TRAINING", "PREPARING")
        is_paused = status == "PAUSED"

        self.btn_start.setEnabled(not is_training and not is_paused)
        self.btn_pause.setEnabled(is_training)
        self.btn_resume.setEnabled(is_paused)
        self.btn_stop.setEnabled(is_training or is_paused)
