"""
Log Console Widget with timestamping, log levels, autoscroll, and collapsible view.
"""
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton
)


class LogConsoleWidget(QWidget):
    """Collapsible dark log viewer with structured message formatting."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header Row
        header = QHBoxLayout()
        title = QLabel("SYSTEM CONSOLE")
        title.setProperty("class", "section-title")
        header.addWidget(title)
        header.addStretch()

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setProperty("class", "browse-btn")
        self.btn_clear.clicked.connect(self.clear)

        self.btn_autoscroll = QPushButton("Auto-Scroll: ON")
        self.btn_autoscroll.setProperty("class", "browse-btn")
        self.btn_autoscroll.setCheckable(True)
        self.btn_autoscroll.setChecked(True)
        self.btn_autoscroll.clicked.connect(self._toggle_autoscroll)

        header.addWidget(self.btn_autoscroll)
        header.addWidget(self.btn_clear)
        layout.addLayout(header)

        # Console Text Display
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #07090e;
                color: #c9d1d9;
                font-family: 'Consolas', 'Cascadia Code', monospace;
                font-size: 11px;
                border: 1px solid #21262d;
                border-radius: 4px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.text_edit)

    def log(self, message: str, level: str = "INFO"):
        """Append a structured log message."""
        time_str = datetime.now().strftime("%H:%M:%S")

        # Color-coded levels
        level_colors = {
            "INFO": "#58a6ff",
            "WARN": "#d29922",
            "ERROR": "#f85149",
            "SUCCESS": "#3fb950",
            "TRAIN": "#a371f7",
        }
        color = level_colors.get(level.upper(), "#8b949e")

        html = f"<span style='color: #6e7681;'>{time_str}</span> " \
               f"<span style='color: {color}; font-weight: bold;'>[{level.upper()}]</span> " \
               f"<span style='color: #f0f6fc;'>{message}</span><br>"

        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertHtml(html)

        if self.btn_autoscroll.isChecked():
            self.text_edit.moveCursor(QTextCursor.End)

    def _toggle_autoscroll(self):
        if self.btn_autoscroll.isChecked():
            self.btn_autoscroll.setText("Auto-Scroll: ON")
        else:
            self.btn_autoscroll.setText("Auto-Scroll: OFF")

    def clear(self):
        self.text_edit.clear()
