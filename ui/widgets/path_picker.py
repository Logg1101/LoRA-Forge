"""
Compact path picker widget with inline text input, browse button, and validation indicator.
"""
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QLabel
)


class PathPicker(QWidget):
    """Line edit with an integrated browse button for files or directories."""

    path_changed = Signal(str)

    def __init__(
        self,
        mode: str = "dir",  # "dir" or "file"
        file_filter: str = "All Files (*.*)",
        placeholder: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.textChanged.connect(self._on_text_changed)

        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setProperty("class", "browse-btn")
        self.btn_browse.clicked.connect(self._on_browse)

        layout.addWidget(self.line_edit, stretch=1)
        layout.addWidget(self.btn_browse)

    def _on_browse(self):
        current_text = self.line_edit.text().strip()
        start_dir = current_text if current_text and Path(current_text).exists() else ""

        if self.mode == "dir":
            selected = QFileDialog.getExistingDirectory(
                self, "Select Directory", start_dir
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Select File", start_dir, self.file_filter
            )

        if selected:
            # Normalize to forward slashes
            norm = selected.replace("\\", "/")
            self.line_edit.setText(norm)

    def _on_text_changed(self, text: str):
        self.path_changed.emit(text.strip())

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, text: str):
        self.line_edit.setText(text)

    def setPlaceholderText(self, text: str):
        self.line_edit.setPlaceholderText(text)
