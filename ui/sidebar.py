"""
Vertical Navigation Sidebar for LoRA Forge.
Sections: PROJECT, DATASET, NETWORK, TRAINING, MEMORY, LOSS, SAMPLES, CHECKPOINTS, ADVANCED.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QButtonGroup


class Sidebar(QWidget):
    """Left navigation panel with section buttons."""

    page_selected = Signal(int)

    SECTIONS = [
        ("MODEL & DATA", "📁"),
        ("ARCHITECTURE", "🧬"),
        ("TRAINING & HW", "⚡"),
        ("TENSORBOARD", "📊"),
        ("OUTPUT & LOGS", "💾"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        self.buttons = []
        for idx, (label, icon) in enumerate(self.SECTIONS):
            btn = QPushButton(f" {icon}  {label}")
            btn.setProperty("class", "nav-btn")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked=False, i=idx: self.page_selected.emit(i))

            self.button_group.addButton(btn, idx)
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def set_current_index(self, index: int):
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)
