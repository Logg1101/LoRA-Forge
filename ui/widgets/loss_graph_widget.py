"""
Real-time high-performance loss graph widget using pyqtgraph.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
import pyqtgraph as pg


class LossGraphWidget(QWidget):
    """Real-time loss curve plotter using pyqtgraph."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()
        title = QLabel("TRAINING LOSS CURVE")
        title.setProperty("class", "section-title")
        header.addWidget(title)
        header.addStretch()

        self.btn_clear = QPushButton("Clear Graph")
        self.btn_clear.setProperty("class", "browse-btn")
        self.btn_clear.clicked.connect(self.clear)
        header.addWidget(self.btn_clear)
        layout.addLayout(header)

        # Configure pyqtgraph styling for dark workstation
        pg.setConfigOption("background", "#0d1117")
        pg.setConfigOption("foreground", "#8b949e")
        pg.setConfigOptions(antialias=True)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel("left", "Loss", color="#8b949e", size="10pt")
        self.plot_widget.setLabel("bottom", "Step", color="#8b949e", size="10pt")
        self.plot_widget.getAxis("left").setPen(pg.mkPen(color="#30363d", width=1))
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen(color="#30363d", width=1))

        # Main loss curve
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#58a6ff", width=1.5),
            name="Training Loss",
        )

        self.steps_data = []
        self.loss_data = []

        layout.addWidget(self.plot_widget)

    def update_loss(self, step: int, loss: float):
        """Append a new loss point and refresh the curve."""
        if loss <= 0 or step <= 0:
            return
        self.steps_data.append(step)
        self.loss_data.append(loss)
        self.curve.setData(self.steps_data, self.loss_data)

    def set_data(self, steps: list[int], losses: list[float]):
        """Set entire dataset at once."""
        self.steps_data = list(steps)
        self.loss_data = list(losses)
        self.curve.setData(self.steps_data, self.loss_data)

    def clear(self):
        """Reset loss plot."""
        self.steps_data.clear()
        self.loss_data.clear()
        self.curve.setData([], [])
