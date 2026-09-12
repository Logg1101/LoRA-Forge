"""
Bottom Status Bar for LoRA Forge.
Displays real-time telemetry: Step / Total, Epoch / Total, Loss, Learning Rate, ETA, Speed.
"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


class StatusBar(QWidget):
    """Bottom telemetry bar with real-time training metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(18)

        # Step
        self.lbl_step = self._add_metric("STEP:", "0 / 0")
        layout.addLayout(self.lbl_step[0])

        # Epoch
        self.lbl_epoch = self._add_metric("EPOCH:", "0 / 0")
        layout.addLayout(self.lbl_epoch[0])

        # Loss
        self.lbl_loss = self._add_metric("LOSS:", "--")
        layout.addLayout(self.lbl_loss[0])

        # LR
        self.lbl_lr = self._add_metric("LR:", "--")
        layout.addLayout(self.lbl_lr[0])

        # Speed
        self.lbl_speed = self._add_metric("SPEED:", "-- it/s")
        layout.addLayout(self.lbl_speed[0])

        # ETA
        self.lbl_eta = self._add_metric("ETA:", "--:--:--")
        layout.addLayout(self.lbl_eta[0])

        layout.addStretch()

    def _add_metric(self, name: str, default_val: str):
        row = QHBoxLayout()
        row.setSpacing(6)

        lbl_name = QLabel(name)
        lbl_name.setObjectName("statusMetricLabel")

        lbl_val = QLabel(default_val)
        lbl_val.setObjectName("statusMetricValue")

        row.addWidget(lbl_name)
        row.addWidget(lbl_val)
        return (row, lbl_val)

    def update_telemetry(self, telemetry: dict):
        step = telemetry.get("step", 0)
        tot_steps = telemetry.get("total_steps", 0)
        self.lbl_step[1].setText(f"{step:,} / {tot_steps:,}" if tot_steps > 0 else f"{step:,}")

        epoch = telemetry.get("epoch", 0)
        tot_epochs = telemetry.get("total_epochs", 0)
        self.lbl_epoch[1].setText(f"{epoch} / {tot_epochs}")

        loss = telemetry.get("loss", 0.0)
        self.lbl_loss[1].setText(f"{loss:.4f}" if loss > 0 else "--")

        lr = telemetry.get("lr", 0.0)
        self.lbl_lr[1].setText(f"{lr:.2e}" if lr > 0 else "--")

        speed = telemetry.get("steps_per_sec", 0.0)
        if speed > 0:
            if speed < 1.0:
                self.lbl_speed[1].setText(f"{1.0 / speed:.1f} s/it")
            else:
                self.lbl_speed[1].setText(f"{speed:.2f} it/s")
        else:
            self.lbl_speed[1].setText("--")

        eta_s = telemetry.get("eta_seconds", 0)
        if eta_s > 0:
            hrs = int(eta_s // 3600)
            mins = int((eta_s % 3600) // 60)
            secs = int(eta_s % 60)
            self.lbl_eta[1].setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")
        else:
            self.lbl_eta[1].setText("--:--:--")
