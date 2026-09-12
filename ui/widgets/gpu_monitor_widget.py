"""
Compact live GPU monitor widget displaying GPU name, VRAM gauge, utilization bar, temperature, and power.
"""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame, QGridLayout
)

from core.gpu_monitor import GPUMonitor, GPUStats


class GPUMonitorWidget(QFrame):
    """Compact telemetry card for real-time GPU/VRAM health."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "group-box")
        self.monitor = GPUMonitor()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header Row: Title & GPU Name
        header_row = QHBoxLayout()
        title_lbl = QLabel("HARDWARE MONITOR")
        title_lbl.setProperty("class", "section-title")
        self.lbl_gpu_name = QLabel("Detecting...")
        self.lbl_gpu_name.setProperty("class", "meta-badge")
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        header_row.addWidget(self.lbl_gpu_name)
        layout.addLayout(header_row)

        # Grid for gauges
        grid = QGridLayout()
        grid.setSpacing(6)

        # VRAM Row
        grid.addWidget(QLabel("VRAM:"), 0, 0)
        self.bar_vram = QProgressBar()
        self.bar_vram.setRange(0, 100)
        self.bar_vram.setValue(0)
        grid.addWidget(self.bar_vram, 0, 1)
        self.lbl_vram_val = QLabel("0.0 / 0.0 GB")
        self.lbl_vram_val.setStyleSheet("font-family: monospace; font-size: 11px;")
        grid.addWidget(self.lbl_vram_val, 0, 2)

        # GPU Utilization Row
        grid.addWidget(QLabel("GPU Load:"), 1, 0)
        self.bar_util = QProgressBar()
        self.bar_util.setRange(0, 100)
        self.bar_util.setValue(0)
        grid.addWidget(self.bar_util, 1, 1)
        self.lbl_util_val = QLabel("0%")
        self.lbl_util_val.setStyleSheet("font-family: monospace; font-size: 11px;")
        grid.addWidget(self.lbl_util_val, 1, 2)

        layout.addLayout(grid)

        # Bottom metrics row (Temp, Power)
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(16)
        
        self.lbl_temp = QLabel("Temp: -- °C")
        self.lbl_temp.setProperty("class", "meta-badge")
        self.lbl_power = QLabel("Power: -- W")
        self.lbl_power.setProperty("class", "meta-badge")
        
        metrics_row.addWidget(self.lbl_temp)
        metrics_row.addWidget(self.lbl_power)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # Auto-refresh timer (1 second)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.update_stats()

    def update_stats(self):
        stats: GPUStats = self.monitor.get_stats()
        self.lbl_gpu_name.setText(stats.name)

        # VRAM
        vram_pct = int(stats.vram_pct)
        self.bar_vram.setValue(vram_pct)
        self.lbl_vram_val.setText(f"{stats.vram_used_gb:.1f} / {stats.vram_total_gb:.1f} GB")

        # Color VRAM progress bar dynamically
        if vram_pct > 90:
            self.bar_vram.setStyleSheet("QProgressBar::chunk { background-color: #f85149; }")
        elif vram_pct > 75:
            self.bar_vram.setStyleSheet("QProgressBar::chunk { background-color: #d29922; }")
        else:
            self.bar_vram.setStyleSheet("QProgressBar::chunk { background-color: #1f6feb; }")

        # Util
        self.bar_util.setValue(stats.utilization_pct)
        self.lbl_util_val.setText(f"{stats.utilization_pct}%")

        # Temp & Power
        if stats.temperature_c > 0:
            self.lbl_temp.setText(f"Temp: {stats.temperature_c}°C")
            if stats.temperature_c > 80:
                self.lbl_temp.setStyleSheet("color: #f85149;")
            else:
                self.lbl_temp.setStyleSheet("color: #8b949e;")
        else:
            self.lbl_temp.setText("Temp: N/A")

        if stats.power_draw_w > 0:
            self.lbl_power.setText(f"Power: {stats.power_draw_w:.0f}W / {stats.power_limit_w:.0f}W")
        else:
            self.lbl_power.setText("Power: N/A")
