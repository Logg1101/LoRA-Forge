"""
Embedded TensorBoard Dashboard Page.
Controls: Live TensorBoard server management, embedded web browser view,
and external browser launcher for multi-metric training tracking.
"""
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import Signal, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


def _find_free_port(start_port=6006):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


class TensorboardPage(QWidget):
    """Page for interacting with real-time TensorBoard metrics and embedded charts."""

    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_dir = "checkpoints/final/logs"
        self.port = 6006
        self._process = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header
        title = QLabel("TENSORBOARD METRICS & DASHBOARD")
        title.setProperty("class", "section-title")
        desc = QLabel("Inspect live loss curves, learning rate dynamics, and hardware metrics directly in the workstation.")
        desc.setProperty("class", "section-desc")
        layout.addWidget(title)
        layout.addWidget(desc)

        # Toolbar
        bar = QFrame()
        bar.setProperty("class", "group-box")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(14, 10, 14, 10)
        bar_layout.setSpacing(10)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #8b949e; font-weight: 600;")
        bar_layout.addWidget(self.lbl_status)
        bar_layout.addStretch()

        self.btn_toggle = QPushButton("Start TensorBoard")
        self.btn_toggle.setProperty("class", "action-btn")
        self.btn_toggle.clicked.connect(self._on_toggle_clicked)
        bar_layout.addWidget(self.btn_toggle)

        self.btn_reload = QPushButton("Reload")
        self.btn_reload.setProperty("class", "browse-btn")
        self.btn_reload.clicked.connect(self.reload_view)
        bar_layout.addWidget(self.btn_reload)

        self.btn_browser = QPushButton("Open in Browser")
        self.btn_browser.setProperty("class", "browse-btn")
        self.btn_browser.clicked.connect(self.open_in_browser)
        bar_layout.addWidget(self.btn_browser)

        layout.addWidget(bar)

        # Web View or Fallback
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(550)
            self._set_placeholder_html()
            layout.addWidget(self.web_view, stretch=1)
        else:
            self.web_view = None
            fallback = QLabel("WebEngine not available. Use 'Open in Browser' to view TensorBoard.")
            fallback.setStyleSheet("color: #8b949e; padding: 40px; font-size: 14px;")
            layout.addWidget(fallback, stretch=1)

    def _set_placeholder_html(self):
        if self.web_view:
            self.web_view.setHtml(
                """
                <html>
                <body style="background-color: #0d1117; color: #8b949e; font-family: sans-serif;
                             display: flex; flex-direction: column; justify-content: center;
                             align-items: center; height: 90vh; text-align: center;">
                    <h2 style="color: #c9d1d9;">TensorBoard Offline</h2>
                    <p>Start a training run or click <b>Start TensorBoard</b> to launch live metrics.</p>
                </body>
                </html>
                """
            )

    def start_server(self, log_dir=None):
        if log_dir:
            self.log_dir = str(log_dir)

        if self._process is not None and self._process.poll() is None:
            return

        self.port = _find_free_port(6006)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(self.log_dir),
            "--port", str(self.port),
            "--reload_interval", "3",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        self.lbl_status.setText(f"Status: Running on port {self.port}")
        self.lbl_status.setStyleSheet("color: #3fb950; font-weight: 600;")
        self.btn_toggle.setText("Stop TensorBoard")

        # Load into web view with small delay for server spin-up
        if self.web_view:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, self.reload_view)

    def stop_server(self):
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        self.lbl_status.setText("Status: Stopped")
        self.lbl_status.setStyleSheet("color: #8b949e; font-weight: 600;")
        self.btn_toggle.setText("Start TensorBoard")
        self._set_placeholder_html()

    def _on_toggle_clicked(self):
        if self._process is not None and self._process.poll() is None:
            self.stop_server()
        else:
            self.start_server()

    def reload_view(self):
        if self.web_view and self._process is not None:
            self.web_view.load(QUrl(f"http://localhost:{self.port}"))

    def open_in_browser(self):
        if self._process is None or self._process.poll() is not None:
            self.start_server()
        webbrowser.open(f"http://localhost:{self.port}")

    def load_config(self, config):
        self.log_dir = str(Path(config.output_dir) / "logs")

    def save_config(self, config):
        pass
