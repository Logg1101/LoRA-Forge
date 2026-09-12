"""
Main Application Window for LoRA Forge.
Integrates TopBar, Sidebar, Pages Stack, TrainingPlanWidget, and StatusBar with live telemetry polling.
"""
import logging
import sys
import warnings
from pathlib import Path

# Suppress harmless startup warnings (triton kernels on Windows, deprecated pynvml)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QScrollArea, QMessageBox
)

from core.config_schema import TrainingConfig
from core.config_handler import load_config, save_config, save_preset
from core.state_manager import TrainingStateManager
from engine.trainer import run_training_loop
from ui.theme import DARK_THEME_QSS
from ui.topbar import TopBar
from ui.sidebar import Sidebar
from ui.statusbar import StatusBar
from ui.widgets.training_plan_widget import TrainingPlanWidget
from ui.pages.project_page import ProjectPage
from ui.pages.dataset_page import DatasetPage
from ui.pages.network_page import NetworkPage
from ui.pages.training_page import TrainingPage
from ui.pages.memory_page import MemoryPage
from ui.pages.loss_page import LossPage
from ui.pages.samples_page import SamplesPage
from ui.pages.checkpoints_page import CheckpointsPage
from ui.pages.advanced_page import AdvancedPage
from ui.pages.tensorboard_page import TensorboardPage


def _combine_pages(*pages):
    """Stack multiple page widgets into a single vertically scrolling view."""
    container = QWidget()
    vbox = QVBoxLayout(container)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(16)
    for p in pages:
        lay = p.layout()
        if lay and lay.count() > 0:
            last_item = lay.itemAt(lay.count() - 1)
            if last_item.spacerItem() is not None:
                lay.takeAt(lay.count() - 1)
        vbox.addWidget(p)
    vbox.addStretch()
    return container


class LoRAForgeApp(QMainWindow):
    """Main workstation application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoRA Forge — AI Training Workstation")
        self.resize(1280, 840)
        self.setMinimumSize(1024, 700)

        # Core State & Config
        self.config = load_config()
        self.state_manager = TrainingStateManager()

        # Build Main UI Shell
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Bar
        self.top_bar = TopBar()
        self.top_bar.start_clicked.connect(self._on_start_training)
        self.top_bar.pause_clicked.connect(self._on_pause_training)
        self.top_bar.resume_clicked.connect(self._on_resume_training)
        self.top_bar.stop_clicked.connect(self._on_stop_training)
        main_layout.addWidget(self.top_bar)

        # 2. Middle Body (Sidebar + Content Workspace)
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.page_selected.connect(self._on_page_selected)
        body_layout.addWidget(self.sidebar)

        # Right Side: Split between Stacked Pages and Training Plan
        content_split = QWidget()
        content_layout = QHBoxLayout(content_split)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Scrollable Page Stack
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.pages_stack = QStackedWidget()

        # Instantiate all 9 pages
        self.page_project = ProjectPage()
        self.page_dataset = DatasetPage()
        self.page_network = NetworkPage()
        self.page_training = TrainingPage()
        self.page_memory = MemoryPage()
        self.page_loss = LossPage()
        self.page_samples = SamplesPage()
        self.page_checkpoints = CheckpointsPage()
        self.page_advanced = AdvancedPage()
        self.page_tensorboard = TensorboardPage()

        self.pages = [
            self.page_project,
            self.page_dataset,
            self.page_network,
            self.page_training,
            self.page_memory,
            self.page_loss,
            self.page_samples,
            self.page_checkpoints,
            self.page_advanced,
            self.page_tensorboard,
        ]

        for p in self.pages:
            p.config_changed.connect(self._on_ui_config_changed)

        # 5 Consolidated Views (Model & Data, Architecture, Training & HW, TensorBoard, Output & Logs)
        self.view_model_data = _combine_pages(self.page_project, self.page_dataset)
        self.view_architecture = self.page_network
        self.view_training_hw = _combine_pages(self.page_training, self.page_loss, self.page_memory)
        self.view_output_logs = _combine_pages(self.page_samples, self.page_checkpoints, self.page_advanced)

        self.consolidated_views = [
            self.view_model_data,
            self.view_architecture,
            self.view_training_hw,
            self.page_tensorboard,
            self.view_output_logs,
        ]

        for v in self.consolidated_views:
            self.pages_stack.addWidget(v)

        self.page_advanced.preset_loaded.connect(self._on_preset_loaded)

        self.scroll_area.setWidget(self.pages_stack)
        content_layout.addWidget(self.scroll_area, stretch=3)

        # Right-docked Training Plan Preview Widget
        self.plan_widget = TrainingPlanWidget()
        self.plan_widget.setMinimumWidth(320)
        self.plan_widget.setMaximumWidth(360)
        content_layout.addWidget(self.plan_widget, stretch=1)

        body_layout.addWidget(content_split, stretch=1)
        main_layout.addWidget(body, stretch=1)

        # 3. Status Bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        # Load initial config into all pages
        self._load_config_to_ui(self.config)

        # Telemetry Polling Timer (250ms)
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self._poll_telemetry)
        self.telemetry_timer.start(250)

    def _on_page_selected(self, index: int):
        self.pages_stack.setCurrentIndex(index)

    def _load_config_to_ui(self, cfg: TrainingConfig):
        for p in self.pages:
            p.load_config(cfg)
        self.top_bar.set_arch(cfg.model_type)
        self._update_plan_preview()

    def _sync_ui_to_config(self):
        for p in self.pages:
            p.save_config(self.config)
        self.top_bar.set_arch(self.config.model_type)
        self.page_dataset.set_dataset_dir(self.config.dataset_dir)
        self.page_checkpoints.set_output_dir(self.config.output_dir)

    def _on_ui_config_changed(self):
        self._sync_ui_to_config()
        # Save to disk automatically
        save_config(self.config)
        # Check if preset save was requested
        if hasattr(self.page_advanced, "save_preset_requested") and self.page_advanced.save_preset_requested:
            save_preset(self.config, self.page_advanced.save_preset_requested)
            self.page_advanced.save_preset_requested = None
            self.page_advanced.log_message("Preset saved successfully.", "SUCCESS")
        self._update_plan_preview()

    def _on_preset_loaded(self, preset_dict: dict):
        self.config = TrainingConfig.from_legacy_dict(preset_dict)
        self._load_config_to_ui(self.config)
        save_config(self.config)

    def _update_plan_preview(self):
        # Estimate image count from dataset dir
        img_count = 0
        p = Path(self.config.dataset_dir)
        if p.exists() and p.is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            img_count = len([f for f in p.rglob("*") if f.suffix.lower() in valid_exts])

        self.plan_widget.update_plan(self.config, image_count=img_count)

    def _on_start_training(self):
        self._sync_ui_to_config()

        # 1. Run fatal error validation
        errors = self.config.validate_for_training()
        if errors:
            err_text = "\n• " + "\n• ".join(errors)
            QMessageBox.critical(
                self,
                "Cannot Start Training",
                f"Please fix the following configuration errors before starting training:\n{err_text}",
            )
            self.page_advanced.log_message("Training aborted: configuration errors detected.", "ERROR")
            return

        # 2. Run advisory validation warnings
        warnings = self.config.validate_config()
        if warnings:
            warn_text = "\n• " + "\n• ".join(warnings)
            reply = QMessageBox.question(
                self,
                "Configuration Warnings",
                f"The following potential configuration issues were detected:\n{warn_text}\n\nDo you want to proceed with training anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Pass engine dict to background process
        engine_dict = self.config.to_engine_dict()
        self.state_manager.start_training(engine_dict, run_training_loop)
        self.page_advanced.log_message(f"Training started: '{self.config.project_name}' on {self.config.model_type}", "TRAIN")

        # Auto-launch TensorBoard server
        tb_dir = Path(self.config.output_dir) / "logs"
        self.page_tensorboard.start_server(tb_dir)

    def _on_pause_training(self):
        self.state_manager.pause_training()
        self.page_advanced.log_message("Pausing engine...", "WARN")

    def _on_resume_training(self):
        self.state_manager.resume_training()
        self.page_advanced.log_message("Resuming engine...", "TRAIN")

    def _on_stop_training(self):
        self.state_manager.stop_training()
        self.page_advanced.log_message("Stop signal sent. Final weights saving...", "WARN")

    def _poll_telemetry(self):
        """Timer callback that reads telemetry dict and refreshes UI widgets."""
        status = self.state_manager.get_status()
        telemetry = dict(self.state_manager.telemetry)
        telemetry["status"] = status

        self.top_bar.set_status(status)
        self.status_bar.update_telemetry(telemetry)

        # Update loss graph
        step = telemetry.get("step", 0)
        loss = telemetry.get("loss", 0.0)
        if step > 0 and loss > 0:
            self.page_loss.update_loss_point(step, loss)

        # Check for fatal error
        err = telemetry.get("error_message", "")
        if status == "ERROR" and err and not getattr(self, "_error_dialog_shown", False):
            self._error_dialog_shown = True
            tb = telemetry.get("error_traceback", "")
            self.page_advanced.log_message(f"ERROR: {err}", "ERROR")

            # Friendly message
            sugg = "Suggested Actions:\n• Enable Gradient Checkpointing\n• Enable Latent Caching\n• Reduce Training Resolution\n• Enable Base Model Quantization (NF4)\n• Reduce Batch Size"
            QMessageBox.critical(
                self,
                "Training Failed",
                f"An error occurred during training:\n\n{err}\n\n{sugg}\n\nTechnical details available in System Console."
            )
        elif status != "ERROR":
            self._error_dialog_shown = False

    def closeEvent(self, event):
        if hasattr(self, "page_tensorboard"):
            self.page_tensorboard.stop_server()
        super().closeEvent(event)


def launch_ui():
    """Application entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME_QSS)

    window = LoRAForgeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_ui()
