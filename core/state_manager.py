"""
Enhanced training state manager with telemetry for the PySide6 UI.
The telemetry dict is shared between the UI process and the training child process.
"""
import multiprocessing
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TrainingStateManager:
    """Controls training lifecycle and provides telemetry to the UI."""

    STATUS_IDLE = "IDLE"
    STATUS_PREPARING = "PREPARING"
    STATUS_TRAINING = "TRAINING"
    STATUS_PAUSING = "PAUSING"
    STATUS_PAUSED = "PAUSED"
    STATUS_RESUMING = "RESUMING"
    STATUS_SAVING = "SAVING"
    STATUS_COMPLETED = "COMPLETED"
    STATUS_ERROR = "ERROR"
    STATUS_STOPPED = "STOPPED"

    def __init__(self):
        self.process = None
        self._manager = multiprocessing.Manager()

        # Events for cross-process signaling
        self.run_event = multiprocessing.Event()
        self.stop_event = multiprocessing.Event()

        # Shared telemetry dict — trainer writes, UI reads
        self.telemetry = self._manager.dict({
            "status": self.STATUS_IDLE,
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "lr": 0.0,
            "eta_seconds": 0,
            "error_message": "",
            "error_traceback": "",
            "steps_per_sec": 0.0,
        })

    def start_training(self, config_dict: dict, training_function):
        """Spawn the training engine in a background process."""
        if self.process is not None and self.process.is_alive():
            return

        self.run_event.set()
        self.stop_event.clear()

        # Reset telemetry
        self.telemetry["status"] = self.STATUS_PREPARING
        self.telemetry["epoch"] = 0
        self.telemetry["step"] = 0
        self.telemetry["loss"] = 0.0
        self.telemetry["lr"] = 0.0
        self.telemetry["eta_seconds"] = 0
        self.telemetry["error_message"] = ""
        self.telemetry["error_traceback"] = ""
        self.telemetry["steps_per_sec"] = 0.0

        self.process = multiprocessing.Process(
            target=self._run_with_error_handling,
            args=(training_function, config_dict, self.run_event, self.stop_event, self.telemetry),
        )
        self.process.start()

    @staticmethod
    def _run_with_error_handling(training_fn, config, run_event, stop_event, telemetry):
        """Wrapper that catches exceptions and writes them to telemetry."""
        try:
            training_fn(config, run_event, stop_event, telemetry)
            if not stop_event.is_set():
                if telemetry.get("step", 0) == 0 and telemetry.get("total_steps", 0) > 0:
                    telemetry["status"] = TrainingStateManager.STATUS_ERROR
                    telemetry["error_message"] = "Training ended with 0 steps executed. Check dataset and batch size settings."
                else:
                    telemetry["status"] = TrainingStateManager.STATUS_COMPLETED
            else:
                telemetry["status"] = TrainingStateManager.STATUS_STOPPED
        except BaseException as e:
            telemetry["status"] = TrainingStateManager.STATUS_ERROR
            telemetry["error_message"] = str(e)
            telemetry["error_traceback"] = traceback.format_exc()
            print(f"\n{'='*60}")
            print(f"TRAINING ERROR: {e}")
            print(traceback.format_exc())
            print(f"{'='*60}")

    def pause_training(self):
        if self.process and self.process.is_alive():
            self.telemetry["status"] = self.STATUS_PAUSING
            self.run_event.clear()
            self.telemetry["status"] = self.STATUS_PAUSED

    def resume_training(self):
        if self.process and self.process.is_alive():
            self.telemetry["status"] = self.STATUS_RESUMING
            self.run_event.set()
            self.telemetry["status"] = self.STATUS_TRAINING

    def stop_training(self):
        if self.process and self.process.is_alive():
            self.telemetry["status"] = self.STATUS_SAVING
            self.stop_event.set()
            self.run_event.set()  # Wake if paused
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
            self.telemetry["status"] = self.STATUS_STOPPED

    def is_running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def get_status(self) -> str:
        # Check if process terminated abnormally while in an active state
        if self.process is not None and not self.process.is_alive():
            curr = str(self.telemetry.get("status", self.STATUS_IDLE))
            if curr in (self.STATUS_PREPARING, self.STATUS_TRAINING, self.STATUS_PAUSING, self.STATUS_RESUMING):
                self.telemetry["status"] = self.STATUS_ERROR
                self.telemetry["error_message"] = f"Training process terminated unexpectedly (exit code {self.process.exitcode})."
        return str(self.telemetry.get("status", self.STATUS_IDLE))