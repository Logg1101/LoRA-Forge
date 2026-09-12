import logging
import multiprocessing
import sys
import warnings

# Suppress harmless startup warnings (triton kernels on Windows, deprecated pynvml)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from ui.app import launch_ui

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Windows + CUDA requires "spawn" start method
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)

    launch_ui()