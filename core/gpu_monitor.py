"""
GPU hardware monitor using pynvml (NVIDIA) with torch.cuda fallback.
Provides polling interface for the UI status bar.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GPUStats:
    name: str = "Unknown"
    vram_used_mb: float = 0
    vram_total_mb: float = 0
    utilization_pct: int = 0
    temperature_c: int = 0
    power_draw_w: float = 0
    power_limit_w: float = 0
    driver_version: str = ""

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_mb / 1024

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0
        return (self.vram_used_mb / self.vram_total_mb) * 100


class GPUMonitor:
    """Polls GPU stats. Tries pynvml/nvidia-ml-py first, falls back to torch.cuda."""

    def __init__(self):
        self._use_pynvml = False
        self._handle = None
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                import pynvml
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._use_pynvml = True
        except Exception:
            pass

    def get_stats(self) -> GPUStats:
        if self._use_pynvml:
            return self._poll_pynvml()
        return self._poll_torch()

    def _poll_pynvml(self) -> GPUStats:
        import pynvml
        h = self._handle
        try:
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            except Exception:
                power = 0
                power_limit = 0
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8")
            except Exception:
                driver = ""
            return GPUStats(
                name=name,
                vram_used_mb=mem.used / (1024 * 1024),
                vram_total_mb=mem.total / (1024 * 1024),
                utilization_pct=util.gpu,
                temperature_c=temp,
                power_draw_w=power,
                power_limit_w=power_limit,
                driver_version=driver,
            )
        except Exception:
            return GPUStats()

    def _poll_torch(self) -> GPUStats:
        try:
            import torch
            if not torch.cuda.is_available():
                return GPUStats()
            name = torch.cuda.get_device_name(0)
            vram_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
            vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            return GPUStats(
                name=name,
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
            )
        except Exception:
            return GPUStats()

    def shutdown(self):
        if self._use_pynvml:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
