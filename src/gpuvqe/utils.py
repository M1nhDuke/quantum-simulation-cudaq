"""Timing, GPU memory (pynvml) and CSV helpers."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cudaq
import pynvml

GIB = 2**30


def set_backend(name: str) -> None:
    """Select a simulator by name.

    ``nvidia-fp64`` is deprecated as a target name in CUDA-Q 0.17; it maps to the
    replacement ``nvidia`` target with ``option="fp64"`` (same cuStateVec fp64 simulator).
    """
    if name == "nvidia-fp64":
        cudaq.set_target("nvidia", option="fp64")
    else:
        cudaq.set_target(name)


def time_runs(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> list[float]:
    """Call ``fn`` ``warmup`` times untimed, then return ``repeats`` wall times (s)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


class GpuMemPoller:
    """Poll device-wide used GPU memory in a background thread.

    ``peak_gib`` is the peak *increase* over the usage seen on entry, i.e. what the
    workload allocated. Device-wide polling is used because WSL2 exposes no
    per-process NVML accounting; the baseline removes the Windows host's own usage.
    """

    def __init__(self, interval_s: float = 0.005, device: int = 0) -> None:
        self.interval_s = interval_s
        self.device = device

    def __enter__(self) -> "GpuMemPoller":
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        self.baseline = self._used()
        self._peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _used(self) -> int:
        return pynvml.nvmlDeviceGetMemoryInfo(self._handle).used

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._used())
            self._stop.wait(self.interval_s)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()

    @property
    def peak_gib(self) -> float:
        return (self._peak - self.baseline) / GIB


def gpu_info(device: int = 0) -> dict[str, str]:
    """GPU name, total memory (GiB) and driver version, straight from NVML."""
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        return {
            "gpu": pynvml.nvmlDeviceGetName(h),
            "vram_gib": f"{pynvml.nvmlDeviceGetMemoryInfo(h).total / GIB:.2f}",
            "driver": pynvml.nvmlSystemGetDriverVersion(),
        }
    finally:
        pynvml.nvmlShutdown()


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    """Write ``rows`` to ``path`` (overwriting), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
