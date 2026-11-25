"""Benchmark circuit kernels: GHZ and a hardware-efficient layered circuit."""
from __future__ import annotations

import cudaq
import numpy as np


@cudaq.kernel
def ghz(n: int):
    """GHZ state (|0..0> + |1..1>)/sqrt(2) on ``n`` qubits."""
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def hardware_efficient(n: int, depth: int, angles: list[float]):
    """``depth`` layers of RY+RZ on every qubit followed by a CNOT ladder.

    ``angles`` holds ``2 * n * depth`` values, ordered (layer, qubit, [ry, rz]).
    """
    q = cudaq.qvector(n)
    for layer in range(depth):
        for i in range(n):
            ry(angles[2 * (layer * n + i)], q[i])
            rz(angles[2 * (layer * n + i) + 1], q[i])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])


def random_angles(n: int, depth: int, seed: int = 42) -> list[float]:
    """Fixed pseudo-random angles in [0, 2*pi) for :func:`hardware_efficient`."""
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 2 * n * depth).tolist()


def kernel_and_args(circuit: str, n: int, depth: int = 10, seed: int = 42) -> tuple:
    """Return ``(kernel, args)`` for a circuit name (``"hwe"`` or ``"ghz"``)."""
    if circuit == "hwe":
        return hardware_efficient, (n, depth, random_angles(n, depth, seed))
    if circuit == "ghz":
        return ghz, (n,)
    raise ValueError(f"unknown circuit {circuit!r}")
