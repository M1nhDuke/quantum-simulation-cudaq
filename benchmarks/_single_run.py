"""Time one (backend, circuit, n) point; prints one JSON line. Called by run_scaling.py.

Each run is ``cudaq.observe`` of Z on qubit 0, so the statevector never leaves the
device. ``signal.alarm`` with the default SIGALRM action kills this process if a
single run exceeds ``--timeout`` (it fires even inside the C++ simulator call).
"""
from __future__ import annotations

import argparse
import json
import signal
import warnings

import cudaq

from gpuvqe.circuits import kernel_and_args
from gpuvqe.utils import set_backend, time_runs

warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True)
    p.add_argument("--circuit", default="hwe", choices=["hwe", "ghz"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300, help="seconds per single run")
    a = p.parse_args()

    set_backend(a.backend)
    kernel, args = kernel_and_args(a.circuit, a.n, a.depth, a.seed)
    z0 = cudaq.spin.z(0)
    values: list[float] = []

    def run() -> None:
        signal.alarm(a.timeout)
        values.append(cudaq.observe(kernel, z0, *args).expectation())
        signal.alarm(0)

    times = time_runs(run, repeats=a.repeats, warmup=1)
    print(json.dumps({"times": times, "expval": values[-1]}))


if __name__ == "__main__":
    main()
