"""GPU vs CPU statevector scaling sweep.

Every (backend, circuit, n) point runs in its own subprocess (``_single_run.py``), so an
out-of-memory crash or a timeout at one size is logged and the sweep continues. Peak GPU
memory is polled with pynvml from this (parent) process while the child runs.

Writes ``results/scaling.csv`` and ``results/correctness.csv`` (n <= 20 agreement check).
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import statistics
import subprocess
import sys
from pathlib import Path

from gpuvqe.utils import GpuMemPoller, write_csv

ROOT = Path(__file__).resolve().parents[1]
MAX_QUBITS = {"nvidia": 29, "nvidia-fp64": 28, "qpp-cpu": 26}
FIELDS = ["backend", "circuit", "n_qubits", "median_s", "min_s", "peak_mem_gb", "status"]
CHECK_MAX_N, CHECK_TOL = 20, 1e-4


def measure(backend: str, circuit: str, n: int, repeats: int, timeout: int) -> tuple[dict, float | None]:
    """Run one point in a subprocess; return its CSV row and <Z0> (None on failure)."""
    cmd = [sys.executable, str(ROOT / "benchmarks" / "_single_run.py"), "--backend", backend,
           "--circuit", circuit, "--n", str(n), "--repeats", str(repeats), "--timeout", str(timeout)]
    row: dict = dict.fromkeys(FIELDS, "") | {"backend": backend, "circuit": circuit, "n_qubits": n}
    expval = None
    with GpuMemPoller() as mem:
        try:
            # backstop only; the child enforces the per-run limit itself via SIGALRM
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout * (repeats + 1) + 120)
        except subprocess.TimeoutExpired:
            proc = None
    if proc is None or proc.returncode == -signal.SIGALRM:
        row["status"] = "TIMEOUT"
    elif proc.returncode != 0:
        oom = proc.returncode == -signal.SIGKILL or re.search(
            r"out of memory|bad_alloc|alloc.*fail|insufficient memory", proc.stderr, re.I)
        row["status"] = "OOM" if oom else "ERROR"
        print(proc.stderr[-1500:], file=sys.stderr)
    else:
        out = json.loads(next(l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")))
        row |= {"median_s": statistics.median(out["times"]), "min_s": min(out["times"]), "status": "OK"}
        expval = out["expval"]
    if backend != "qpp-cpu":  # GPU memory is meaningless for the CPU backend
        row["peak_mem_gb"] = round(mem.peak_gib, 3)
    return row, expval


def correctness_rows(expvals: dict, backends: list[str]) -> list[dict]:
    """Per (circuit, n): <Z0> from every backend and whether they agree to CHECK_TOL."""
    rows = []
    for circuit, n in sorted({(c, n) for c, n, _ in expvals}):
        vals = {b: expvals.get((circuit, n, b)) for b in backends}
        if None in vals.values():
            continue
        diff = max(vals.values()) - min(vals.values())
        rows.append({"circuit": circuit, "n_qubits": n, **vals,
                     "max_abs_diff": diff, "agree": diff < CHECK_TOL})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends", default="nvidia,nvidia-fp64,qpp-cpu")
    p.add_argument("--circuits", default="hwe,ghz")
    p.add_argument("--min-qubits", type=int, default=10)
    p.add_argument("--max-qubits", type=int, default=99, help="cap on each backend's default max")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300, help="seconds per single run")
    p.add_argument("--out", default=str(ROOT / "results" / "scaling.csv"))
    a = p.parse_args()
    backends, circuits = a.backends.split(","), a.circuits.split(",")

    rows, expvals = [], {}
    for backend in backends:
        for circuit in circuits:
            for n in range(a.min_qubits, min(MAX_QUBITS[backend], a.max_qubits) + 1):
                row, ev = measure(backend, circuit, n, a.repeats, a.timeout)
                print(json.dumps(row), flush=True)
                rows.append(row)
                write_csv(a.out, rows, FIELDS)  # rewrite after every point: partial runs keep data
                if ev is not None and n <= CHECK_MAX_N:
                    expvals[(circuit, n, backend)] = ev

    check = correctness_rows(expvals, backends)
    write_csv(Path(a.out).with_name("correctness.csv"), check,
              ["circuit", "n_qubits", *backends, "max_abs_diff", "agree"])
    print(f"correctness (n<={CHECK_MAX_N}, tol {CHECK_TOL}): "
          f"{sum(r['agree'] for r in check)}/{len(check)} points agree")


if __name__ == "__main__":
    main()
