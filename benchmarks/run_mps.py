"""Bonus: GHZ on the ``tensornet-mps`` target at 30/35/40 qubits.

A 40-qubit statevector would need 8 TiB, but GHZ has bond dimension 2, so a matrix
product state represents it exactly in a few kB. Checks that only |0..0> and |1..1>
are sampled. Skips (status SKIPPED) if the target is unavailable. Writes ``results/mps.csv``.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import cudaq

from gpuvqe.circuits import ghz
from gpuvqe.utils import set_backend, write_csv

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["n_qubits", "shots", "n_all0", "n_all1", "n_other", "wall_s", "status"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qubits", default="30,35,40")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--out", default=str(ROOT / "results" / "mps.csv"))
    a = p.parse_args()
    sizes = [int(n) for n in a.qubits.split(",")]

    if not cudaq.has_target("tensornet-mps"):
        print("tensornet-mps target not available; skipping")
        write_csv(a.out, [{"n_qubits": n, "shots": a.shots, "status": "SKIPPED"} for n in sizes], FIELDS)
        return
    set_backend("tensornet-mps")

    rows = []
    for n in sizes:
        row: dict = {"n_qubits": n, "shots": a.shots}
        try:
            t0 = time.perf_counter()
            counts = dict(cudaq.sample(ghz, n, shots_count=a.shots).items())
            row["wall_s"] = time.perf_counter() - t0
            row["n_all0"], row["n_all1"] = counts.get("0" * n, 0), counts.get("1" * n, 0)
            row["n_other"] = sum(counts.values()) - row["n_all0"] - row["n_all1"]
            ok = row["n_other"] == 0 and row["n_all0"] > 0 and row["n_all1"] > 0
            row["status"] = "OK" if ok else "FAIL"
        except Exception as e:  # report and continue with the next size
            row["status"] = f"ERROR: {e}"[:120]
        print(row, flush=True)
        rows.append(row)
    write_csv(a.out, rows, FIELDS)


if __name__ == "__main__":
    main()
