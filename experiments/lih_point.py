"""Bonus: LiH / STO-3G at 1.6 A, UCCSD-VQE in a small active space vs PySCF CASCI.

Default active space CAS(4e, 4o) -> 8 qubits, on the nvidia-fp64 target. Gives up
gracefully (status TIMEOUT) after ``--time-limit`` seconds. Writes ``results/lih_point.csv``.
"""
from __future__ import annotations

import argparse
import signal
import time
import warnings
from pathlib import Path

from gpuvqe.hamiltonians import molecule
from gpuvqe.utils import set_backend, write_csv
from gpuvqe.vqe import num_params, run_vqe

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["R", "ncas", "nelecas", "n_qubits", "n_params", "E_HF", "E_CASCI", "E_VQE",
          "abs_err_mHa", "n_iters", "wall_s", "status"]


def _timeout(*_: object) -> None:
    raise TimeoutError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--R", type=float, default=1.6)
    p.add_argument("--ncas", type=int, default=4)
    p.add_argument("--nelecas", type=int, default=4)
    p.add_argument("--time-limit", type=int, default=900)
    p.add_argument("--out", default=str(ROOT / "results" / "lih_point.csv"))
    a = p.parse_args()
    set_backend("nvidia-fp64")

    row: dict = {"R": a.R, "ncas": a.ncas, "nelecas": a.nelecas}
    t0 = time.perf_counter()
    signal.signal(signal.SIGALRM, _timeout)  # fires between energy evaluations
    signal.alarm(a.time_limit)
    try:
        mol = molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, a.R))],
                       ncas=a.ncas, nelecas=a.nelecas)
        row |= {"n_qubits": mol.n_qubits, "n_params": num_params(mol),
                "E_HF": mol.e_hf, "E_CASCI": mol.e_ref}
        res = run_vqe(mol)
        row |= {"E_VQE": res.energy, "abs_err_mHa": abs(res.energy - mol.e_ref) * 1e3,
                "n_iters": res.n_iters, "status": "OK"}
    except TimeoutError:
        row["status"] = f"TIMEOUT (>{a.time_limit}s, skipped)"
    finally:
        signal.alarm(0)
    row["wall_s"] = time.perf_counter() - t0
    print(row)
    write_csv(a.out, [row], FIELDS)


if __name__ == "__main__":
    main()
