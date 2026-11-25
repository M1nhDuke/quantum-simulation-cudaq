"""H2 / STO-3G potential-energy scan with UCCSD-VQE on the nvidia-fp64 target.

First checks VQE against FCI at 0.7474 A (must agree to 1e-5 Ha, else exit 1). Each scan
point is optimised from (a) the previous point's optimum (warm start) and (b) zeros, and
the lower energy is kept. Writes ``results/h2_pes.csv``.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from gpuvqe.hamiltonians import h2
from gpuvqe.utils import set_backend, write_csv
from gpuvqe.vqe import run_vqe

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["R", "E_HF", "E_FCI", "E_VQE", "abs_err_Ha", "abs_err_mHa", "n_iters", "wall_s"]
CHEM_ACC_MHA = 1.6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=int, default=28)
    p.add_argument("--out", default=str(ROOT / "results" / "h2_pes.csv"))
    a = p.parse_args()
    set_backend("nvidia-fp64")

    ref = h2(0.7474)
    check = run_vqe(ref)
    err = abs(check.energy - ref.e_ref)
    print(f"sanity R=0.7474: E_VQE={check.energy:.10f} E_FCI={ref.e_ref:.10f} err={err:.2e} Ha")
    if err > 1e-5:
        sys.exit("VQE does not reproduce FCI at 0.7474 A; check orbital ordering / HF state")

    rows, prev = [], None
    for r in np.linspace(0.30, 3.00, a.points):
        t0 = time.perf_counter()
        mol = h2(float(r))
        starts = [None] if prev is None else [prev, None]  # None -> zero init
        best = min((run_vqe(mol, x0) for x0 in starts), key=lambda res: res.energy)
        prev = best.thetas
        err = abs(best.energy - mol.e_ref)
        rows.append({"R": round(float(r), 6), "E_HF": mol.e_hf, "E_FCI": mol.e_ref,
                     "E_VQE": best.energy, "abs_err_Ha": err, "abs_err_mHa": err * 1e3,
                     "n_iters": best.n_iters, "wall_s": time.perf_counter() - t0})
        print(f"R={r:.4f}  E_VQE={best.energy:.8f}  E_FCI={mol.e_ref:.8f}  "
              f"err={err * 1e3:.2e} mHa  iters={best.n_iters}", flush=True)
    write_csv(a.out, rows, FIELDS)

    errs = np.array([row["abs_err_mHa"] for row in rows])
    print(f"max err {errs.max():.3e} mHa, mean err {errs.mean():.3e} mHa, "
          f"all < {CHEM_ACC_MHA} mHa: {bool((errs < CHEM_ACC_MHA).all())}")


if __name__ == "__main__":
    main()
