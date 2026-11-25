"""Hartree-Fock + UCCSD ansatz, energy function and the classical optimisation loop.

The caller picks the simulator with ``cudaq.set_target`` (``nvidia-fp64`` for VQE).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cudaq
import numpy as np
from cudaq.kernels import uccsd
from scipy.optimize import minimize

from .hamiltonians import Molecule


@cudaq.kernel
def uccsd_ansatz(thetas: list[float], n_qubits: int, n_electrons: int):
    """HF reference (X on the first ``n_electrons`` spin-orbitals) followed by UCCSD."""
    q = cudaq.qvector(n_qubits)
    for i in range(n_electrons):
        x(q[i])
    uccsd(q, thetas, n_electrons, n_qubits)


def num_params(mol: Molecule) -> int:
    """Number of UCCSD amplitudes, from CUDA-Q's own helper."""
    return cudaq.kernels.uccsd_num_parameters(mol.n_electrons, mol.n_qubits)


def energy(thetas: np.ndarray | list[float], mol: Molecule) -> float:
    """<psi(thetas)|H|psi(thetas)> on the current target (exact, no shots)."""
    return cudaq.observe(
        uccsd_ansatz, mol.hamiltonian, [float(t) for t in thetas], mol.n_qubits, mol.n_electrons
    ).expectation()


@dataclass
class VQEResult:
    energy: float
    thetas: np.ndarray
    n_iters: int
    n_evals: int
    method: str
    wall_s: float


def run_vqe(
    mol: Molecule, x0: np.ndarray | None = None, tol: float = 1e-8, fd_step: float = 1e-5
) -> VQEResult:
    """Minimise the UCCSD energy.

    L-BFGS-B with central finite-difference gradients (UCCSD excitations are not
    single Pauli rotations, so the two-term parameter-shift rule is not exact here).
    If L-BFGS-B reports failure, COBYLA continues from its point and the lower energy
    of the two is kept.
    """
    t0 = time.perf_counter()
    n_evals = 0

    def f(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        return energy(x, mol)

    def grad(x: np.ndarray) -> np.ndarray:
        g = np.empty_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = fd_step
            g[i] = (f(x + e) - f(x - e)) / (2 * fd_step)
        return g

    x0 = np.zeros(num_params(mol)) if x0 is None else np.asarray(x0, dtype=float)
    res = minimize(
        f, x0, jac=grad, method="L-BFGS-B",
        options={"gtol": tol, "ftol": 1e-12, "maxiter": 2000},
    )
    best, method, n_iters = res, "L-BFGS-B", int(res.nit)
    if not res.success:
        res2 = minimize(f, res.x, method="COBYLA", tol=tol, options={"maxiter": 20000})
        n_iters += int(res2.nfev)  # COBYLA has no iteration count; one eval per step
        if res2.fun < res.fun:
            best, method = res2, "L-BFGS-B+COBYLA"
    return VQEResult(
        float(best.fun), np.asarray(best.x), n_iters, n_evals, method, time.perf_counter() - t0
    )
