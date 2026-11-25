"""Acceptance checks: backend agreement, H2 Hamiltonian/ansatz shape, VQE vs FCI."""
from __future__ import annotations

import cudaq
import pytest

from gpuvqe.circuits import kernel_and_args
from gpuvqe.hamiltonians import h2
from gpuvqe.utils import set_backend
from gpuvqe.vqe import energy, num_params, run_vqe


@pytest.fixture(scope="module")
def h2_eq():
    return h2(0.7474)


@pytest.mark.parametrize("circuit", ["hwe", "ghz"])
@pytest.mark.parametrize("n", [10, 16])
def test_gpu_cpu_expectation_agree(circuit: str, n: int) -> None:
    kernel, args = kernel_and_args(circuit, n)
    values = []
    for backend in ("nvidia", "nvidia-fp64", "qpp-cpu"):
        set_backend(backend)
        values.append(cudaq.observe(kernel, cudaq.spin.z(0), *args).expectation())
    assert max(values) - min(values) < 1e-4, values


def test_h2_hamiltonian_has_4_qubits(h2_eq) -> None:
    assert h2_eq.n_qubits == 4
    assert h2_eq.hamiltonian.qubit_count == 4
    assert h2_eq.n_electrons == 2


def test_uccsd_param_count_matches_helper(h2_eq) -> None:
    assert num_params(h2_eq) == cudaq.kernels.uccsd_num_parameters(2, 4) == 3  # 2 singles + 1 double
    # the kernel accepts exactly that many amplitudes; all-zero amplitudes == HF state
    set_backend("nvidia-fp64")
    assert energy([0.0] * num_params(h2_eq), h2_eq) == pytest.approx(h2_eq.e_hf, abs=1e-10)


def test_h2_vqe_matches_fci(h2_eq) -> None:
    set_backend("nvidia-fp64")
    assert abs(run_vqe(h2_eq).energy - h2_eq.e_ref) < 1e-5
    assert h2_eq.e_ref == pytest.approx(-1.1372, abs=1e-4)
