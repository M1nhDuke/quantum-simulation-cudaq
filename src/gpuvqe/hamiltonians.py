"""Molecule -> qubit Hamiltonian (``cudaq.SpinOperator``) with PySCF reference energies.

Pipeline: PySCF (via ``openfermionpyscf``) -> second-quantised Hamiltonian ->
Jordan-Wigner -> ``cudaq.SpinOperator``. OpenFermion orders spin-orbitals
interleaved (even = alpha, odd = beta), which is the same convention as
``cudaq.kernels.uccsd``, so the Hartree-Fock state is X on qubits 0..n_e-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import cudaq
from openfermion import MolecularData, get_fermion_operator, jordan_wigner
from openfermion.ops import QubitOperator
from openfermionpyscf import run_pyscf
from pyscf import mcscf

Geometry = list[tuple[str, tuple[float, float, float]]]


@dataclass
class Molecule:
    """Qubit Hamiltonian plus the sizes and reference energies needed for VQE."""

    hamiltonian: cudaq.SpinOperator
    n_qubits: int
    n_electrons: int
    e_hf: float
    e_ref: float  # FCI for the full space, CASCI when an active space is used


def to_spin_operator(qop: QubitOperator) -> cudaq.SpinOperator:
    """Convert an OpenFermion ``QubitOperator`` (real coefficients) to ``cudaq.SpinOperator``."""
    paulis = {"X": cudaq.spin.x, "Y": cudaq.spin.y, "Z": cudaq.spin.z}
    h = cudaq.SpinOperator.empty()
    for term, coef in qop.terms.items():
        assert abs(coef.imag) < 1e-10, f"non-Hermitian term {term}: {coef}"
        op = cudaq.spin.identity()
        for qubit, pauli in term:
            op *= paulis[pauli](qubit)
        h += float(coef.real) * op
    return h


def molecule(
    geometry: Geometry,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    ncas: int | None = None,
    nelecas: int | None = None,
) -> Molecule:
    """Build the JW qubit Hamiltonian of a molecule.

    With ``ncas``/``nelecas`` the lowest doubly-occupied orbitals are frozen and only
    ``ncas`` spatial orbitals are kept; the reference is then PySCF CASCI on the same
    RHF orbitals. Otherwise the full space is used and the reference is FCI.
    """
    data = run_pyscf(
        MolecularData(geometry, basis, multiplicity, charge),
        run_scf=True,
        run_fci=ncas is None,
    )
    if ncas is None:
        occupied, active = None, None
        n_qubits, n_electrons, e_ref = data.n_qubits, data.n_electrons, data.fci_energy
    else:
        assert nelecas is not None, "nelecas is required with ncas"
        ncore = (data.n_electrons - nelecas) // 2
        occupied, active = list(range(ncore)), list(range(ncore, ncore + ncas))
        n_qubits, n_electrons = 2 * ncas, nelecas
        casci = mcscf.CASCI(data._pyscf_data["scf"], ncas, nelecas)
        casci.verbose = 0
        e_ref = float(casci.kernel()[0])
    fermion_h = get_fermion_operator(
        data.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    )
    return Molecule(
        to_spin_operator(jordan_wigner(fermion_h)), n_qubits, n_electrons, data.hf_energy, e_ref
    )


def h2(bond_length: float, basis: str = "sto-3g") -> Molecule:
    """H2 at ``bond_length`` angstrom: 4 qubits, 2 electrons in STO-3G."""
    return molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))], basis)
