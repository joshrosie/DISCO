"""Micro-benchmark: where does graph construction time go?

Breaks down the per-structure cost into:
  - cellpar_to_cell
  - ASE Atoms construction
  - find_points_in_spheres (neighbor search)
  - compute_threebody_indices
  - PyG Data construction
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase import Atoms
from ase.geometry import cellpar_to_cell
from mattersim.datasets.utils.convertor import (
    GraphConvertor,
    compute_threebody_indices,
    get_fixed_radius_bonding,
)


def main():
    # Generate realistic-ish test structures (random cells + positions)
    rng = np.random.default_rng(42)
    N_STRUCTS = 1024
    structures = []
    for _ in range(N_STRUCTS):
        n_atoms = rng.integers(2, 21)  # MP20 range
        lengths = rng.uniform(3.0, 12.0, size=3)
        angles = rng.uniform(60.0, 120.0, size=3)
        cell = cellpar_to_cell(np.concatenate([lengths, angles]))
        frac = rng.uniform(0, 1, size=(n_atoms, 3))
        numbers = rng.integers(1, 95, size=n_atoms)
        structures.append((numbers, frac, cell, lengths, angles))

    convertor = GraphConvertor(
        model_type="m3gnet", twobody_cutoff=5.0,
        has_threebody=True, threebody_cutoff=4.0,
    )

    # --- Full convertor.convert baseline ---
    t0 = time.perf_counter()
    for numbers, frac, cell, _, _ in structures:
        atoms = Atoms(numbers=numbers, positions=frac @ cell, cell=cell, pbc=True)
        convertor.convert(atoms)
    t_full = time.perf_counter() - t0

    # --- Break down components ---

    # 1. cellpar_to_cell
    t0 = time.perf_counter()
    for _, _, _, lengths, angles in structures:
        cellpar_to_cell(np.concatenate([lengths, angles]))
    t_cellpar = time.perf_counter() - t0

    # 2. ASE Atoms construction
    t0 = time.perf_counter()
    atoms_list = []
    for numbers, frac, cell, _, _ in structures:
        atoms = Atoms(numbers=numbers, positions=frac @ cell, cell=cell, pbc=True)
        atoms_list.append(atoms)
    t_atoms = time.perf_counter() - t0

    # 3. find_points_in_spheres (neighbor search)
    # First wrap positions like convertor.convert does
    for atoms in atoms_list:
        sp = atoms.get_scaled_positions()
        atoms.set_scaled_positions(np.mod(sp, 1))

    t0 = time.perf_counter()
    neighbor_results = []
    for atoms in atoms_list:
        result = get_fixed_radius_bonding(atoms, cutoff=5.0, pbc=True)
        neighbor_results.append(result)
    t_neighbor = time.perf_counter() - t0

    # 4. compute_threebody_indices
    t0 = time.perf_counter()
    for i, atoms in enumerate(atoms_list):
        sent_idx, recv_idx, _, distances = neighbor_results[i]
        edge_index = np.array([sent_idx, recv_idx])
        if edge_index.shape[1] > 0:
            compute_threebody_indices(
                bond_atom_indices=edge_index.T,
                bond_length=distances,
                n_atoms=len(atoms),
                atomic_number=atoms.get_atomic_numbers(),
                threebody_cutoff=4.0,
            )
    t_threebody = time.perf_counter() - t0

    # 5. vectorized cellpar_to_cell (all at once)
    all_cellpars = np.array([np.concatenate([l, a]) for _, _, _, l, a in structures])
    t0 = time.perf_counter()
    for cp in all_cellpars:
        cellpar_to_cell(cp)
    t_cellpar_loop = time.perf_counter() - t0

    # Vectorized version
    t0 = time.perf_counter()
    _cellpar_to_cell_batch(all_cellpars)
    t_cellpar_vec = time.perf_counter() - t0

    print(f"{'='*60}")
    print(f"  Graph Construction Breakdown ({N_STRUCTS} structures)")
    print(f"{'='*60}")
    print(f"  Full convertor.convert:     {t_full:.3f}s  ({t_full/N_STRUCTS*1000:.2f}ms/struct)")
    print()
    print(f"  cellpar_to_cell (loop):     {t_cellpar:.3f}s  ({t_cellpar/t_full*100:.1f}%)")
    print(f"  cellpar_to_cell (vectorized):{t_cellpar_vec:.3f}s")
    print(f"  ASE Atoms construction:     {t_atoms:.3f}s  ({t_atoms/t_full*100:.1f}%)")
    print(f"  find_points_in_spheres:     {t_neighbor:.3f}s  ({t_neighbor/t_full*100:.1f}%)")
    print(f"  compute_threebody_indices:  {t_threebody:.3f}s  ({t_threebody/t_full*100:.1f}%)")
    accounted = t_cellpar + t_atoms + t_neighbor + t_threebody
    print(f"  PyG Data + overhead:        {t_full - accounted:.3f}s  ({(t_full-accounted)/t_full*100:.1f}%)")
    print()


def _cellpar_to_cell_batch(cellpars: np.ndarray) -> np.ndarray:
    """Vectorized cellpar_to_cell for (N, 6) array. Returns (N, 3, 3)."""
    a, b, c = cellpars[:, 0], cellpars[:, 1], cellpars[:, 2]
    alpha = np.radians(cellpars[:, 3])
    beta = np.radians(cellpars[:, 4])
    gamma = np.radians(cellpars[:, 5])

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    N = len(cellpars)
    cells = np.zeros((N, 3, 3))
    cells[:, 0, 0] = a
    cells[:, 1, 0] = b * cos_gamma
    cells[:, 1, 1] = b * sin_gamma
    cells[:, 2, 0] = c * cos_beta
    cells[:, 2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cells[:, 2, 2] = np.sqrt(
        np.maximum(c**2 - cells[:, 2, 0]**2 - cells[:, 2, 1]**2, 0.0)
    )
    return cells


if __name__ == "__main__":
    main()
