from __future__ import annotations

import csv as _csv
import json
import math
from pathlib import Path

import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from src.data.mp20_tokens import MP20Tokens, collate_mp20_tokens
from src.data.synthetic_augmentation import (
    AugmentedCrystalDataset,
    _validate_structure,
    build_augmented_train_dataset,
    make_synthetic_dataset,
)


def _make_synthetic_root_with_cifs(
    root: Path, entries: list[tuple[str, Structure]]
) -> None:
    """Create a minimal MP20Tokens-compatible synthetic root under ``root``.

    Writes ``raw/train.csv`` with one row per (material_id, structure) pair.
    """
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=[
                "material_id",
                "cif",
                "formation_energy_per_atom",
                "e_above_hull",
            ],
        )
        writer.writeheader()
        for sample_id, struct in entries:
            writer.writerow(
                {
                    "material_id": sample_id,
                    "cif": str(CifWriter(struct)),
                    "formation_energy_per_atom": "",
                    "e_above_hull": "",
                }
            )


class FakeOracle:
    def __init__(self, results: list[dict]):
        self.results = list(results)
        self.calls = 0

    def call_many(self, structs: list[Structure]) -> list[dict]:
        out = []
        for _ in structs:
            out.append(self.results[self.calls])
            self.calls += 1
        return out


def _item(idx: int, nmax: int = 4, num_atoms: int = 2) -> dict:
    A0 = torch.zeros(nmax, dtype=torch.long)
    A0[:num_atoms] = torch.tensor([3, 8][:num_atoms], dtype=torch.long)
    F1 = torch.zeros(nmax, 3, dtype=torch.float32)
    F1[:num_atoms] = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]][:num_atoms])
    pad_mask = torch.ones(nmax, dtype=torch.bool)
    pad_mask[:num_atoms] = False
    return {
        "mp_id": f"real_{idx}",
        "A0": A0,
        "F1": F1,
        "Y1": torch.tensor([1.2, 1.2, 1.2, 0.0, 0.0, 0.0], dtype=torch.float32),
        "pad_mask": pad_mask,
        "num_atoms": num_atoms,
    }


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.items = [_item(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def test_augmented_dataset_lengths_and_collation(tmp_path: Path):
    real = TinyDataset(3)
    # AugmentedCrystalDataset is agnostic to the synthetic dataset class — any
    # Dataset with __len__ + __getitem__ works. We use TinyDataset here; the
    # filesystem-backed MP20Tokens path is covered by the
    # build_augmented_train_dataset tests below.
    synthetic = TinyDataset(2)

    concat = AugmentedCrystalDataset(real, synthetic, augmentation_mode="synthetic_concat")
    assert len(concat) == 5
    assert concat[3]["mp_id"] == "real_0"

    oversampled = AugmentedCrystalDataset(
        real,
        augmentation_mode="oversample_real",
        num_extra_samples=4,
        seed=7,
    )
    assert len(oversampled) == 7
    batch = collate_mp20_tokens([oversampled[0], oversampled[3], concat[4]])
    assert batch["A0"].shape == (3, 4)
    assert batch["F1"].shape == (3, 4, 3)


def _li2o_structure(a: float = 4.0) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_build_augmented_train_dataset_reports_composition(tmp_path: Path):
    real = TinyDataset(2)
    synthetic_path = tmp_path / "synthetic"
    _make_synthetic_root_with_cifs(
        synthetic_path,
        [
            ("syn_0", _li2o_structure(4.0)),
            ("syn_1", _li2o_structure(4.1)),
            ("syn_2", _li2o_structure(4.2)),
        ],
    )
    (synthetic_path / "metadata.jsonl").write_text("{}", encoding="utf-8")

    ds, comp = build_augmented_train_dataset(
        real,
        augmentation_mode="synthetic_concat",
        synthetic_data=synthetic_path,
        seed=0,
        nmax=4,
    )
    assert len(ds) == 5
    assert comp["real_train_count"] == 2
    assert comp["synthetic_train_count"] == 3
    assert comp["effective_train_count"] == 5
    assert comp["synthetic_metadata_path"].endswith("metadata.jsonl")
    assert comp["synthetic_metadata_paths"] == [comp["synthetic_metadata_path"]]


def test_build_augmented_train_dataset_accepts_multiple_synthetic_roots(tmp_path: Path):
    real = TinyDataset(2)
    roots = []
    for name, n in [("s0", 2), ("s1", 3)]:
        synthetic_path = tmp_path / name
        _make_synthetic_root_with_cifs(
            synthetic_path,
            [(f"{name}_{i}", _li2o_structure(4.0 + 0.05 * i)) for i in range(n)],
        )
        (synthetic_path / "metadata.jsonl").write_text("{}", encoding="utf-8")
        roots.append(synthetic_path)

    ds, comp = build_augmented_train_dataset(
        real,
        augmentation_mode="synthetic_concat",
        synthetic_data=roots,
        seed=0,
        nmax=4,
    )
    assert len(ds) == 7
    assert comp["real_train_count"] == 2
    assert comp["synthetic_train_count"] == 5
    assert comp["synthetic_dataset_count"] == 2
    assert comp["effective_train_count"] == 7
    assert len(comp["synthetic_metadata_paths"]) == 2


def test_make_synthetic_dataset_writes_metadata_summary_and_samples(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    struct_a = Structure(
        Lattice.cubic(4.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    struct_b = Structure(
        Lattice.cubic(4.1),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    (input_dir / "a.cif").write_text(str(CifWriter(struct_a)), encoding="utf-8")
    (input_dir / "b.cif").write_text(str(CifWriter(struct_b)), encoding="utf-8")

    real_root = tmp_path / "mp20"
    (real_root / "raw").mkdir(parents=True)
    (real_root / "raw" / "train.csv").write_text(
        "pretty_formula\nLi2O2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "synthetic_out"

    summary = make_synthetic_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        real_train_path=real_root,
        dedup_mode="formula",
        filter_level="valid",
        nmax=4,
    )
    assert summary["num_input"] == 2
    assert summary["num_kept"] == 1
    assert summary["num_dedup_against_train"] == 1
    assert (output_dir / "metadata.jsonl").exists()
    assert (output_dir / "rejected.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "raw" / "train.csv").exists()

    loaded_summary = json.loads((output_dir / "summary.json").read_text())
    assert loaded_summary["num_kept"] == 1
    ds = MP20Tokens(root=str(output_dir), split="train", nmax=4, augment_translate=False)
    assert len(ds) == 1
    assert collate_mp20_tokens([ds[0]])["num_atoms"].tolist() == [2]


def test_validate_structure_rejects_exact_overlapping_sites():
    struct = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    ok, reason = _validate_structure(struct, nmax=4, min_distance=0.5)

    assert not ok
    assert reason == "min_distance_too_small"


def test_make_synthetic_dataset_canonicalizes_reference_metadata_formulas(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    struct = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    (input_dir / "nacl.cif").write_text(str(CifWriter(struct)), encoding="utf-8")

    ref_root = tmp_path / "previous_round"
    ref_root.mkdir()
    (ref_root / "metadata.jsonl").write_text(
        json.dumps({"formula": "Cl1 Na1"}) + "\n",
        encoding="utf-8",
    )

    summary = make_synthetic_dataset(
        input_dir=input_dir,
        output_dir=tmp_path / "synthetic_out",
        reference_data_root=ref_root,
        dedup_mode="formula",
        filter_level="valid",
        nmax=4,
    )

    assert summary["num_kept"] == 0
    assert summary["num_dedup_against_train"] == 1


def test_make_synthetic_dataset_relaxed_filter_requires_oracle(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="requires row_oracle"):
        make_synthetic_dataset(
            input_dir=input_dir,
            output_dir=tmp_path / "synthetic_out",
            dedup_mode="none",
            filter_level="msun_like",
            nmax=4,
        )


def test_make_synthetic_dataset_dedups_against_multiple_reference_roots(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    struct_mp = Structure(
        Lattice.cubic(4.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    struct_s0 = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    struct_new = Structure(
        Lattice.cubic(4.0),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    for name, struct in [("mp.cif", struct_mp), ("s0.cif", struct_s0), ("new.cif", struct_new)]:
        (input_dir / name).write_text(str(CifWriter(struct)), encoding="utf-8")

    mp_root = tmp_path / "mp20"
    (mp_root / "raw").mkdir(parents=True)
    (mp_root / "raw" / "train.csv").write_text(
        "pretty_formula\nLi2O2\n",
        encoding="utf-8",
    )
    s0_root = tmp_path / "s0"
    s0_root.mkdir()
    (s0_root / "metadata.jsonl").write_text(
        json.dumps({"formula": "NaCl"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "synthetic_out"

    summary = make_synthetic_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        reference_data_root=[mp_root, s0_root],
        dedup_mode="formula",
        filter_level="valid",
        nmax=4,
    )
    assert summary["num_input"] == 3
    assert summary["num_kept"] == 1
    assert summary["num_dedup_against_train"] == 2
    assert summary["reference_data_root"] == [str(mp_root), str(s0_root)]


def test_make_synthetic_dataset_msun_like_uses_relaxed_structures(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    raw_keep = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    raw_drop = Structure(
        Lattice.cubic(4.0),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    relaxed_keep = Structure(
        Lattice.cubic(5.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    relaxed_drop = Structure(
        Lattice.cubic(5.0),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    (input_dir / "a.cif").write_text(str(CifWriter(raw_keep)), encoding="utf-8")
    (input_dir / "b.cif").write_text(str(CifWriter(raw_drop)), encoding="utf-8")

    oracle = FakeOracle(
        [
            {
                "final_structure": relaxed_keep,
                "e_total": -10.0,
                "e_form": -1.5,
                "e_above_hull": 0.05,
                "nsteps": 12,
                "err": "",
            },
            {
                "final_structure": relaxed_drop,
                "e_total": -9.0,
                "e_form": -1.0,
                "e_above_hull": 0.5,
                "nsteps": 10,
                "err": "",
            },
        ]
    )
    output_dir = tmp_path / "synthetic_out"
    summary = make_synthetic_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        dedup_mode="none",
        filter_level="msun_like",
        nmax=4,
        row_oracle=oracle,
    )

    assert summary["num_relax_attempted"] == 2
    assert summary["num_unstable"] == 1
    assert summary["num_kept"] == 1
    metadata = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert metadata[0]["relax_success"] is True
    assert metadata[0]["stability_label"] == "metastable"
    assert metadata[0]["is_msun"] is True
    ds = MP20Tokens(root=str(output_dir), split="train", nmax=4, augment_translate=False)
    kept = ds[0]
    assert torch.isclose(kept["Y1"][0], torch.tensor(math.log(5.0)))


def test_make_synthetic_dataset_stable_like_rejects_metastable_band(tmp_path: Path):
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    raw_stable = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    raw_metastable = Structure(
        Lattice.cubic(4.0),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    relaxed_stable = Structure(
        Lattice.cubic(5.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    relaxed_metastable = Structure(
        Lattice.cubic(5.0),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    (input_dir / "a.cif").write_text(str(CifWriter(raw_stable)), encoding="utf-8")
    (input_dir / "b.cif").write_text(str(CifWriter(raw_metastable)), encoding="utf-8")

    oracle = FakeOracle(
        [
            {
                "final_structure": relaxed_stable,
                "e_total": -10.0,
                "e_form": -1.5,
                "e_above_hull": -0.01,
                "nsteps": 12,
                "err": "",
            },
            {
                "final_structure": relaxed_metastable,
                "e_total": -9.0,
                "e_form": -1.0,
                "e_above_hull": 0.05,
                "nsteps": 10,
                "err": "",
            },
        ]
    )
    output_dir = tmp_path / "synthetic_out"
    summary = make_synthetic_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        dedup_mode="none",
        filter_level="stable_like",
        nmax=4,
        row_oracle=oracle,
    )

    assert summary["num_relax_attempted"] == 2
    assert summary["num_unstable"] == 1
    assert summary["num_kept"] == 1
    kept_metadata = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rejected_metadata = [
        json.loads(line)
        for line in (output_dir / "rejected.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert kept_metadata[0]["stability_label"] == "stable"
    assert kept_metadata[0]["is_stable"] is True
    assert rejected_metadata[0]["stability_label"] == "metastable"
    assert rejected_metadata[0]["filter_reason"] == "not_stable"
