"""MLIP relaxation + hull-scoring oracles for the Flywheel curator.

Each oracle exposes ``call_many(structures) -> list[row]`` where a row carries
``e_form``, ``e_above_hull``, ``e_total``, ``nsteps``, ``final_structure`` and
``err``. They relax a proposed structure with an MLIP, then score it against an
MP phase diagram (optionally with MP2020 corrections) using the shared helpers
in ``src.utils.sample_stats`` / ``src.eval.stability`` — the same machinery the
train-time ``StabilityLogger`` uses.

Extracted from the (removed) CFG eval harness; nothing here is CFG-specific.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


class NequipFormEnergyOracle:
    """Relax with NequIP, then compute MP-consistent formation energy and
    e_above_hull per atom via the MP phase diagram.

    Per structure::

        NequIP relax(struct) -> e_total
                ↓
        ComputedStructureEntry(e_total) -> (optional) MP2020 corrections
                ↓
        ppd.get_form_energy_per_atom(entry)  [eV/atom, matches MP20 labels]
        ppd.get_e_above_hull(entry)          [eV/atom, >= 0]

    The relaxed structure is returned alongside the scalars so callers can
    run SpacegroupAnalyzer on the relaxed geometry without a second relax.
    """

    def __init__(
        self,
        *,
        nequip_compile_path: str,
        ppd_path: str,
        stability_device: str,
        optimizer: str = "FIRE",
        cell_filter: str = "frechet",
        fmax: float = 0.05,
        max_force_abort: float = 1e6,
        relax_steps: int = 200,
        apply_mp2020: bool = True,
        relax_mode: str = "sequential",
        batch_size: int = 32,
    ) -> None:
        from src.eval.stability import load_phase_diagram

        self.relax_mode = str(relax_mode).strip().lower()
        if self.relax_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"relax_mode must be 'sequential' or 'batch', got {relax_mode!r}."
            )
        if self.relax_mode == "batch":
            from src.utils.sample_stats import make_nequip_batch_relaxer
            _, self.relaxer, self.device, _ = make_nequip_batch_relaxer(
                compile_path=nequip_compile_path,
                stability_device=stability_device,
                optimizer_name=optimizer,
                cell_filter=cell_filter,
                max_force_abort=max_force_abort,
            )
        else:
            from src.utils.sample_stats import make_nequip_relaxer
            _, self.relaxer, self.device, _ = make_nequip_relaxer(
                compile_path=nequip_compile_path,
                stability_device=stability_device,
                optimizer_name=optimizer,
                cell_filter=cell_filter,
                fmax=fmax,
                max_force_abort=max_force_abort,
            )
        self.ppd = load_phase_diagram(ppd_path)
        self.relax_steps = int(relax_steps)
        self.apply_mp2020 = bool(apply_mp2020)
        # Cap the number of structures handed to NequIP's batched relax_many
        # in a single call. Matches StabilityLogger's thermo_stability_batch.
        self.batch_size = max(1, int(batch_size))

        self._mp2020_compat = None
        if self.apply_mp2020:
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
            self._mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)

    @staticmethod
    def _empty_row(*, err: str = "") -> dict:
        return {
            "e_form": float("nan"),
            "e_above_hull": float("nan"),
            "e_total": float("nan"),
            "nsteps": -1,
            "final_structure": None,
            "err": err,
        }

    def _score_relaxed(self, final_struct, e_total: float, nsteps: int) -> dict:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from src.utils.sample_stats import (
            _build_mp2020_parameters,
            compute_e_above_hull_mp2020_like,
            compute_e_above_hull_uncorrected,
        )

        out = self._empty_row()
        out["e_total"] = float(e_total)
        out["nsteps"] = int(nsteps)
        out["final_structure"] = final_struct
        if not math.isfinite(e_total):
            out["err"] = "nan_total_energy"
            return out

        if self.apply_mp2020:
            e_above_val, fail_reason = compute_e_above_hull_mp2020_like(
                self.ppd,
                final_struct,
                float(e_total),
                mp2020_compat=self._mp2020_compat,
            )
        else:
            e_above_val, fail_reason = compute_e_above_hull_uncorrected(
                self.ppd, final_struct, float(e_total)
            )
        if fail_reason is not None:
            out["err"] = fail_reason
        elif e_above_val is not None:
            out["e_above_hull"] = float(e_above_val)

        try:
            params = None
            if self.apply_mp2020:
                params = _build_mp2020_parameters(
                    composition=final_struct.composition,
                    mp2020_compat=self._mp2020_compat,
                )
            entry = ComputedStructureEntry(
                composition=final_struct.composition,
                energy=float(e_total),
                structure=final_struct,
                parameters=params,
            )
            if self.apply_mp2020:
                entry = self._mp2020_compat.process_entry(
                    entry.copy(), on_error="raise"
                )
                if entry is None:
                    out["err"] = out["err"] or "mp2020_removed"
                    return out
        except Exception as exc:  # noqa: BLE001
            out["err"] = out["err"] or f"entry_exc:{type(exc).__name__}"
            return out

        try:
            out["e_form"] = float(self.ppd.get_form_energy_per_atom(entry))
        except Exception as exc:  # noqa: BLE001
            out["err"] = out["err"] or f"form_exc:{type(exc).__name__}"
        return out

    def _postprocess_relax_result(self, result: dict) -> dict:
        final_struct = result["final_structure"]
        energies = getattr(result["trajectory"], "energies", [])
        if not energies:
            return self._empty_row(err="no_energy")
        return self._score_relaxed(
            final_struct,
            float(energies[-1]),
            int(result.get("nsteps", -1)),
        )

    def call_many(self, structs: list) -> list[dict]:
        if not structs:
            return []
        if self.relax_mode != "batch":
            return [self._relax_single(s) for s in structs]

        # Batched path: chunk into at most `self.batch_size` structures per
        # NequIP call so neighbor-list allocation stays bounded, and recursively
        # bisect within each chunk on structure-level exceptions so one bad
        # sample doesn't kill its neighbors (pattern ported from
        # StabilityLogger.relax_batched — see src/utils/stability_logger.py).
        results: list[dict | None] = [None] * len(structs)
        for start in range(0, len(structs), self.batch_size):
            end = min(len(structs), start + self.batch_size)
            chunk = list(structs[start:end])
            chunk_idx = list(range(start, end))
            self._relax_batch_with_bisection(chunk, chunk_idx, results)

        for i, row in enumerate(results):
            if row is None:
                results[i] = self._empty_row(err="relax_bisect_no_result")
        return results  # type: ignore[return-value]

    def _relax_batch_with_bisection(
        self,
        chunk: list,
        orig_idx: list[int],
        results: list,
    ) -> None:
        """Recursive-halve on exception until either the chunk succeeds or a
        single struct bottoms out with an empty row (err=relax_exc_...)."""
        if not chunk:
            return
        try:
            relaxations = self.relaxer.relax_many(chunk, steps=self.relax_steps)
            if len(relaxations) != len(chunk):
                raise RuntimeError(
                    f"batched NequIP returned {len(relaxations)} relaxations "
                    f"for chunk of {len(chunk)}"
                )
        except Exception as exc:  # noqa: BLE001
            if len(chunk) == 1:
                results[orig_idx[0]] = self._empty_row(
                    err=f"relax_exc_{type(exc).__name__}"
                )
                return
            mid = len(chunk) // 2
            self._relax_batch_with_bisection(chunk[:mid], orig_idx[:mid], results)
            self._relax_batch_with_bisection(chunk[mid:], orig_idx[mid:], results)
            return

        for i, relaxation in enumerate(relaxations):
            try:
                results[orig_idx[i]] = self._postprocess_relax_result(relaxation)
            except Exception as exc:  # noqa: BLE001
                results[orig_idx[i]] = self._empty_row(
                    err=f"postprocess_exc_{type(exc).__name__}"
                )

    def _relax_single(self, struct) -> dict:
        try:
            if self.relax_mode == "batch":
                results = self.relaxer.relax_many([struct], steps=self.relax_steps)
                if not results:
                    return self._empty_row(err="relax_exc_no_result")
                return self._postprocess_relax_result(results[0])
            result = self.relaxer.relax(struct, steps=self.relax_steps)
            return self._postprocess_relax_result(result)
        except Exception as exc:  # noqa: BLE001
            return self._empty_row(err=f"relax_exc_{type(exc).__name__}")


# ---------------------------------------------------------------------------
# EquiformerV3 oracle (subprocess bridge to the inner equiformer_v3 venv)
# ---------------------------------------------------------------------------


class EquiformerV3FormEnergyOracle:
    """Relax with EquiformerV3-OAM in an inner venv, then score against a PPD.

    The EquiformerV3 inference stack (fairchem fork @977a803, torch 2.7.1,
    PyG) cannot share a Python process with the main repo's nequip 0.17.0 +
    torch 2.8.0 stack. We bridge to it by spawning
    `scripts/equiformer_v3_inference_wrapper.py` under the inner venv's
    interpreter, passing structures as CIFs in a tempdir and reading back a
    JSONL manifest. See docs/augmentation/equiformer_v3_setup.md for the
    three-protocol breakdown (this oracle implements Protocol 3 — candidate
    relaxation).

    Constructor signature mirrors NequipFormEnergyOracle so the curator's
    `_build_row_oracle` can swap them with no other code changes.
    """

    def __init__(
        self,
        *,
        inner_python: str,
        wrapper_path: str,
        checkpoint_path: str,
        ppd_path: str,
        fmax: float = 0.02,
        max_steps: int = 500,
        cell_filter: str = "frechet",
        optimizer: str = "FIRE",
        apply_mp2020: bool = True,
        device: str = "auto",
        keep_tempdirs: bool = False,
    ) -> None:
        from src.eval.stability import load_phase_diagram

        self.inner_python = str(inner_python)
        self.wrapper_path = str(wrapper_path)
        self.checkpoint_path = str(checkpoint_path)
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.cell_filter = str(cell_filter)
        self.optimizer = str(optimizer)
        self.device = str(device)
        self.apply_mp2020 = bool(apply_mp2020)
        self.keep_tempdirs = bool(keep_tempdirs)

        for path_label, path_str in (
            ("inner_python", self.inner_python),
            ("wrapper_path", self.wrapper_path),
            ("checkpoint_path", self.checkpoint_path),
        ):
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"EquiformerV3 oracle: {path_label} missing: {path_str}")

        self.ppd = load_phase_diagram(ppd_path)
        self._mp2020_compat = None
        if self.apply_mp2020:
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
            self._mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)

    @staticmethod
    def _empty_row(*, err: str = "") -> dict:
        return {
            "e_form": float("nan"),
            "e_above_hull": float("nan"),
            "e_total": float("nan"),
            "nsteps": -1,
            "final_structure": None,
            "err": err,
        }

    def _score_relaxed(self, final_struct, e_total: float, nsteps: int) -> dict:
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from src.utils.sample_stats import (
            _build_mp2020_parameters,
            compute_e_above_hull_mp2020_like,
            compute_e_above_hull_uncorrected,
        )

        out = self._empty_row()
        out["e_total"] = float(e_total)
        out["nsteps"] = int(nsteps)
        out["final_structure"] = final_struct
        if not math.isfinite(e_total):
            out["err"] = "nan_total_energy"
            return out

        if self.apply_mp2020:
            e_above_val, fail_reason = compute_e_above_hull_mp2020_like(
                self.ppd, final_struct, float(e_total), mp2020_compat=self._mp2020_compat,
            )
        else:
            e_above_val, fail_reason = compute_e_above_hull_uncorrected(
                self.ppd, final_struct, float(e_total),
            )
        if fail_reason is not None:
            out["err"] = fail_reason
        elif e_above_val is not None:
            out["e_above_hull"] = float(e_above_val)

        try:
            params = None
            if self.apply_mp2020:
                params = _build_mp2020_parameters(
                    composition=final_struct.composition, mp2020_compat=self._mp2020_compat,
                )
            entry = ComputedStructureEntry(
                composition=final_struct.composition,
                energy=float(e_total),
                structure=final_struct,
                parameters=params,
            )
            if self.apply_mp2020:
                entry = self._mp2020_compat.process_entry(entry.copy(), on_error="raise")
                if entry is None:
                    out["err"] = out["err"] or "mp2020_removed"
                    return out
        except Exception as exc:
            out["err"] = out["err"] or f"entry_exc:{type(exc).__name__}"
            return out

        try:
            out["e_form"] = float(self.ppd.get_form_energy_per_atom(entry))
        except Exception as exc:
            out["err"] = out["err"] or f"form_exc:{type(exc).__name__}"
        return out

    def call_many(self, structs: list) -> list[dict]:
        import shutil
        import subprocess
        import tempfile
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter

        if not structs:
            return []

        tmpdir = Path(tempfile.mkdtemp(prefix="eqv3_oracle_"))
        in_dir = tmpdir / "in"
        out_dir = tmpdir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        # Dump each structure as a CIF; sample_id is the filename stem.
        for idx, s in enumerate(structs):
            cif_path = in_dir / f"sample_{idx:06d}.cif"
            cif_path.write_text(str(CifWriter(s)))

        cmd = [
            self.inner_python, self.wrapper_path,
            "--input-dir", str(in_dir),
            "--output-dir", str(out_dir),
            "--checkpoint", self.checkpoint_path,
            "--max-steps", str(self.max_steps),
            "--fmax", str(self.fmax),
            "--cell-filter", self.cell_filter,
            "--optimizer", self.optimizer,
            "--device", self.device,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as exc:
            if not self.keep_tempdirs:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return [self._empty_row(err=f"subprocess_exc:{type(exc).__name__}") for _ in structs]

        # Subprocess might still have written a partial manifest even if it
        # crashed late; try to read it before giving up.
        manifest_path = out_dir / "manifest.jsonl"
        rows_by_id: dict[str, dict] = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    sample_id = rec.get("sample_id")
                    if sample_id:
                        rows_by_id[sample_id] = rec

        if proc.returncode != 0 and not rows_by_id:
            # Hard subprocess failure with nothing recoverable from the manifest.
            stderr_tail = (proc.stderr or "").splitlines()[-5:]
            if not self.keep_tempdirs:
                shutil.rmtree(tmpdir, ignore_errors=True)
            err = f"subprocess_returncode_{proc.returncode}"
            if stderr_tail:
                err += ": " + " | ".join(stderr_tail)
            return [self._empty_row(err=err) for _ in structs]

        # Build per-structure rows in input order.
        rows: list[dict] = []
        for idx, _ in enumerate(structs):
            sample_id = f"sample_{idx:06d}"
            rec = rows_by_id.get(sample_id)
            if rec is None:
                rows.append(self._empty_row(err="manifest_missing"))
                continue
            if not rec.get("success"):
                rows.append(self._empty_row(err=str(rec.get("err") or "relax_failed")))
                continue
            out_path = rec.get("output_path")
            if not out_path:
                rows.append(self._empty_row(err="output_path_missing"))
                continue
            # output_path is relative to the parent of out_dir (= tmpdir);
            # see how the wrapper writes it.
            cif_full = tmpdir / out_path
            if not cif_full.exists():
                rows.append(self._empty_row(err="output_cif_missing"))
                continue
            try:
                final_struct = Structure.from_file(str(cif_full))
            except Exception as exc:
                rows.append(self._empty_row(err=f"output_parse_exc:{type(exc).__name__}"))
                continue

            e_total = rec.get("e_total")
            nsteps = rec.get("nsteps", -1)
            if e_total is None:
                rows.append(self._empty_row(err="e_total_missing"))
                continue
            rows.append(self._score_relaxed(final_struct, float(e_total), int(nsteps)))

        if not self.keep_tempdirs:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return rows
