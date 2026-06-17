# Flywheel — Ratification Manifest

The set of files that constitute the Flywheel recipe. When experiments have
landed and the design is stable, *these specific files* migrate to
`../crystalite` as the canonical implementation. Everything else in
`atom-reps` is brain-dump / provenance and stays here.

Migration timing: after the four chapter-critical experiments land
(Round 0 NequIP ✓, Round 0 eqv3, Round 1 / M2, S_big control). See
[run_tracker.md](run_tracker.md) for current status.

## In scope — migrates to `../crystalite`

### Core curator module

| File | Role |
|---|---|
| [src/data/synthetic_augmentation.py](../../src/data/synthetic_augmentation.py) | Single self-contained module. `make_synthetic_dataset` (curator; emits CIFs + `raw/train.csv` per kept candidate), `_build_row_oracle` (oracle dispatcher), `_ChgnetThermoOracle`, `AugmentedCrystalDataset`, `build_augmented_train_dataset`. Synthetic data is tokenized at load time via `MP20Tokens` so it goes through the same `preprocess(niggli=True, primitive=False)` path as MP20. |
| [src/eval/cfg_eval.py:NequipFormEnergyOracle](../../src/eval/cfg_eval.py) | The NequIP curation oracle class. Lives in a larger file but is a self-contained class; extract or migrate the whole file. |
| [src/eval/cfg_eval.py:EquiformerV3FormEnergyOracle](../../src/eval/cfg_eval.py) | The EquiformerV3 curation oracle (subprocess bridge). Same self-contained class shape as NequIP oracle. |
| [src/utils/sample_stats.py](../../src/utils/sample_stats.py) (subset) | The MP2020-correction helpers used by the curator: `_build_mp2020_parameters`, `compute_e_above_hull_mp2020_like`, `compute_e_above_hull_uncorrected`. The rest of this file is broader Crystalite infrastructure that's already likely in `../crystalite`. |

### Scripts

| File | Role |
|---|---|
| [scripts/make_synthetic_dataset.py](../../scripts/make_synthetic_dataset.py) | CLI wrapper around `make_synthetic_dataset`. |
| [scripts/make_synthetic_dataset_msun_25k.slurm](../../scripts/make_synthetic_dataset_msun_25k.slurm) | Slurm: Round 0 NequIP curation (canonical). |
| [scripts/make_synthetic_dataset_round1_msun_54k.slurm](../../scripts/make_synthetic_dataset_round1_msun_54k.slurm) | Slurm: Round 1 NequIP curation (M1 → S1, dedup vs MP20+S0). |
| [scripts/make_synthetic_dataset_round0_eqv3_msun_27k.slurm](../../scripts/make_synthetic_dataset_round0_eqv3_msun_27k.slurm) | Slurm: Round 0 EquiformerV3 curation. |
| [scripts/train_crystalite_synthetic_round0.slurm](../../scripts/train_crystalite_synthetic_round0.slurm) | Slurm: train M1 on MP20+S0. Doubles as the oversample-real control via env vars. |
| [scripts/train_crystalite_synthetic_round1.slurm](../../scripts/train_crystalite_synthetic_round1.slurm) | Slurm: train M2 on MP20+S0+S1. |

### EquiformerV3 verifier infrastructure

| File | Role |
|---|---|
| [scripts/setup_equiformer_v3.sh](../../scripts/setup_equiformer_v3.sh) | Provisions `external/equiformer_v3/` (cloned repo + inner venv + checkpoint). Idempotent. Auto-detect CPU/CUDA. |
| [scripts/check_equiformer_v3_install.sh](../../scripts/check_equiformer_v3_install.sh) | Verifies outer + inner venvs + checkpoint. |
| [scripts/smoke_equiformer_v3.py](../../scripts/smoke_equiformer_v3.py) | Local CPU smoke: load OAM checkpoint, single-point 5 known structures, verify finite energies. |
| [scripts/equiformer_v3_inference_wrapper.py](../../scripts/equiformer_v3_inference_wrapper.py) | The subprocess wrapper. Runs in the inner venv. CIFs in / relaxed CIFs + manifest out. |
| [scripts/build_equiformer_v3_hull.py](../../scripts/build_equiformer_v3_hull.py) | Protocol 1: MP-20 structures → optional EquiformerV3 relaxation (`--relax`) or single-point fallback → `PatchedPhaseDiagram` pickle in EquiformerV3 energy space. |
| [scripts/build_equiformer_v3_hull.slurm](../../scripts/build_equiformer_v3_hull.slurm) | Slurm wrapper for the hull build. |

### Verifier calibration / analysis scripts

| File | Role |
|---|---|
| [scripts/equiformer_v3_vs_mp20_ehull.py](../../scripts/equiformer_v3_vs_mp20_ehull.py) | Compare EquiformerV3 e_hull predictions to MP-20 stored DFT labels under raw vs MP2020 conditions. Produces the corrections-decision CSV. |
| [scripts/plot_equiformer_v3_vs_mp20.py](../../scripts/plot_equiformer_v3_vs_mp20.py) | Histogram + scatter from the corrections CSV. |
| [scripts/check_singlepoint_vs_relax.py](../../scripts/check_singlepoint_vs_relax.py) | Empirical check that single-point at DFT-min ≈ full eqv3-relax. Justifies Protocol 1's `max_steps=0`. |
| [scripts/analyze_nary_stratified.py](../../scripts/analyze_nary_stratified.py) | Joins manifest.json + LeMat JSON; per-arity and per-e_hull-band SUN/MSUN stratification + plots. Used for Round 0 NequIP results breakdown. |

### Tests

| File | Role |
|---|---|
| [tests/test_synthetic_augmentation.py](../../tests/test_synthetic_augmentation.py) | Covers `make_synthetic_dataset`, `AugmentedCrystalDataset`, multi-root synthetic, dedup against multiple references, msun_like with relaxed structures. |

### Docs (migrate the whole directory)

| File | Role |
|---|---|
| [docs/augmentation/README.md](README.md) | Index. |
| [docs/augmentation/experiment_design.md](experiment_design.md) | Method, controls, eval pipeline, filter tiers. The "what & why". |
| [docs/augmentation/equiformer_v3_setup.md](equiformer_v3_setup.md) | EquiformerV3 setup + Inference Protocols (1, 2, 3) + MP2020 decision. |
| [docs/augmentation/run_tracker.md](run_tracker.md) | Per-job status. Migrate as-is then drift away from cluster-specific paths. |
| [docs/augmentation/results_log.md](results_log.md) | Append-only per-round results. |
| [docs/augmentation/writeup_skeleton.md](writeup_skeleton.md) | Paper-shaped sections. |
| [docs/augmentation/followups.md](followups.md) | Backlog. |
| [docs/augmentation/MANIFEST.md](MANIFEST.md) | This file. |

### Reference artifacts (small, useful to bundle)

| File | Role |
|---|---|
| `results/equiformer_v3_vs_mp20_n1000.csv` | Verifier calibration CSV (1000 samples). |
| `results/singlepoint_vs_relax.csv` | Single-point vs relax check (20 entries). |
| `figures/augmentation/round0_*.png` | Round 0 NequIP stratified-analysis plots. |
| `figures/augmentation/equiformer_v3_vs_mp20_n1000_*.png` | Calibration figures. |

## Out of scope — stays in atom-reps

| File | Why it stays |
|---|---|
| ITS chapter scripts (`scripts/its_*`, `src/its/`, `scripts/eval_crystalite_cfg_*`) | Archived prior chapter direction. |
| Diagnostic / probe / smoke scripts (`scripts/diagnose_*`, `scripts/local_*`, `scripts/smoke_*` except the named EquiformerV3 ones, `scripts/probe_*`) | Brain-dump scratch work. |
| `external/equiformer_v3/` | Gitignored on purpose — provisioned by `setup_equiformer_v3.sh` at install time. |
| `outputs/`, `samples/`, `results_final/`, `wandb/`, `logs/` | Run artifacts. Final published numbers go in `docs/augmentation/results_log.md`. |
| Crystalite-core changes (Chapter 1 base model) | Already in `../crystalite`. |

## Cross-dependencies to be aware of

When migrating, these are the touch-points where Flywheel code reaches into broader Crystalite infrastructure:

```text
src/data/synthetic_augmentation.py
  ├── src/data/mp20_tokens.py            # NMAX, tokens_to_structure, lattice_to_Y
  ├── src/eval/cfg_eval.py               # NequipFormEnergyOracle / EquiformerV3FormEnergyOracle
  ├── src/utils/sample_stats.py          # MP2020 helpers (3 functions)
  ├── src/utils/dataset.py               # compute_allowed_elements
  └── src/eval_crystalite_ckpt.py        # _build_model_from_ckpt, _resolve_sampler_name, ...
                                          # (for the internal-generation path that
                                          #  loads M0 to produce candidates)

scripts/build_equiformer_v3_hull.py
  └── (no main-repo imports; standalone)

scripts/equiformer_v3_inference_wrapper.py
  └── (no main-repo imports; runs in inner venv only)
```

The `eval_crystalite_ckpt` dependency for internal-generation should already exist in `../crystalite` (it's Chapter 1 infrastructure). The MP2020 helpers may or may not be there — easiest to copy the three functions into the migrated `synthetic_augmentation.py` if not.

## Migration recipe (when ready)

After all four chapter-critical experiments have landed:

```bash
# From ../crystalite, on a fresh feature branch:
git checkout -b flywheel

# Core code (adjust paths to ../crystalite's module layout if different):
cp ../atom-reps/src/data/synthetic_augmentation.py src/data/
cp ../atom-reps/src/eval/cfg_eval.py src/eval/
# Or extract just the two oracle classes if cfg_eval.py has unrelated stuff.

# Scripts (curator + EquiformerV3 + analysis):
cp ../atom-reps/scripts/make_synthetic_dataset*.py scripts/
cp ../atom-reps/scripts/make_synthetic_dataset_*.slurm scripts/
cp ../atom-reps/scripts/train_crystalite_synthetic_*.slurm scripts/
cp ../atom-reps/scripts/{setup,check,smoke}_equiformer_v3.{sh,py} scripts/
cp ../atom-reps/scripts/equiformer_v3_inference_wrapper.py scripts/
cp ../atom-reps/scripts/build_equiformer_v3_hull.{py,slurm} scripts/
cp ../atom-reps/scripts/equiformer_v3_vs_mp20_ehull.{py,slurm} scripts/
cp ../atom-reps/scripts/plot_equiformer_v3_vs_mp20.py scripts/
cp ../atom-reps/scripts/check_singlepoint_vs_relax.py scripts/
cp ../atom-reps/scripts/analyze_nary_stratified.py scripts/

# Tests + docs:
cp ../atom-reps/tests/test_synthetic_augmentation.py tests/
cp -r ../atom-reps/docs/augmentation/ docs/

# Reference artifacts (small):
mkdir -p results figures/augmentation
cp ../atom-reps/results/equiformer_v3_vs_mp20_n1000.csv results/
cp ../atom-reps/results/singlepoint_vs_relax.csv results/
cp ../atom-reps/figures/augmentation/*.png figures/augmentation/

# Then:
# 1. Audit imports and fix any references that touch atom-reps-specific code.
# 2. Run `uv run pytest tests/test_synthetic_augmentation.py`.
# 3. Run scripts/check_equiformer_v3_install.sh (after setup_equiformer_v3.sh
#    re-provisions external/ in the new repo).
# 4. Update ../crystalite's README to reference docs/augmentation/.
# 5. Squash + PR.
```

## What gets archived in atom-reps after migration

The `synthetic-augmentation-curation` branch in `atom-reps` becomes a frozen
historical artifact. Future Flywheel development moves to `../crystalite`'s
mainline. atom-reps stays as the scratch / exploration repo for the next
chapter or paper.
