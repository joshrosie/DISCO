# Flywheel — Docs Index

**Thesis Chapter 2.** Methodological contribution 2 of 2 (Chapter 1 is Crystalite).

Flywheel is the verifier-curated self-training recipe: generator-proposed
candidates are relaxed with an MLIP, hull-filtered, deduplicated, and added
back into the training set. The recipe is generator-agnostic; here we
instantiate it on the Crystalite generator (Chapter 1) and evaluate on
LeMat-GenBench. The earlier ITS line is archived indefinitely.

## What lives where

| Doc | Cadence | Purpose |
|---|---|---|
| [experiment_design.md](experiment_design.md) | stable | Method, controls, eval pipeline, filter tiers, curation defaults. The "what & why" — change only when the design changes. |
| [run_tracker.md](run_tracker.md) | volatile | Per-job status: dataset paths, ckpts, LeMat JSONs, what's running / queued / done. The punch-list. |
| [results_log.md](results_log.md) | append-only | Per-round numbers + interpretation (Round 0 → Round 1 → ablations). Don't rewrite history; append the next round below. |
| [writeup_skeleton.md](writeup_skeleton.md) | living | Paper-shaped sections (title, abstract, method para, main table, discussion, limitations). Update when a result lands so the writeup converges to one editing pass. |
| [followups.md](followups.md) | append/prune | Open ablations and online-extension backlog. Promote items to `run_tracker` when scheduled; prune when killed or shipped. |
| [equiformer_v3_setup.md](equiformer_v3_setup.md) | reference | EquiformerV3 verifier: which checkpoint, why, how to install, matbench-aligned hyperparameters, MP2020 corrections open question. Update only when the verifier choice or protocol changes. |
| [MANIFEST.md](MANIFEST.md) | reference | Files that constitute the Flywheel implementation. Used to migrate the recipe to `../crystalite` when experiments have landed. Update when scope changes (add file, remove file). |

## Core code

- Curator: [src/data/synthetic_augmentation.py](../../src/data/synthetic_augmentation.py) — `make_synthetic_dataset` (writes CIFs + `raw/train.csv`), oracle adapters, `AugmentedCrystalDataset`, `build_augmented_train_dataset`. Synthetic data is tokenized at load time through `MP20Tokens(root=<synthetic-run>, split="train")` — same `preprocess(niggli=True, primitive=False)` path as MP20.
- CLI: [scripts/make_synthetic_dataset.py](../../scripts/make_synthetic_dataset.py)
- Tests: [tests/test_synthetic_augmentation.py](../../tests/test_synthetic_augmentation.py)
- Slurm:
  - [scripts/make_synthetic_dataset_msun_25k.slurm](../../scripts/make_synthetic_dataset_msun_25k.slurm) — Round 0 / canonical curation
  - [scripts/make_synthetic_dataset_round1_msun_54k.slurm](../../scripts/make_synthetic_dataset_round1_msun_54k.slurm) — Round 1 (uses M1 ckpt, dedup against MP20+S0)
  - [scripts/train_crystalite_synthetic_round0.slurm](../../scripts/train_crystalite_synthetic_round0.slurm) — train M1; same script does the oversampled-real control via env vars
  - [scripts/train_crystalite_synthetic_round1.slurm](../../scripts/train_crystalite_synthetic_round1.slurm) — train M2 on MP20 + S0 + S1
