# Run Tracker

Live status of curation jobs, training jobs, and evaluations. Update inline as
jobs land. Move stale entries to a `## Done` section at the bottom; don't delete
artifact paths — they're needed for re-evaluation and the writeup.

## Conventions

- `[ ]` queued / not yet submitted
- `[~]` running
- `[x]` complete

Job naming convention: `dng_synthetic_round{N}_{filter}_{size}` for training,
`crystalite_round{N}_{filter}_{size}` for synthetic datasets.

## Checkpoint-selection bug (2026-05-13) — re-train cascade

**Bug:** training's primary checkpoint selector computed MSUN with novelty
referenced against MP20 only, not against the full augmented training set
(MP20 + S0 + …). Selected checkpoints were biased toward replaying synthetic
training data that happens to be novel against MP20. Inflation grew with each
round (+7.7 pp at M1, +11.4 pp at M2 between LeMat MSUN and train-novel MSUN).

**Fix:** [src/utils/checkpoint.py](../../src/utils/checkpoint.py) `auto`
selector now prefers `Train_MSUN`. `--sample_full_train_novelty` builds the
novelty reference from the augmented dataset
([src/train_crystalite.py:371-379](../../src/train_crystalite.py#L371-L379)).
Defaults in `train_crystalite_synthetic_round0.slurm` enable both. All new
trainings inherit the fix; the `_v1` `best.pt` files for M1 and M2 are
preserved on disk as evidence but are no longer load-bearing.

**Cascade:** re-train M1 (`_v2`) → re-curate S1 (`_v2`) → re-train M2 (`_v2`)
→ then Round 2 from clean M2_v2. Ablations and S_big are unaffected (curated
from M0 `dng.pt`, which had no synthetic data and therefore no bug).

## Round 0 — re-running (v2)

Generator: `dng.pt` (M0, public Crystalite MP-20). S0 is clean (curated from
M0, no contamination); M1 must be re-trained under the corrected selector.

| stage | status | artifact |
|---|---|---|
| Curate S0 | [x] | `data/synthetic/crystalite_round0_msun_27k/` (msun_like, structure dedup, 27,138 kept) — unchanged |
| Train M1 (v1, buggy) | [x] | `outputs/dng_synthetic_round0_msun_27k/` — preserved for bug exhibit; superseded |
| Train M1 (v2, corrected selector) | [x] | `outputs/dng_synthetic_round0_msun_27k_v2/` |
| Sample M1_v2 (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval — M1_v2 | [ ] | filename TBD |
| Train control (oversample 27k) | [x] | `outputs/dng_oversample_real_27k/` — unaffected (no synthetic in training set, so Train_MSUN ≡ MSUN) |
| Sample control (n=2500) | [x] | NequIP-relaxed CIFs exported |
| LeMat eval — control | [x] | `crystalite_oversample_real_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_171859.json` |
| LeMat eval — M1_v1 (superseded) | [x] | `crystalite_synthetic_aug_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_072436.json` |
| Corrected numbers in [results_log.md](results_log.md) | [ ] | append as Round 0 v2 block; keep v1 for bug exhibit |
| Writeup table corrected | [ ] | [writeup_skeleton.md](writeup_skeleton.md) |

## Round 1 — re-running (v2)

Generator: M1_v2 (`outputs/dng_synthetic_round0_msun_27k_v2/checkpoints/best.pt`).
Reference for dedup: `data/mp20:data/synthetic/crystalite_round0_msun_27k`.
Target size: 54,276 kept (doubles the additive budget; M2 train ≈ 108k).

| stage | status | artifact |
|---|---|---|
| Curate S1 (v1, from buggy M1) | [x] | `data/synthetic/crystalite_round1_msun_54k/` — preserved; superseded |
| Train M2 (v1, on buggy S1, buggy selector) | [x] | `outputs/dng_synthetic_round1_msun_108k/` — preserved; superseded |
| LeMat eval — M2_v1 | [~] | running 2026-05-12 — let it finish; useful as a v1-baseline number for the bug exhibit |
| Curate S1_v2 from M1_v2 | [x] | `data/synthetic/crystalite_round1_msun_54k_v2/` (initial yield); `crystalite_round1_msun_54k_v2_full` = `_v2` + `crystalite_round1_topup_1015` merged (54,276 final) — `_full` is the canonical training set |
| Train M2_v2 (on MP20 + S0 + S1_v2_full, corrected selector) | [x] | `outputs/dng_synthetic_round1_msun_108k_v2_resume/` (canonical; `_v2` and `_v2_resume.stale_20260516` siblings superseded) |
| Sample M2_v2 (n=2500) | [x] | `outputs/external_eval/synthetic_round1_v2_n2500_nequip_relaxed/` (raw + NequIP-relaxed CIFs) |
| LeMat eval — M2_v2 | [x] | `m2v2.json` (repo root) — MSUN 35.06, SUN 2.28, novelty 70.91, S+M 37.34. Replay re-scorings against corrected train-novel reference in `results_final/replay_m2v2_*.json` |
| Train Round 1 oversample control (81k) | [~] | target: `outputs/dng_oversample_real_81k/`; unaffected by bug (no synthetic in training set), let finish |
| Sample Round 1 control (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval — Round 1 control | [ ] | filename TBD |
| Corrected numbers in [results_log.md](results_log.md) | [ ] | Round 1 v2 block |
| Writeup updates | [ ] | main table + iterative-self-improvement paragraph |

## Round 2

Generator: M2_v2 (`outputs/dng_synthetic_round1_msun_108k_v2_resume/checkpoints/best.pt`).
Reference for dedup: `data/mp20:data/synthetic/crystalite_round0_msun_27k:data/synthetic/crystalite_round1_msun_54k_v2_full`.
Target size: 108,552 kept (M3 train ≈ 217k).

| stage | status | artifact |
|---|---|---|
| Curate S2 from M2_v2 | [x] | `data/synthetic/crystalite_round2_msun_108k_v2/` (108,552 kept) |
| Train M3 (corrected selector) | [x] | `outputs/dng_synthetic_round2_msun_217k_v2/` (MP20 + S0 + S1_v2_full + S2_v2 = 217,104) |
| Sample M3 (n=2500) | [x] | `outputs/external_eval/m3_msun_217k_v2_n2500/` (raw + NequIP-relaxed CIFs) |
| LeMat eval — M3 | [x] | `~/lemat-genbench/results_final/crystalite_m3_msun_217k_v2_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260602_182933.json` — MSUN 40.43, SUN 2.54, novelty 76.51, S+M 42.96. Replay (vs S0∪S1_full∪S2): `outputs/msun_replay/m3/summary.json` — 7.69 pp MSUN, 0.57 pp SUN → train-novel MSUN 32.74, train-novel SUN 1.97 |
| Train Round 2 oversample control (217k) | [ ] | target: `outputs/dng_oversample_real_217k/` |
| Sample Round 2 control (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval — Round 2 control | [ ] | filename TBD |
| Numbers in [results_log.md](results_log.md) | [ ] | Round 2 section |
| Writeup updates | [ ] | main table + saturation paragraph |

## Round 3 — done

Generator: M3 (`outputs/dng_synthetic_round2_msun_217k_v2/checkpoints/best.pt`).
Reference for dedup: `data/mp20:S0:S1_v2_full:S2_v2`.
Target size: 217,104 kept (M4 train ≈ 434k).

| stage | status | artifact |
|---|---|---|
| Curate S3 from M3 | [x] | Main run undershot (206,215 in CSV / 207,914 in metadata) due to 96h wall-clock SIGTERM + buffered csv.writer rows lost. Two topups closed the gap: `crystalite_round3_topup_9190/` (9,190 kept) and `crystalite_round3_topup_1699/` (1,699 kept). Merged to `data/synthetic/crystalite_round3_msun_217k_v2_full/` (217,104 total). Slurms: `make_synthetic_dataset_round3_msun_217k.slurm`, `make_synthetic_dataset_round3_topup_9190.slurm`, `make_synthetic_dataset_round3_topup_1699.slurm`. |
| Train M4 (corrected selector) | [x] | `outputs/dng_synthetic_round3_msun_434k_v2/` (MP20 + S0 + S1_v2_full + S2_v2 + S3_v2_full = 434,208) |
| Sample M4 (n=2500) | [x] | `outputs/external_eval/m4_msun_434k_v2_n2500/` (raw + NequIP-relaxed CIFs) |
| LeMat eval — M4 | [x] | `~/lemat-genbench/results_final/crystalite_m4_msun_434k_v2_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260613_004319.json` — MSUN 48.20, SUN 2.90, novelty 81.14, S+M 51.10, validity 98.00%, ē_hull 0.086, HHI_prod 4.12 |
| Replay (vs S0+S1+S2+S3) | [x] | `outputs/msun_replay/m4/summary.json` — replay 10.86 pp MSUN, 0.78 pp SUN → train-novel MSUN 37.34, train-novel SUN 2.12. Per-reference-structure replay rate 2.67 ppm (falling monotonically from M1's 12.0 ppm) |
| Numbers in [results_log.md](results_log.md) | [ ] | Round 3 section |
| Writeup updates | [x] | extended cascade tables, replay-corrected table, abstract, mechanism #4, dataset schedule in [writeup_skeleton.md](writeup_skeleton.md); plots re-rendered (`figures/flywheel/flywheel_lemat_*.png`, `figures/flywheel/flywheel_*_vs_dataset_size.png`) |

## Ablation: dedup-only — queued

Filtering variant from M0: validity prescreen + structure dedup, no MLIP relaxation, no e_hull gate.
Isolates whether the physical verifier (relaxation + metastability) drives the gain vs. just "novel non-duplicate generated structures."

| stage | status | artifact |
|---|---|---|
| Curate S_dedup | [ ] | target: `data/synthetic/crystalite_dedup_only_27k/` (`scripts/make_synthetic_dataset_dedup_only_27k.slurm`) |
| Train M_dedup | [ ] | target: `outputs/dng_synthetic_dedup_only_54k/` (`scripts/train_crystalite_synthetic_dedup_only.slurm`) |
| Sample M_dedup (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval | [ ] | filename TBD |

## Ablation: raw (no filter, no dedup) — queued

Filtering variant from M0: no geometry check, no dedup, no MLIP. Straightforward self-distillation baseline.
Tests whether any curation whatsoever is required for the Flywheel effect.

| stage | status | artifact |
|---|---|---|
| Curate S_raw | [ ] | target: `data/synthetic/crystalite_raw_27k/` (`scripts/make_synthetic_dataset_raw_27k.slurm`) |
| Train M_raw | [ ] | target: `outputs/dng_synthetic_raw_54k/` (`scripts/train_crystalite_synthetic_raw.slurm`) |
| Sample M_raw (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval | [ ] | filename TBD |

## Ablation: stable-only selection — queued

Filtering variant from M0: NeQuIP relaxation + MP2020 hull scoring + StructureMatcher dedup,
but accept only strict-stable structures (`e_hull <= 0.0`) instead of the broader
`msun_like` metastable band (`e_hull <= 0.1`). This directly tests the hypothesis
that the current strict-stability drop is caused by training on metastable-band
survivors.

Expected issue: stable-only yield may be low, so `NUM_GENERATE` likely needs to
be substantially larger than the S0 run. Target accepted size is 27,138 if
feasible.

| stage | status | artifact |
|---|---|---|
| Add `stable_like` / stable-only filter level | [x] | `src/data/synthetic_augmentation.py`; `scripts/make_synthetic_dataset_stable_only_27k.slurm` |
| Curate S_stable | [ ] | target: `data/synthetic/crystalite_stable_only_27k/` (`scripts/make_synthetic_dataset_stable_only_27k.slurm`) |
| Train M_stable | [ ] | target: `outputs/dng_synthetic_stable_only_54k/` (`scripts/train_crystalite_synthetic_stable_only.slurm`) |
| Sample M_stable (n=2500) | [ ] | NequIP-relaxed CIFs |
| LeMat eval | [ ] | filename TBD |
| Training-set novelty re-score | [ ] | train refs: `data/mp20:data/synthetic/crystalite_stable_only_27k` |

## Distribution shift analysis — queued

Characterize how composition/element/e_hull distributions shift as the flywheel iterates.
Script: [`scripts/analyze_flywheel_dataset_shift.py`](../../scripts/analyze_flywheel_dataset_shift.py).

| stage | status | artifact |
|---|---|---|
| Run on MP20 + S0 + S1 (available now) | [ ] | `figures/flywheel_dataset_shift/` |
| Re-run after S2 lands | [ ] | update same dir |
| Re-run after S_big lands | [ ] | iterative vs one-shot distribution comparison |

## Ablations — backlog

Promote to a Round-style block when scheduled. See [followups.md](followups.md)
for full motivation.

| ablation | status | notes |
|---|---|---|
| `stable_like` filter (e_hull ≤ 0) | [ ] | promoted above as queued-after-implementation selection ablation |
| `low_hull_topk` + diversity cap | [ ] | tests sharper survivor band, controls n-ary inflation |
| n-ary stratified evaluation | [x] | 2026-05-11. `scripts/analyze_nary_stratified.py`; synthetic wins SUN+MSUN in every multi-element bucket. Confound closed. See [results_log.md](results_log.md) "N-ary stratified re-scoring". |
| e_hull stratified evaluation | [x] | 2026-05-11. Same script; MSUN gain concentrates in mid-metastable band (+65, vs +36 near, +33 far). Within strict-stable, SUN conversion rises 15.4%→22.9%. Plots in [figures/augmentation/](../../figures/augmentation/). See [results_log.md](results_log.md) "E-above-hull stratification". |
| `S_big` one-shot from M0 (matched-budget control for **M2_v2**, not M3) | [x] | Curated: `data/synthetic/crystalite_S_big_81k/` (69,299 initial) + `crystalite_S_big_topup_12115/` (12,115) → `crystalite_S_big_81k_merged/` (81,414 = S0+S1 budget). Trained: `outputs/dng_synthetic_S_big_matched_81k/` (canonical; `dng_synthetic_S_big_v2` / `_v2_resume` were earlier aborted runs on the un-merged 69k). Comparison: M_big vs M2_v2 (both ≈108k train) tests iteration-vs-one-shot at matched budget; no matched control planned for M3. Samples: `outputs/external_eval/s_big_matched_81k_n2500/` (raw + NequIP-relaxed). LeMat: `~/lemat-genbench/results_final/crystalite_s_big_matched_81k_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260602_184302.json` — MSUN 32.91, SUN 1.82, novelty 69.46, S+M 34.73. Replay (vs S_big_81k_merged): `outputs/msun_replay/s_big/summary.json` — 5.14 pp MSUN, 0.21 pp SUN → train-novel MSUN 27.77, train-novel SUN 1.61. **Headline: iteration > one-shot at matched 108k budget — M2_v2 (35.06 MSUN, 30.75 train-novel MSUN) beats M_big (32.91 MSUN, 27.77 train-novel MSUN). The +2.98 pp train-novel gap is *wider* than the +2.15 pp raw-LeMat gap because M_big leans harder on replay (15.6% of LeMat-MSUN vs M2's 12.3%).** |
| EquiformerV3 verifier swap | [~] | NequIP→EquiformerV3-OAM. Env installed local+cluster. Single-point vs full-relax check done (|Δe|/atom < 3 meV; single-points sufficient). MP-20 calibration done at n=1000 (MP2020 RMSE 0.037 / metastable agreement 99.1%). **Hull build (full-MP via mp-api) in flight 2026-05-12.** Round 0 eqv3 curation queued behind hull. |
| Stronger / ensemble verifier curation | [ ] | swap NequIP→MACE-MP or NequIP+MACE for curation; eval still external |
| Alex-MP / Alex-MP20 training base | [ ] | tests transfer of recipe across training distributions |
| Online / semi-online buffered loop | [ ] | replay-buffer prototype; cost-bounded prescreener |

## Done

(none yet — Round 0 still considered in active discussion until Round 1 lands.)
