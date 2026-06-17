**# Experiment Design — Flywheel (Verifier-Curated Self-Training)

Stable design notes for the chapter's methodological contribution. Change this
doc when the design changes; not when a job runs.

## Position in the thesis

Methodological contribution 2 of 2. Chapter 1 (Crystalite) introduces the
generator architecture used here as the experimental setting. Chapter 2 (this
chapter) introduces the Flywheel recipe. The recipe is **generator-agnostic**
— it would apply unchanged to any base diffusion generator with a tokenized
training-data path. The Crystalite checkpoint `dng.pt` is used as `M0`
throughout for concrete experiments; results would not transfer numerically
to a different backbone, but the recipe and its analyses are not specific to
Crystalite.

## Research question

> Does augmenting a generator's training set with its own samples — relaxed,
> hull-filtered, and deduplicated — improve generative performance compared to
> (a) training on the real data alone and (b) a size-matched oversampled-real
> control?

The oversampled-real control is **mandatory**. Without it, any gain is
attributable either to "real synthetic novelty" or to "more rows / more
optimizer exposure," and we cannot tell which.

## Framing

Not "training the model on its own data." The synthetic samples are
generator-proposed but **externally postprocessed**: relaxed with an MLIP
(NequIP), scored against the convex hull (MP2020-corrected),
filtered for metastability, deduplicated against the training set by
structure. This weakens but does not eliminate proposal bias; the failure
mode to watch is self-distillation / distribution narrowing — track
diversity, novelty, formula entropy, and n-ary distribution.

## Core comparison

```text
A. real only                       (baseline)
B. real + N curated synthetic      (treatment)
C. real + N oversampled real       (size-matched control)
```

`N = |S_round|`. Currently `N ≈ 27k` for Round 0.

## Pipeline

```text
generate raw proposals from M_{r-1}
decode tokens → pymatgen Structure
cheap geometry prescreen (valid tier)
NequIP relax + MP2020-corrected e_above_hull
metastable label (e_hull ≤ 0.1 eV/atom)
StructureMatcher dedup within S_r
StructureMatcher dedup against MP20 ∪ {S_0, ..., S_{r-1}}
export accepted relaxed structures as MP20-token training examples
```

Implementation: [src/data/synthetic_augmentation.py:make_synthetic_dataset](../../src/data/synthetic_augmentation.py).

## Filter tiers

Defined in [src/data/synthetic_augmentation.py](../../src/data/synthetic_augmentation.py)
as `FILTER_LEVELS = ("raw", "valid", "relaxed", "relaxed_filtered", "stable_like", "msun_like")`.

| tier | gate |
|---|---|
| `raw` | parses |
| `valid` | + finite, sane volume, min interatomic distance, atom count ≤ nmax |
| `relaxed` | + relaxation succeeded |
| `relaxed_filtered` | + sane post-relax energy and volume change |
| `stable_like` | + strict-stable (e_hull ≤ 0.0) + unique within S + novel against reference |
| `msun_like` | + metastable (e_hull ≤ 0.1) + unique within S + novel against reference |

**Smoke tests** can use `valid` + formula dedup. **Main runs use `msun_like` +
StructureMatcher dedup**. Training examples are exported from the
post-relaxation structure when relaxation succeeded.

## Checkpoint selection (training-time)

For runs with synthetic augmentation, the primary `best.pt` selector is
`Train_MSUN` — MSUN computed against the **full augmented training set**
(MP20 ∪ S_0 ∪ … ∪ S_{r-1}), not MP20 alone. Without this, the selector
rewards a model that regenerates its own synthetic training data because
those regenerations look novel against the MP20-only reference. The bug was
fixed 2026-05-13 (see [run_tracker.md](run_tracker.md)); M1 and M2 trained
prior to the fix are preserved on disk under their original paths but
superseded by `_v2` re-trains.

Enable via `--sample_full_train_novelty --best_ckpt_selector auto` (defaults
in [scripts/train_crystalite_synthetic_round0.slurm](../../scripts/train_crystalite_synthetic_round0.slurm)).
For oversampled-real and real-only runs the augmented dataset reduces to MP20
duplicates, so `Train_MSUN ≡ MSUN` and the selector is unaffected — no
re-training needed for the control arms.

## Verifier (curation-time)

| element | choice |
|---|---|
| MLIP | NequIP-OAM-L batched (`aot_batch.nequip.pt2`) |
| Hull corrections | MP2020 (`thermo_ehull_method=mp2020_like`) |
| PPD | `mp_02072023/2023-02-07-ppd-mp.pkl` |
| Stability gate | `e_hull ≤ 0.1` (metastable); `≤ 0` (stable) |
| Dedup | `StructureMatcher` on relaxed structures |

Curation references: `--reference_data_root data/mp20[:S_0[:S_1...]]`.
Generation count priors / allowed elements come from `--generation_data_root data/mp20`.

## Evaluation (post hoc)

The MP hull used for curation is a **training-data construction heuristic**, not
the headline metric. Final claims come from external evaluation:

- LeMat-GenBench `comprehensive_multi_mlip_hull` with **MACE / ORB / UMA**
  enabled, on NequIP pre-relaxed CIFs, n=2500
- Comparators: public Crystalite MP-20 row, oversampled-real control,
  previous synthetic round

Headline metrics: `valid`, `unique`, `novel`, `stable`, `metastable`, `SUN`,
`MSUN`, `SUN+MSUN`, mean `e_hull`, `relax RMSD`. Stratify by n-ary bucket
(binary / ternary / quaternary / 5+) for confound control (see
[followups.md](followups.md)).

## Training

Same Crystalite recipe as the public MP-20 baseline. Augmentation is wired in
[src/train_crystalite.py](../../src/train_crystalite.py) via:

```bash
--augmentation {none|synthetic_concat|oversample_real}
--synthetic_data path[:path[...]]
--num_extra_samples N
```

Logged to wandb as `dataset_splits/{real,synthetic,oversampled_real,effective}_train_count`,
`dataset/augmentation_mode`, `dataset/synthetic_metadata_path`.

## Dataset-size schedule

Each round's accepted-sample target is set so the **effective training set
doubles** (MP20 ≈ 27k + cumulative synthetic). The per-round synthetic batch
therefore itself doubles: S0 = 27,138, S1 = 54,276 (2×), S2 = 108,552 (4×).
These are deliberate `MAX_SAMPLES` targets, not yield artifacts.

```text
Model   New batch     Cumulative synthetic   Effective train
M1      S0 = 27k      27k                     54k
M2      S1 = 54k      81k                     109k
M3      S2 = 108k     189k                    217k
```

This keeps the MSUN-vs-training-size figure on a clean log₂ x-axis and lets
each oversampled-real control be matched to the same effective-training-set
size. In the matrix below, the parenthetical "(…k)" is the **new batch size**
for that round, not the cumulative synthetic count.

## Experiment matrix

```text
A. real only                                            done (public Crystalite row)
B. real + S0 (msun_like, 27k)                           Round 0 — done
C. real + 27k oversampled real                          Round 0 control — done
D. real + S0 + S1 (msun_like, S1=54k)                   Round 1 — in flight
   alt control: real + 54k oversampled real             Round 1 control — in flight
E. real + S0 + S1 + S2 (msun_like, S2=108k)             Round 2 — queued
   alt control: real + 108k oversampled real            Round 2 control — queued
F. real + S_big one-shot (81k, msun_like, M0)           compute-comparison — in flight
G. real + S_dedup (valid+struct dedup, no MLIP, 27k)    ablation — queued; isolates verifier
H. real + S_raw (no filter, no dedup, 27k)              ablation — queued; self-distillation baseline
I. real + S0 (stable_like)                              ablation — queued
J. real + S0 (low_hull_topk + diversity caps)           ablation — pending
```

Conditions G and H are the curation ablations: G has dedup but no relaxation or e_hull filter
(`--filter_level valid --dedup_mode structure`); H has no curation at all
(`--filter_level raw --dedup_mode none`). Together with B (full Flywheel) and A (real-only),
they form the curation ladder: A → H → G → B.

See [run_tracker.md](run_tracker.md) for live status, [followups.md](followups.md)
for the broader ablation backlog.

## Target ablation suite

These are the ablations we are actively targeting for the Flywheel chapter. They
are designed to answer separate failure modes, not to be a generic sweep.

| ablation | train set | curation recipe | question answered | priority |
|---|---|---|---|---|
| **Round scaling** | `MP20 + S0`, `MP20 + S0 + S1`, `MP20 + S0 + S1 + S2` | iterative `msun_like`, StructureMatcher dedup against MP20 and prior synthetic rounds | Does the flywheel keep improving, and where does it saturate? | primary |
| **Oversampled-real controls** | `MP20 + N real resamples` | no synthetic data | Are gains just more optimizer exposure / larger effective training set? | required control |
| **One-shot `S_big`** | `MP20 + S_big` where `|S_big| = |S0| + |S1|` | M0 proposals only, `msun_like`, StructureMatcher dedup | Is iteration itself useful, or is one large curated buffer enough? | required control |
| **Dedup-only** | `MP20 + S_dedup_27k` | `valid` geometry + StructureMatcher dedup, no MLIP relaxation, no e_hull filter | Is novelty/deduplication alone enough without a physical verifier? | curation ablation |
| **Raw self-distillation** | `MP20 + S_raw_27k` | `raw`, no dedup, no MLIP | Does uncurated self-training help or hurt? | curation ablation |
| **Stable-only** | `MP20 + S_stable_27k` | relaxed + `e_hull <= 0.0` + StructureMatcher dedup | Does strict-stable selection recover stable yield or reduce the MSUN gains from broad metastable selection? | selection ablation |
| **Low-hull / diversity-capped** | TBD | tighter `e_hull` band plus formula/n-ary caps | Is the current gain partly from high-arity or mid-metastable distribution shift? | optional if compute allows |

Operational notes:

- The stable-only condition is not the current `msun_like` filter. `msun_like`
  keeps `e_hull <= 0.1`, including both stable and metastable structures.
  Stable-only should require `e_hull <= 0.0`.
- Stable-only may have much lower yield than S0. Match the accepted sample
  budget at 27,138 if feasible; if not, report the achieved budget and either
  train a size-matched oversampled-real control or downsample S0 to match.
- For every LeMat-evaluated model, report both external novelty and
  training-set novelty. Training-set novelty references are the exact structures
  used to train that model, e.g. `MP20 + S0 + S1` for M2.

## Verification checklist (per round)

- [ ] Curation: `--filter_level msun_like`, `--dedup_mode structure`,
      `--thermo_ehull_method mp2020_like`, `--thermo_mlip nequip`
- [ ] Reference roots include MP20 **and all previous synthetic rounds**
- [ ] Training: matched control `oversample_real` with `num_extra_samples = |S_round|`
      submitted alongside the synthetic run
- [ ] Wandb logs include `dataset_splits/*` and `dataset/augmentation_mode`
- [ ] Eval: NequIP pre-relax → LeMat `comprehensive_multi_mlip_hull` with
      MACE/ORB/UMA, n=2500, same seed across treatment + control
- [ ] Results landed in [results_log.md](results_log.md) with comparison vs
      public Crystalite, oversampled-real, and previous synthetic round
**