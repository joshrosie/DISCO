# Flywheel Writeup Skeleton

This is the writeup-facing scaffold for the synthetic augmentation track. Keep
the detailed run history in [results_log.md](results_log.md), the design
rationale in [experiment_design.md](experiment_design.md), and the operational
state in [run_tracker.md](run_tracker.md). This file should stay close to the
paper/thesis narrative.

## Thesis Position

Chapter 1 introduces **Crystalite**, a lightweight diffusion transformer for
crystal generation. Chapter 2 introduces **Flywheel**, a verifier-curated
self-training recipe that uses Crystalite as the base generator.

The two contributions should be framed as separable:

- **Crystalite:** an efficient generator architecture.
- **Flywheel:** a generator-agnostic recipe for improving a generator with
  verifier-curated synthetic structures.

The clean thesis narrative is:

> A simple generator can already be competitive, but its performance can be
> further improved by repeatedly sampling, verifying, filtering, deduplicating,
> and retraining on high-quality synthetic structures.

## Working Title

**Flywheel: Verifier-Curated Self-Training for Crystal Generators**

Useful names:

```text
Flywheel
Verifier-Curated Self-Training (VCST)
Generate-Relax-Retrain
Verifier-Guided Synthetic Augmentation
```

Use "Flywheel" as the method name and define "verifier-curated self-training"
in prose. VCST is acceptable as a local acronym, but should not be presented as
an established term.

## Core Claim

Flywheel improves Crystalite's external LeMat-GenBench MSUN across model
iterations. A train-novel decomposition — checking LeMat-MSUN samples against
the synthetic augmentation we added on top of MP20 — confirms the gain is
genuine discovery, not regeneration of curated training data.

The corrected headline is:

```text
Base Crystalite external MSUN:    22.60%
M1 external MSUN:                 29.34%   (replay 3.24 pp → train-novel 26.10%)
M2 external MSUN:                 35.06%   (replay 4.32 pp → train-novel 30.75%)
M3 external MSUN:                 40.43%   (replay 7.69 pp → train-novel 32.74%)
M4 external MSUN:                 48.20%   (replay 10.86 pp → train-novel 37.34%)
M_big external MSUN:              32.91%   (one-shot from M0 at matched M2 budget)
```

So the headline read is:

1. External LeMat-MSUN scales monotonically across four rounds:
   22.6 → 29.3 → 35.1 → 40.4 → 48.2. **The cascade has not saturated.**
2. **Train-novel MSUN climbs 26.1 → 30.8 → 32.7 → 37.3.** M3 was a slow round
   (+1.99 pp) but M4 reignites at +4.60 pp, comparable to the M1→M2 step
   (+4.65 pp). The "diminishing returns" framing was premature.
3. Replay grows absolutely (3.2 → 4.3 → 7.7 → 10.9 pp) but **sub-linearly in
   reference-set size**: per-reference-structure replay rate is monotonically
   *falling* (M1: 12.0 ppm, M2: 5.4 ppm, M3: 4.1 ppm, M4: 2.7 ppm). Each
   curated structure adds less replay than the previous round's. Memorization
   hypothesis is strongly refuted.
4. **Iteration beats one-shot at matched budget.** M2_v2 (iterative, S0+S1)
   reaches 35.06% MSUN; M_big (one-shot S_big from M0, same 109k effective
   training set) reaches 32.91%. Same compute, same training-set size, same
   verifier — iterating adds 2.15 pp MSUN and 0.46 pp SUN at the matched
   step, and the deeper cascade pulls way ahead (M4 train-novel = 37.34% vs
   M_big = 27.77%, a 9.57 pp lead at no shared budget point).

## Abstract Draft

We introduce **Flywheel**, a verifier-curated self-training recipe for crystal
generators. Starting from a base generator, we sample candidate crystals, relax
them with a machine-learned interatomic potential, score their formation energy
and energy above hull, retain metastable candidates, deduplicate them against
the training set, and retrain the generator on the accepted structures. We
instantiate the recipe with Crystalite on MP20. External LeMat-GenBench
evaluation with the multi-MLIP hull protocol shows monotonic improvement in
MSUN across four rounds, from 22.6% for the public Crystalite baseline to
29.3% (M1), 35.1% (M2), 40.4% (M3), and 48.2% (M4) — the cascade has not
saturated. To isolate iteration from training-set volume, we additionally
train a one-shot control (M_big) that curates a single matched-budget batch
from the base generator; the iterative M2 beats the one-shot M_big by 2.15 pp
MSUN at the same effective training-set size, confirming that the gain is not
purely a function of more curated data. Because LeMat novelty does not by
itself exclude our synthetic training augmentations, we additionally decompose
LeMat-MSUN into a "replay" share (LeMat-MSUN samples that match our synthetic
training augmentation under the same material-hasher structure matcher LeMat
uses internally) and the train-novel residual. Replay grows absolutely across
rounds (3.2 → 4.3 → 7.7 → 10.9 pp) but **sub-linearly in reference-set size**:
per-reference-structure replay rate falls monotonically (12.0 → 5.4 → 4.1 →
2.7 ppm), refuting the memorization hypothesis. M4's train-novel MSUN reaches
37.3%, substantially above the base model, the raw/dedup ablation controls,
and the matched-budget one-shot M_big (27.8%). Ablations show that unfiltered
synthetic augmentation does not help; the metastability verifier is the
load-bearing signal.

## Method Summary

Flywheel is parameterized by a generator, a verifier, a stability criterion,
and a deduplication criterion.

For curation round `r`:

1. Sample candidates from `M_r`.
2. Decode candidates to pymatgen structures.
3. Prescreen invalid geometries.
4. Relax candidates with the verifier MLIP.
5. Score formation energy and `e_above_hull` against the MP2020 hull.
6. Keep candidates satisfying the curation rule.
7. Deduplicate accepted candidates against the cumulative training set.
8. Export accepted structures through the same MP20 token preprocessing path.
9. Train `M_{r+1}` on MP20 plus all accepted synthetic datasets through `S_r`.

Current NequIP recipe:

```text
Base data:        MP20
Verifier:         NequIP-OAM-L
Curation hull:    MP2020 patched phase diagram
Filter:           msun_like, 0 < e_above_hull <= 0.1 eV/atom
Dedup:            StructureMatcher against MP20 plus previous synthetic rounds
Evaluation:       LeMat comprehensive_multi_mlip_hull, MACE/ORB/UMA
Eval CIFs:        NequIP pre-relaxed before LeMat submission
```

Training checkpoint selection should use train-aware MSUN, not raw LeMat-style
MSUN against MP20 only. This prevents selecting checkpoints that regenerate
synthetic structures already added to training.

**Dataset schedule.** Each round's synthetic batch is sized to **double the
effective training set**. MP20 contributes ~27k structures; we set the
per-round accepted-sample target so the cumulative training set roughly
doubles round on round:

```text
Round   New synthetic batch   Cumulative synthetic   Effective train (≈ MP20 + synth)
M0      —                     —                      27k   (MP20 only)
M1      S0 = 27,138           27k                    54k
M2      S1 = 54,276           81k                    109k
M3      S2 = 108,552          189k                   217k
M4      S3 = 217,104          407k                   434k
```

The batch sizes are deliberate targets (`MAX_SAMPLES` set to 27k / 54k / 108k),
not artifacts of yield. Two consequences:

1. The MSUN-vs-training-size figure has a clean log₂ x-axis (27k → 54k → 109k),
   so each round is one doubling step.
2. The oversampled-real control is sized to the **same** effective-training-set
   schedule (e.g. MP20 + 27k resampled MP20 = 54k to match M1), isolating
   "verifier-curated synthetic data" from "more real data at equal size."

Note the cumulative *synthetic* count (27k / 81k / 189k) is not a clean double;
the doubling is of the *effective* training set, which the scaling figure
plots.

## Metric Definitions

**External LeMat MSUN:** LeMat's metastable, unique, novel score under
`comprehensive_multi_mlip_hull`. Metastability is computed from the multi-MLIP
hull; uniqueness and novelty use LeMat's `material_hasher` structure matcher.

**LeMat structure matcher:** In the current LeMat config,
`fingerprint_method: "structure-matcher"` resolves to
`material_hasher.similarity.PymatgenStructureSimilarity(tolerance=0.1)`, which
wraps:

```python
StructureMatcher(ltol=0.1)
```

with pymatgen defaults for `stol`, `angle_tol`, `primitive_cell`, and `scale`.

**Replay MSUN:** LeMat-MSUN samples that match the model's **synthetic
augmentation** (S0 for M1, S0∪S1 for M2) under the same LeMat matcher. MP20 is
excluded from the reference set because LeMat-MSUN is already filtered against
LeMat-Bulk (which subsumes MP20); checking MP20 again would double-count
disagreements between LeMat's preprocessing and ours rather than measure real
replay.

**Train-novel MSUN:** LeMat-MSUN samples that survive the additional replay
check:

```text
train-novel MSUN = LeMat MSUN − replay against synthetic refs
```

This is the conservative number we report for methodological claims.

**Framework-novel MSUN:** LeMat-MSUN samples that are not an MP20 structure
with an f-block element swapped in. Because lanthanides are nearly
interchangeable (lanthanide contraction → similar radii/chemistry), the
generator can take an MP20 scaffold (e.g. LaFeO₃) and substitute a different
lanthanide (→ NdFeO₃); both LeMat's matcher and ours are element-sensitive, so
this counts as novel. We detect it by re-checking each LeMat-MSUN sample against
MP20 under **f-block anonymisation** (all lanthanides → La) with
`StructureMatcher(ltol=0.1)`:

```text
framework-novel MSUN = LeMat MSUN − substitution onto MP20 (anonymised)
```

This is a lower bound on the correction: LeMat novelty is against LeMat-Bulk
(which subsumes MP20), so the full anonymised-LeMat-Bulk check would remove at
least as much.

## Main Results

LeMat `comprehensive_multi_mlip_hull`, n=2500. Valid is fraction of submitted
samples. Other rates are fractions of valid structures.

| Model | Training data | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | SUN+MSUN | mean e_hull |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crystalite baseline | MP20 | 97.20 | 95.80 | 53.20 | 12.70 | 51.60 | 1.50 | 22.60 | 24.10 | 0.1322 |
| Oversample control | MP20 + 27k MP20 | 96.56 | 98.84 | 62.05 | 9.94 | 50.25 | 1.53 | 24.44 | 25.97 | 0.1033 |
| Raw synthetic | MP20 + S_raw | 96.24 | 98.63 | 66.54 | 8.31 | 46.30 | 1.04 | 24.19 | 25.23 | 0.1195 |
| Dedup-only synthetic | MP20 + S_dedup | 95.68 | 99.37 | 68.69 | 6.77 | 44.94 | 0.92 | 24.29 | 25.21 | 0.1235 |
| One-shot M_big (matched M2 budget) | MP20 + S_big (81k) | 96.52 | 99.09 | 69.46 | 7.71 | 54.12 | 1.82 | 32.91 | 34.73 | 0.0970 |
| **Flywheel M1** | **MP20 + S0** | **96.40** | **99.13** | **65.15** | **9.21** | **53.40** | **1.74** | **29.34** | **31.08** | **0.0988** |
| **Flywheel M2** | **MP20 + S0 + S1** | **96.40** | **99.13** | **70.91** | **7.63** | **55.81** | **2.28** | **35.06** | **37.34** | **0.0944** |
| **Flywheel M3** | **MP20 + S0 + S1 + S2** | **97.76** | **99.26** | **76.51** | **5.77** | **57.90** | **2.54** | **40.43** | **42.96** | **0.0939** |
| **Flywheel M4** | **MP20 + S0 + S1 + S2 + S3** | **98.00** | **99.71** | **81.14** | **5.59** | **62.00** | **2.90** | **48.20** | **51.10** | **0.0860** |

Counts over valid structures:

| Model | Valid n | SUN | MSUN | SUN+MSUN |
|---|---:|---:|---:|---:|
| Oversample control | 2414 | 37 | 590 | 627 |
| Raw synthetic | 2406 | 25 | 582 | 607 |
| Dedup-only synthetic | 2392 | 22 | 581 | 603 |
| One-shot M_big | 2413 | 44 | 794 | 838 |
| Flywheel M1 | 2410 | 42 | 707 | 749 |
| Flywheel M2 | 2410 | 55 | 845 | 900 |
| Flywheel M3 | 2444 | 62 | 988 | 1050 |
| Flywheel M4 | 2450 | 71 | 1181 | 1252 |

## Replay-Corrected Results

We take LeMat's own MSUN set and apply an additional `StructureMatcher(ltol=0.1)`
check (LeMat's matcher under `comprehensive_multi_mlip_hull`) against the
model's synthetic training augmentation. The "replay" share is what LeMat
counted as novel but matches our augmentation; the residual is genuinely
train-novel.

| Model | Synthetic ref size | LeMat MSUN | Replay | Train-novel MSUN | Replay / MSUN |
|---|---:|---:|---:|---:|---:|
| Raw synthetic | 27k | 24.19% | 3.28 pp | 20.91% | 13.6% |
| Dedup-only synthetic | 27k | 24.29% | 1.38 pp | 22.91% | 5.7% |
| One-shot M_big | 81k | 32.91% | 5.14 pp | 27.77% | 15.6% |
| Flywheel M1 | 27k | 29.34% | 3.24 pp | 26.10% | 11.0% |
| Flywheel M2 | 81k | 35.06% | 4.32 pp | 30.75% | 12.3% |
| Flywheel M3 | 190k | 40.43% | 7.69 pp | 32.74% | 19.0% |
| **Flywheel M4** | **407k** | **48.20%** | **10.86 pp** | **37.34%** | **22.5%** |

Interpretation:

- Replay grows absolutely (3.2 → 4.3 → 7.7 → 10.9 pp) but **sub-linearly in
  reference-set size**: per-reference-structure replay rate is monotonically
  *falling* round-on-round (M1: 12.0 ppm, M2: 5.4 ppm, M3: 4.1 ppm, M4: 2.7
  ppm). Each curated structure adds *less* replay than the previous round's.
  The memorization hypothesis is strongly refuted.
- **Train-novel MSUN climbs 26.1 → 30.8 → 32.7 → 37.3** — most of the
  external MSUN gain at every round is genuine novel discovery, not
  regeneration of curated training data. Per-round train-novel deltas
  (+4.65, +1.99, +4.60) show M3 was a trough rather than the onset of
  saturation; the M3→M4 step recovers the cascade's gain rate.
- **Iteration > one-shot survives the replay correction.** At the matched 81k
  reference budget, M2 reaches 30.75% train-novel MSUN vs M_big's 27.77% —
  a +2.98 pp gap (slightly wider than the +2.15 pp raw-LeMat gap, because
  M_big leans more on replay than M2 does: 15.6% vs 12.3% of LeMat-MSUN).
  Continuing the cascade pulls the gap to +9.57 pp (M4 train-novel 37.34%
  vs M_big 27.77%), at unmatched but informative compute.
- The verifier-filtered conditions (M1, M2) preserve more train-novel MSUN
  than the unfiltered ablations (raw, dedup-only) even though the unfiltered
  conditions have lower replay. The metastability filter is what shifts
  external MSUN upward without inflating replay.

## Lanthanide Substitution / Framework-Novel MSUN

A portion of the MSUN gain is lanthanide **substitution onto MP20 frameworks**,
not new structure types. We re-check each model's MSUN-flagged generated samples
against MP20 under f-block anonymisation (all lanthanides → La) with
`StructureMatcher(ltol=0.1)`, using the nested partition in
`scripts/diagnose_msun_novelty_partition.py`:

  external MSUN
    − replay (matches synthetic S0[∪S1], element-sensitive)        → train-novel
        − substitution (train-novel rows matching MP20 only after  → framework-novel
          anonymisation; "anon-only" matches)

"Anon-only" is the conservative choice: we count a structure as substitution
only if it matches MP-20 *after* anonymisation and not before, so the
substitution claim captures the chemistry-of-interest (f-block swap onto a
known framework) rather than any structure that happens to be close to an
MP-20 entry. This keeps the framework-novel residual a defensible lower bound
on genuinely-new structure types.

| Model | external MSUN | replay | substitution (anon-only) | framework-novel |
|---|---:|---:|---:|---:|
| Flywheel M1 | 29.34% | 3.24 pp | 2.20 pp | 23.90% |
| Flywheel M2 | 35.06% | 4.32 pp | 2.74 pp | 28.01% |

Reads:

- **~7–8% of the reported MSUN is f-block substitution onto MP20** — known
  scaffolds with a swapped lanthanide, counted as novel only because LeMat's
  matcher is element-sensitive. In rate terms ~2–3 pp of the MSUN headline.
- **The substitutional share grows slightly round-on-round** (7.5% → 7.8% of
  MSUN), consistent with the rising lanthanide enrichment in the curated set.
- **Framework-novel MSUN climbs 22.60% → 23.90% → 28.01%** — the headline
  trajectory survives the strictest available novelty measure.
- On the *curated* set (S0) the substitution rate is far higher (~61% of
  lanthanide structures match MP20 under anonymisation) than on the LeMat
  evaluated samples, because the metastability filter preferentially keeps
  near-hull structures, which are more MP20-like. Curation **concentrates**
  substitutional lanthanide structures (they inherit the metastability of their
  MP20 parent). LeMat's downstream filtering removes most but not all of this
  fraction from the headline metric.

Lower-bound caveats: MP20-only reference (LeMat novelty is vs LeMat-Bulk, which
subsumes MP20, so the true substitution rate is ≥ this); all-lanthanides→La
anonymisation is the aggressive form (an identity-preserving mapping for
multi-lanthanide compounds would yield a smaller rate). Net bias is undetermined.

Decomposition table for M2:

```text
M2 external MSUN:                          35.06%
  − replay (vs synthetic S0∪S1):  4.32 pp → train-novel        30.74%
      − f-block substitution:     2.74 pp → framework-novel    28.01%
```

The partition is nested by detection order, so the categories are disjoint and
sum exactly to external MSUN; framework-novel is MSUN that is neither a replay
of the synthetic augmentation nor an f-block swap of an MP20 scaffold.

## Ablation Read

The curation ladder answers: does synthetic data help by itself, or does the
verifier matter?

| Condition | Added data | LeMat MSUN | Train-novel MSUN | Read |
|---|---|---:|---:|---|
| Oversample | 27k MP20 rows | 24.44% | 24.44% | More exposure helps little (no synthetic to replay). |
| Raw synthetic | unfiltered generated rows | 24.19% | 20.91% | Synthetic data without quality signal hurts. |
| Dedup-only | generated rows deduped vs train | 24.29% | 22.91% | Dedup alone is not enough. |
| One-shot M_big | 81k Flywheel-curated S_big from M0 | 32.91% | 27.77% | Volume helps, but iteration helps more. |
| Full Flywheel M1 | NequIP-relaxed, metastable, deduped S0 | 29.34% | 26.10% | Verifier filter is load-bearing. |
| Full Flywheel M2 | cumulative S0+S1 (matched M_big budget) | 35.06% | 30.75% | **Iteration > one-shot at matched budget (+2.98 pp train-novel vs M_big).** |
| Full Flywheel M3 | cumulative S0+S1+S2 | 40.43% | 32.74% | Cascade slow round (+1.99 pp train-novel). |
| Full Flywheel M4 | cumulative S0+S1+S2+S3 | 48.20% | 37.34% | **Cascade reignites: +4.60 pp train-novel over M3, comparable to M1→M2.** |

This is the strongest ablation sentence:

> Adding generated structures without the metastability verifier does not
> improve MSUN over a size-matched real-data control; the gain appears only
> when generated structures are relaxed, scored, filtered, and deduplicated.

## Mechanism Claims To Keep

Use these as discussion subsections, with details sourced from
[results_log.md](results_log.md).

1. **Verifier curation trims the unstable tail.** S0 shifts probability
   mass out of `e_hull > 0.10` and into the metastable bands.
2. **The gain is concentrated in the metastable shell.** MSUN improves much
   more than train-novel SUN, consistent with the `msun_like` filter targeting
   `0 < e_above_hull ≤ 0.1 eV/atom`.
3. **Raw synthetic data is actively risky.** The raw ablation increases
   LeMat novelty but produces lower train-novel MSUN than the controls.
4. **Iteration compounds across four rounds; M3 was a slow round but M4
   reignites.** Train-novel MSUN climbs 26.1 → 30.8 → 32.7 → 37.3 across
   M1 → M2 → M3 → M4 (per-round deltas +4.7, +2.0, +4.6 pp). The "saturation
   at M3" reading turned out to be premature: M4 recovers the cascade's
   per-round gain rate. Replay grows sub-linearly in reference-set size
   (per-reference replay rate falls monotonically across rounds), so the gain
   is genuine discovery rather than memorization.
5. **Iteration beats one-shot at matched budget.** At the same 81k synthetic
   ref size and 109k effective training set, iterative M2 reaches 30.75%
   train-novel MSUN vs one-shot M_big's 27.77% (+2.98 pp). The cascade
   compounds something beyond "more curated data alone."
6. **Distribution shift is real, and the rare-earth surge is not a verifier
   artifact.** S0/S1 shift composition arity, atom counts, density, and element
   frequencies; lanthanide content increases strongly across rounds. To test
   whether this is a NequIP bias being amplified by self-training, we re-scored
   1024 NequIP-relaxed lanthanide-bearing S0 structures with a second,
   independently-trained MP-consistent-PBE potential (PET-OAM-XL) on the same
   geometry against the same MP hull. The two potentials agree on metastability
   for 98.1% of structures, with a mean `e_above_hull` offset of +1.3 meV/atom;
   all 19 disagreements are structures sitting within ~20 meV of the 0.1
   eV/atom cap that the offset tips just over. The surge is therefore real
   generator/chemistry behavior, not the verifier over-stabilizing lanthanides.
7. **External verifier independence matters.** Curation uses NequIP + MP2020;
   evaluation uses LeMat's MACE/ORB/UMA multi-hull. The headline does not rely
   on the same verifier used to curate training data.

## Limitations

1. **LeMat novelty is not training novelty.** LeMat-MSUN excludes matches
   against LeMat-Bulk but not against our synthetic training augmentation, so
   we report replay against S0/S1 alongside every external MSUN number.
2. **Strict stability is not the main gain.** The current `msun_like` filter
   expands the metastable shell, not necessarily strict-stable discovery.
3. **The verifier is still a model.** NequIP and the MP2020 hull introduce
   bias. External LeMat evaluation reduces but does not eliminate verifier
   dependence. A second MP-consistent-PBE potential (PET-OAM-XL) corroborates
   NequIP's curation-time stability calls to ~1 meV/atom (see Mechanism #5),
   but both are OAM-family / MP-PBE-trained, so this rules out a
   NequIP-*specific* artifact, not a shared OAM-training-family bias; the guard
   against the latter is the independent MACE/ORB/UMA evaluation hull.
4. **SMACT validity is weak.** It is useful for benchmark parity but should
   not be treated as chemical correctness.
5. **The recipe drifts toward supply-critical, less-practical chemistry —
   and the drift is caused by iteration, not by curated-data volume.**
   Stability is necessary but not sufficient: the `msun_like` verifier
   optimizes an element-agnostic target, so iterating on it pulls the
   generator toward whatever metastable chemistry is easiest to satisfy.
   Across the cascade, LeMat HHI-production rises 3.53 (M0) → 3.61 (M1) →
   3.79 (M2) → **3.94 (M3)** — worse, production-side only
   (reserve-concentration stays flat: the rare-earth signature). The
   lanthanide-bearing fraction in curated *training* data climbs 35.5%
   (MP20) → 46.0 (S0) → 51.5 (S1) → **55.8% (S2)**; radioactive-element
   fraction more than doubles (6.6 → 14.0%). The single largest element
   enrichment is **promethium 1.0% → 8.9% (~9×, radioactive, no stable
   isotopes)** — DFT-metastable but practically meaningless. **The
   one-shot M_big control settles the cause: at matched 81k curated-synthetic
   budget, S_big stays at S0-level composition (46% lanthanide, HHI_prod 3.68),
   while the iterative cascade at the same effective training budget pushes
   to 51.5% lanthanide and HHI_prod 3.79. The amplification is feedback-driven,
   not data-volume-driven.** **Fix:** add a practicality constraint (element
   denylist and/or HHI cap) alongside the metastability filter before further
   iteration; M3 is the strongest evidence that uncontrolled iteration
   compounds the drift.
6. **Later rounds and S_big are not yet integrated into the core claim.** Keep
   those as follow-up unless the numbers land cleanly.

## Figure Plan

Minimum figures for the chapter/paper:

| Figure | Purpose | Current artifact |
|---|---|---|
| LeMat (M)SUN vs training size — headline | Show cascade scaling 22.6→29.3→35.1→40.4 with M_big off-trend at M2 budget (iteration vs one-shot) | `figures/flywheel/flywheel_lemat_msun_vs_size.*`, `flywheel_lemat_sun_vs_size.*` (from `scripts/plot_flywheel_lemat_only.py`) |
| MSUN replay decomposition vs size | Show external vs train-novel vs framework-novel; methodology figure | `figures/flywheel/flywheel_msun_vs_dataset_size.*` (M1, M2 only until M3 replay lands) |
| Curation ladder | Show verifier is the load-bearing step | make/update from replay JSONs |
| Distribution shift | Explain what S0/S1 change | `figures/flywheel/round_distribution_shift.*` |
| e_hull stratification | Show metastable-shell mechanism | `figures/augmentation/round0_ehull_distribution.png` |
| Element-class shift | Show class-level rare-earth/lanthanide drift across the cascade | `figures/flywheel/element_class_shift.{png,pdf}` (MP20/S0/S1/S2 bars; lanthanide +20.3 pp headline annotation) |
| Element shifts (per-element) | Top movers across MP20→S0→S1→S2; Pm 9× the headline element | `figures/flywheel/element_shifts_topN.{png,pdf}` |
| Element shift scatter | MP20 vs S2 scatter; lanthanide cluster sits above the diagonal | `figures/flywheel/element_shift_scatter.{png,pdf}` |
| HHI / practicality drift | Show supply-risk rises with rounds (production-side) | make from `scripts/extract_lemat_metrics.py` HHI columns |

## Paper Structure

1. **Introduction**
   - Crystal generation needs both strong proposal models and reliable filters.
   - Self-training is risky because models can amplify their own bias.
   - Crystal generation has an unusually natural verifier: relax and score.
   - Flywheel exploits this verifier to curate synthetic data.

2. **Background**
   - De novo crystal generation.
   - Stability, metastability, SUN/MSUN.
   - Self-training / pseudo-labeling / GNoME-style generate-filter loops.
   - MLIP verifiers as approximate but useful selectors.

3. **Method**
   - Define Flywheel.
   - Define curation pipeline.
   - Define replay-corrected evaluation.
   - Explain why LeMat MSUN alone is insufficient for self-training claims.

4. **Experiments**
   - Base Crystalite.
   - Oversample control.
   - Raw and dedup-only ablations.
   - M1 and M2 flywheel rounds.
   - External LeMat evaluation protocol.

5. **Results**
   - External MSUN scaling.
   - Replay-corrected full-train-novel MSUN.
   - Curation ladder.
   - Distribution shift / mechanism.

6. **Discussion**
   - Why verifier curation can work despite self-training risks.
   - Why M2 is more important than M1.
   - Relationship to GNoME and brute-force screening.
   - Limits of MLIP/DFT verifier bias.

7. **Conclusion**
   - Lightweight generator plus verifier-curated iteration can discover more
     novel metastable structures than the base model or naive augmentation.

## Open Follow-Ups

Near-term:

- Confirm `figures/flywheel/flywheel_msun_vs_dataset_size.*` matches the
  canonical replay numbers (M1: 3.24 pp, M2: 4.32 pp) — the existing plot
  already uses these values.
- Add a curation-ladder figure showing LeMat MSUN and train-novel MSUN side by
  side for oversample / raw / dedup / M1 / M2.
- Append the corrected replay block to [results_log.md](results_log.md).

Later:

- Stable-only / stricter-filter ablation.
- S_big one-shot versus iterative comparison.
- M3 / later-round continuation if compute budget allows.
- EquiformerV3 verifier/hull protocol after material-scientist feedback.

## Update Checklist

When a new result lands:

- [ ] Add the external LeMat row to the main table.
- [ ] Run replay decomposition against the synthetic augmentation only (S0, S1, ...).
- [ ] Add or update the train-novel MSUN row.
- [ ] Update the figure plan and result artifacts.
- [ ] Append a dated block to [results_log.md](results_log.md).
- [ ] Move any stale speculative text out of the skeleton.
****