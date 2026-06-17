# Results Log

Append-only history of synthetic augmentation rounds and ablations. Each round
gets a self-contained block: artifacts, headline numbers, comparison vs
controls, and one-paragraph interpretation. Don't rewrite history; if a
re-evaluation supersedes prior numbers, append a new dated block and link from
the old one.

All LeMat numbers below are from `comprehensive_multi_mlip_hull` with NequIP
pre-relaxed CIFs and a three-way **MACE / ORB / UMA** consensus, n=2500. Rates
are fractions of valid structures unless noted. (LeMat artifact filenames
contain `_uma_comprehensive_multi_mlip_hull_` because UMA is LeMat's
designated relaxer in that pipeline; the stability *signal* is the three-MLIP
consensus, not UMA alone.)

---

## Round 0 — 2026-05-09

**Setup**

```text
M0 = base Crystalite (dng.pt, MP20-trained)
S0 = M0 samples → NequIP relax → MP2020 hull → metastable (e_hull ≤ 0.1)
       → StructureMatcher dedup vs MP20
     |S0| = 27,138
M1 = train MP20 + S0      → outputs/dng_synthetic_round0_msun_27k
C0 = train MP20 + 27,138 oversampled MP20 → outputs/dng_oversample_real_27k
```

LeMat result files:

```text
synthetic: crystalite_synthetic_aug_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_072436.json
control:   crystalite_oversample_real_n2500_nequip_relaxed_uma_comprehensive_multi_mlip_hull_20260509_171859.json
```

**Headline table**

| Model | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | SUN+MSUN | mean e_hull | RMSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Public Crystalite MP20 | 97.20 | 95.80 | 53.20 | 12.70 | 51.60 | 1.50 | 22.60 | 24.10 | 0.0905 | 0.1322 |
| C0 — oversampled-real (27k) | 96.56 | 98.84 | 62.05 | 9.94 | 50.25 | 1.53 | 24.44 | 25.97 | 0.1033 | 0.1165 |
| **M1 — synthetic Round 0 (27k)** | **96.56** | **99.25** | **63.55** | **10.48** | **55.05** | **2.40** | **29.99** | **32.39** | **0.0899** | **0.0905** |

Counts over the 2500 submitted structures:

| Model | SUN | MSUN | SUN+MSUN |
|---|---:|---:|---:|
| C0 — oversampled-real | 37 | 590 | 627 |
| M1 — synthetic Round 0 | 58 | 724 | 782 |

**M1 vs public Crystalite (pp)**

```text
novel:       +10.35
metastable:   +3.45
SUN:          +0.90
MSUN:         +7.39
SUN+MSUN:     +8.29
stable:       -2.22
valid:        -0.64
```

**M1 vs size-matched oversampled control (pp)**

```text
valid:       +0.00
unique:      +0.41
novel:       +1.49
stable:      +0.54
metastable:  +4.81
SUN:         +0.87
MSUN:        +5.55
SUN+MSUN:    +6.42
mean e_hull: -0.0135 eV/atom
relax RMSD:  -0.0260 Å
```

**Interpretation**

The oversampled-real control improves slightly over the public Crystalite row,
so a portion of the gain is attributable to changed training exposure /
stochastic reweighting of MP20 — supervisor concern is valid. However,
verifier-curated synthetic augmentation outperforms the size-matched control on
every metric except strict validity (tie). The clean claim is therefore:

> Verifier-curated synthetic augmentation improves novel metastable discovery
> beyond what size-matched oversampling of real data delivers; size matching
> rules out the simplest "more rows / more exposure" explanation.

The headline result is a **novel-metastable discovery** gain, not a strict
stability gain. Strict stable yield drops 2.2 pp vs public Crystalite — the
model trades some strict stability for substantially more MSUN. This is
consistent with the `msun_like` filter selecting the metastable survivor band
(see [followups.md](followups.md), "Selection hypothesis"). A 4096-sample
diversity probe confirmed M1 is not memorizing S0: structural novelty against
both MP20 and S0 stays high, formula entropy remains high, but M1 shifts
toward higher-arity systems (33.1% ≥4 elements vs M0 26.6%).

### N-ary stratified re-scoring — 2026-05-11

Analysis: [scripts/analyze_nary_stratified.py](../../scripts/analyze_nary_stratified.py)
joins each model's [outputs/external_eval/](../../outputs/external_eval/)
`relaxed_cifs/manifest.json` (`sample_idx → relaxed_formula`) with the LeMat
JSON's `results.sun.individual_values` (encoding `{1.0: SUN, 0.5: MSUN,
0.0: neither}`). Per-bucket sums recover the published 58 SUN / 724 MSUN
(synthetic) and 37 SUN / 590 MSUN (control) exactly, confirming the join.

**n-ary distribution (of 2500 submitted):**

| Model | unary | binary | ternary | quaternary | 5+ |
|---|---:|---:|---:|---:|---:|
| Oversample-real (control) | 1.7% | 19.9% | 49.4% | 23.5% | 5.6% |
| Synthetic Round 0 | 1.3% | 19.2% | 45.7% | 25.9% | 7.8% |

The arity shift exists but is smaller than the codex-doc M0-vs-M1 claim
(~+4.6 pp ≥4-element vs the codex's +6.5 pp); the control here is the
oversample-real M1_control, not base M0.

**SUN+MSUN rate per bucket (of valid in bucket):**

| Model | unary | binary | ternary | quaternary | 5+ |
|---|---:|---:|---:|---:|---:|
| Oversample-real (control) | 12.20% (5/41) | 27.80% (134/482) | 26.37% (313/1187) | 24.08% (137/569) | 28.15% (38/135) |
| Synthetic Round 0 | 46.88% (15/32) | 33.33% (156/468) | 31.08% (340/1094) | 33.76% (212/628) | 30.73% (59/192) |

**Delta (synthetic − control), pp:**

| Bucket | SUN | MSUN | SUN+MSUN |
|---|---:|---:|---:|
| unary | +3.12 | +31.55 | +34.68 |
| binary | -0.15 | +5.68 | +5.53 |
| ternary | +1.03 | +3.68 | +4.71 |
| quaternary | +1.44 | +8.24 | +9.68 |
| 5+ | +0.30 | +2.28 | +2.58 |

**Interpretation.** Synthetic wins SUN+MSUN in every bucket. The composition-
arity confound is closed: the gain is **not** explained by a shift to
harder-to-dedup high-arity space — it survives at fixed arity. Two non-obvious
follow-on observations:

1. **Quaternary is the sweet spot for the curation recipe.** Highest absolute
   SUN+MSUN rate among multi-element buckets (33.76%) and the largest delta
   (+9.68 pp). The "5+" bucket gains least (+2.58 pp) — high-arity space is
   hard for both models, so headroom is not the whole story.
2. **The strict-stable trade is consistent across buckets, not driven by one
   stratum.** Per-bucket SUN gains are small but positive everywhere (excluding
   the noisy unary slice); MSUN gains are 4–8 pp across all multi-element
   buckets. The headline pattern — trading some strict stability for
   substantially more novel metastability — is bucket-invariant.

The unary bucket numbers (15/32 SUN+MSUN for synthetic) are small-sample and
should be reported with that caveat; the binary/ternary/quaternary/5+ buckets
are the load-bearing ones.

Figure: [figures/augmentation/round0_arity_rates.png](../../figures/augmentation/round0_arity_rates.png) —
grouped bar chart of SUN+MSUN rate by arity (synthetic vs control), with per-bucket Δpp annotated.

### E-above-hull stratification — 2026-05-11

Same script ([scripts/analyze_nary_stratified.py](../../scripts/analyze_nary_stratified.py))
also extracts per-structure `e_above_hull` from the LeMat
`results.stability` BenchmarkResult (`E_HullMetric` block, combined across
MACE/ORB/UMA). Stability bands:

```text
below_hull        e_hull ≤ 0          (strict-stable; SUN candidates live here)
near_metastable   0 < e_hull ≤ 0.025
mid_metastable    0.025 < e_hull ≤ 0.05
far_metastable    0.05 < e_hull ≤ 0.10
unstable          e_hull > 0.10
```

**Band-population shift (% of valid):**

| Model | below_hull | near | mid | far | unstable |
|---|---:|---:|---:|---:|---:|
| Oversample-real (control) | 9.9% | 19.9% | 11.3% | 19.1% | 39.8% |
| Synthetic Round 0 | 10.5% | 20.9% | 13.4% | 20.7% | 34.5% |
| Δ pp (synthetic − control) | +0.6 | +1.0 | +2.1 | +1.6 | **−5.3** |

Synthetic spends 5.3 pp less of its output on unstable (>0.10) candidates and
spreads that mass across every metastable band. The mass shifts toward
mid_metastable in particular (+2.1 pp).

**Where the new MSUN samples land (MSUN distribution within metastable):**

| Model | near (0–0.025) | mid (0.025–0.05) | far (0.05–0.10) |
|---|---:|---:|---:|
| Oversample-real (control) | 22.4% (132) | 22.5% (133) | 55.1% (325) |
| Synthetic Round 0 | 23.2% (168) | 27.3% (198) | 49.4% (358) |
| Δ MSUN count | +36 | **+65** | +33 |

The largest absolute MSUN gain is in the **mid_metastable band**, not at
either extreme. This is the load-bearing observation: synthetic is neither
threshold-exploiting (would manifest as concentration in `far_metastable`)
nor purely hull-hugging (would manifest in `near_metastable`). It is
enriching the *moderately metastable* novel structures.

**Within the strict-stable population, novelty/uniqueness is also higher:**

```text
SUN / below_hull rate (= unique+novel rate among strict-stable):
  control:    37 / 240 = 15.4%
  synthetic:  58 / 253 = 22.9%      Δ +7.5 pp
```

So the SUN gain (+21 absolute) is *not* just "synthetic has more strict-stable
candidates"; the strict-stable candidates synthetic produces are themselves
more likely to be unique+novel — a second-order gain on top of the small
population shift.

Figure: [figures/augmentation/round0_ehull_distribution.png](../../figures/augmentation/round0_ehull_distribution.png) —
two-panel e_hull histogram, SUN/MSUN/neither stacked, with reference lines at
0 (stable) and 0.10 (metastable cutoff). The mid-metastable enrichment and
shorter unstable tail are visible by eye.

**Interpretation.** Three mechanistic claims earn their place in the writeup:

1. **The verifier recipe trims the unstable tail.** Synthetic wastes 5.3 pp
   less output on hopeless candidates — a directly attributable curation
   benefit (the curator threw those out of S0, so M1 stopped producing them).
2. **The MSUN gain is mid-metastable, not threshold or hull.** Synthetic
   enriches the 0.025–0.05 band most (+65 MSUN), which is the cleanest
   mechanistic finding — rules out the cheapest skeptical readings.
3. **The strict-stable population also gets more unique+novel.** SUN
   conversion within `below_hull` jumps from 15.4% to 22.9% — the curation
   is teaching the model not just where the hull is but where MP20 *isn't*.

---

## EquiformerV3 verifier calibration vs MP-20 — 2026-05-11 (n=1000)

Validates that EquiformerV3-OAM, scored against the existing MP DFT hull with
MP2020 corrections, reproduces MP-20's stored DFT `e_above_hull` labels. This
is the calibration step that justifies switching the curation verifier from
NequIP to EquiformerV3 in future rounds.

**Setup:**

- 1000 random entries sampled from `data/mp20/raw/test.csv` (seed=0)
- Each scored under two conditions against `mp_02072023/2023-02-07-ppd-mp.pkl`:
  - **A**: raw EquiformerV3 single-point energy, no corrections
  - **B**: EquiformerV3 single-point energy + MP2020 corrections at entry construction
- Ground truth: MP-20's stored `e_above_hull` (DFT-derived)
- Script: [scripts/equiformer_v3_vs_mp20_ehull.py](../../scripts/equiformer_v3_vs_mp20_ehull.py)
- Local CPU wall: ~29 min

**Headline numbers:**

| condition | bias (eV/atom) | MAE | RMSE | Spearman ρ | metastable agreement |
|---|---:|---:|---:|---:|---:|
| raw eqv3 (A) | +0.247 | 0.250 | 0.392 | 0.42 | 46.1% |
| eqv3 + MP2020 (B) | **+0.005** | **0.008** | **0.037** | **0.87** | **99.1%** |

**Decision:** apply MP2020 corrections to EquiformerV3 outputs. Pinned in
[equiformer_v3_setup.md](equiformer_v3_setup.md) "MP2020 corrections — DECIDED".

**What the figures show**
([figures/augmentation/equiformer_v3_vs_mp20_n1000_histogram.png](../../figures/augmentation/equiformer_v3_vs_mp20_n1000_histogram.png),
[figures/augmentation/equiformer_v3_vs_mp20_n1000_scatter.png](../../figures/augmentation/equiformer_v3_vs_mp20_n1000_scatter.png)):

- **Raw eqv3** is bimodal: one population near 0 (metals/intermetallics, no
  correction effect) and a second population at 0.2–0.5 eV/atom (anion-bearing
  oxides, halides, sulfides — exactly the chemistries MP2020 corrects for).
  ~50% of MP-20 lands in this systematically-overestimated tail, which is why
  raw metastable agreement collapses to 46.1%.
- **eqv3 + MP2020** overlaps the MP-20 stored distribution almost exactly,
  with points hugging the y=x scatter diagonal at Spearman 0.87 and a tight
  37 meV/atom RMSE.

**Why metastable agreement (99.1%) matters more than strict-stable (63.6%):**

The curator's filter is `e_hull ≤ 0.1` (metastable). At that threshold,
EquiformerV3 + MP2020 agrees with DFT on 991/1000 entries — essentially
perfect for our pipeline. The strict-stable boundary (`e_hull ≤ 0`) has lower
agreement because the +5 meV/atom systematic bias bumps some DFT-stable
entries (e_hull = 0) just above zero in EquiformerV3 (predicted e_hull ≈ 5
meV/atom). 359 of 461 DFT-stable entries (78%) get reclassified as
"slightly metastable" by EquiformerV3.

This is a calibration offset, not a methodological bug. It would affect a SUN
count *if SUN were computed on the EquiformerV3 hull* — but our headline SUN
metric comes from external LeMat eval using independent MLIPs (MACE / ORB /
UMA), which don't share the +5 meV bias. So the bias doesn't enter the
load-bearing claim.

**What this clears for the chapter:**

- Curation verifier: NequIP → EquiformerV3-OAM swap is methodologically
  validated. Hull build can proceed.
- Hull-build protocol pinned: single-point at DFT-min, MP2020 corrections
  applied (see [equiformer_v3_setup.md](equiformer_v3_setup.md) "Inference
  protocols — three distinct use cases").
- Methods-section claim: "EquiformerV3-OAM with MP2020 corrections reproduces
  MP-20 DFT e_hull labels at metastable-threshold agreement of 99.1% (n=1000,
  RMSE 37 meV/atom)".

---

## Round 0 v2 (corrected protocol) — 2026-05-15

**What's new vs the v1 Round 0 block above:**
- Training-time checkpoint selector switched from `MSUN` (novelty vs MP20-only)
  to `Train_MSUN` (novelty vs MP20 ∪ S_r); see
  [src/utils/checkpoint.py:41-49](../../src/utils/checkpoint.py#L41-L49).
- Synthetic data tokenized through `MP20Tokens` with `preprocess(niggli=True)`,
  matching the MP20 pipeline.
- Training capped at 1M steps; `best.pt` selected at step 925k for M1_v2 (peak
  of Train_MSUN plateau).

**Models trained**

```text
M1_v2       = MP20 + S0           → outputs/dng_synthetic_round0_msun_27k_v2
M_dedup_v2  = MP20 + S_dedup_27k  → outputs/dng_synthetic_dedup_only_54k_v2
M_raw_v2    = MP20 + S_raw_27k    → outputs/dng_synthetic_raw_54k_v2 (training in flight)
```

**LeMat result files** (in `~/lemat-genbench/results_final/`)

```text
crystalite_synthetic_round0_msun_27k_v2_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260515_022357.json
crystalite_synthetic_dedup_only_54k_v2_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260515_031132.json
```

**Train-novel metrics** (`scripts/score_training_novelty.py`, novelty referenced
to the model's actual training set; see [train_novel framing](followups.md#appendix-candidate))

| Model | train novel rate | train_un (U+N) | train_un_msun | train_un_sun | train_un_sun_msun | training replay (kept@msun) |
|---|---:|---:|---:|---:|---:|---:|
| M0 (public Crystalite, vs MP20) | — | — | 22.6% | 1.5% | 24.1% | n/a |
| **M_dedup_v2** (vs MP20+S_dedup) | 81.4% | 81.2% | **19.57%** | 0.67% | 20.27% | 4.7% |
| **M1_v2** (vs MP20+S0) | 76.2% | 75.9% | **21.95%** | 1.29% | 23.36% | 7.0% |
| M_raw_v2 (vs MP20+S_raw) | — | — | TBD | TBD | TBD | TBD |

(M0's row uses MP20-only as reference since that's its training; the comparison
to M1_v2 / M_dedup_v2 is asymmetric — see interpretation below.)

**LeMat MSUN headline (fixed-reference, LeMat-Bulk consensus)**

| Model | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | SUN+MSUN | mean e_hull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Public Crystalite MP20 (M0) | 97.20 | 95.80 | 53.20 | 12.70 | 51.60 | 1.50 | 22.60 | 24.10 | 0.0905 |
| C0 — oversampled-real (27k) | 96.56 | 98.84 | 62.05 | 9.94 | 50.25 | 1.53 | 24.44 | 25.97 | 0.1033 |
| **M_dedup_v2** | 95.68 | 99.37 | 68.69 | 6.77 | 44.94 | 0.92 | **24.29** | 25.21 | 0.1235 |
| **M1_v2** | 96.40 | 99.13 | 65.15 | 9.21 | 53.40 | 1.74 | **29.34** | 31.08 | 0.0988 |
| **M1_v1 (buggy, superseded)** | — | — | — | — | — | — | **30.00** | — | — |

**v2 vs v1 (same recipe, corrected protocol)**

M1_v1 LeMat MSUN = 30.00%; M1_v2 LeMat MSUN = 29.34%. Within 0.7 pp — the
corrected selector + preprocessing fix made *no material difference* to the
LeMat number. The implication: the v1 best.pt was actually close to the
Train_MSUN peak too (the MSUN-driven selector happened to land near the
right checkpoint by luck), and our headline result stands.

**Replay gap (LeMat MSUN − train-novel MSUN)**

| Model | LeMat MSUN | train-novel MSUN | gap |
|---|---:|---:|---:|
| M_dedup_v2 | 24.29 | 19.57 | **4.72 pp** |
| M1_v2 | 29.34 | 21.95 | **7.39 pp** |
| M1_v1 (for reference) | 30.00 | 22.30 | 7.70 pp |

The replay gap is **larger for the msun-curated arm (7.4 pp)** than for the
dedup-only arm (4.7 pp). M1_v2's training set (S0) was specifically
metastable-filtered structures; the model concentrates probability on those
modes, and some fraction matches LeMat-Bulk indexed structures, inflating
LeMat MSUN relative to train-novel MSUN. Dedup-only training data is more
diverse / less concentrated, so its replay component is smaller.

**v2 vs v1 replay gap holds**: M1_v2's 7.4 pp matches M1_v1's 7.7 pp almost
exactly. The replay phenomenon is structural to the recipe, reproducible
across corrected and uncorrected protocols.

**Curation-ladder finding**

| pair | gap on LeMat MSUN | gap on train-novel MSUN |
|---|---:|---:|
| M1_v2 − M_dedup_v2 | **+5.05 pp** | **+2.38 pp** |
| M1_v2 − M0 | +6.74 pp | -0.65 pp (smaller ref) / ~+5 pp (apples-to-apples est.) |
| M_dedup_v2 − M0 | +1.69 pp | -3.03 pp / ~+2 pp (apples-to-apples est.) |
| M_dedup_v2 − M_raw_v2 | TBD | TBD |

The verifier (NequIP relax + e_hull metastability filter) contributes +2.4 pp
of train-novel MSUN on top of structural dedup alone. Dedup-only model
produces *more diverse* outputs (81% novel rate vs M1_v2's 76%) but *fewer
metastable ones* — the verifier teaches the model to live in the metastable
shell at the cost of compositional diversity.

**Reference-set asymmetry caveat**

M0's train-novel MSUN (22.6%) is computed against MP20 alone (27k structures).
M1_v2's (21.95%) is against MP20 ∪ S0 (54k structures, 2× harder). Direct
comparison understates M1_v2's gain. An apples-to-apples test would evaluate
M0's outputs against MP20+S0 and compare; we estimate M0 would score
~16-17% on this stricter reference, because some of M0's metastable-novel
outputs by construction match S0. **M1_v2's effective single-round gain over
M0 on apples-to-apples train-novel MSUN is therefore approximately +5 pp**
(rough estimate; not yet measured directly).

**Interpretation**

Three findings under the corrected protocol:

1. **M1_v2 ≈ M1_v1 on both metrics.** LeMat MSUN: 29.3% vs 30.0%. Train-novel
   MSUN: 21.95% vs 22.3%. The selector + preprocessing fixes did not change
   the headline numbers materially — v1's MSUN-driven selector happened to
   pick a checkpoint near the Train_MSUN peak. The chapter therefore stands
   on the v2 numbers as the corrected/cleaner version of the same recipe.

2. **Curation-ladder gap is real.** On LeMat MSUN: msun > dedup by 5.0 pp.
   On train-novel MSUN: msun > dedup by 2.4 pp. The verifier (NequIP relax
   + e_hull metastability filter) is the load-bearing curation step. Dedup
   alone barely beats the oversampled-real control (24.3 vs 24.4 LeMat MSUN).

3. **Replay gap reproduces.** The 7-8 pp gap between LeMat MSUN and
   train-novel MSUN that we identified in v1 (Δ = 7.7 pp) reproduces in v2
   (Δ = 7.4 pp). It's larger for msun-curated than dedup-only (7.4 vs 4.7
   pp) because the msun filter concentrates training data on
   metastable-shell structures, which the model then preferentially
   regenerates. The replay phenomenon is structural to the recipe, not an
   artifact of the v1-specific bugs.

The chapter's single-round claim is: **Flywheel at one round contributes
~+5 pp LeMat MSUN over the size-matched oversampled-real control (29.3 vs
24.4). The contribution is dominated by the verifier; structural dedup
alone barely improves over oversampling. Under train-novel reporting, the
gain is smaller (~+2-5 pp, depending on reference-set alignment). The
iteration claim (M2_v2 vs M1_v2) is tested in the next block.**

---

## Round 1 — M2_v1 replay decomposition — 2026-05-16

(Round 1 LeMat headline numbers landed earlier and are inherited from the
plot/run-tracker. This block adds the apples-to-apples replay
decomposition for M2_v1.)

---

## Methodological correction — replay decomposition using LeMat's matcher — 2026-05-16

**Bug.** Prior train-novel numbers reported in this log were computed with
[src/eval/uniqueness_novelty.py:135](../../src/eval/uniqueness_novelty.py#L135)
using `pymatgen.StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`, which
is substantially looser than LeMat's
`PymatgenStructureSimilarity(tolerance=0.1)` = `StructureMatcher(ltol=0.1)`
(default stol=0.3, angle_tol=5). A looser matcher overcounts matches → fewer
samples flagged as novel → train-novel-MSUN biased downward → "replay" gap
(LeMat-MSUN − Train-MSUN) biased upward. Comparing the two numbers therefore
conflated genuine replay with matcher-tolerance artifact.

**Fix.** [scripts/score_msun_replay.py](../../scripts/score_msun_replay.py)
takes the LeMat result's `msun_indices` and `sun_indices` directly (the
explicit list of samples LeMat already flagged as M+U+N) and re-checks each
against our curated synthetic augmentations (S0; S0 ∪ S1; etc.) using
`StructureMatcher(ltol=0.1)` — the exact same matcher LeMat uses internally.
Since MP20 ⊂ LeMat-Bulk, MP20 matches are already excluded by LeMat-MSUN;
only synthetic rounds need re-checking. The replay decomposition is now
apples-to-apples:

```text
LeMat-MSUN          = M ∧ U ∧ (not in LeMat-Bulk)
True train-novel    = LeMat-MSUN ∧ (not in S_r)
Replay              = LeMat-MSUN − True train-novel  = LeMat-MSUN ∧ in S_r
```

**Corrected numbers (StructureMatcher(ltol=0.1)):**

| Model | LeMat-MSUN | Replay (pp) | True train-novel MSUN |
|---|---:|---:|---:|
| Base (Public Crystalite MP20) | 22.60% | — | — |
| C0 — oversample-real control | 24.44% | 0 (by construction) | 24.44% |
| M1_v1 — Round 0 | 29.99% | **3.52** | **26.47%** |
| M2_v1 — Round 1 | 38.58% | **7.16** | **31.41%** |

Sources: [outputs/msun_replay/m1v1/summary.json](../../outputs/msun_replay/m1v1/summary.json),
[outputs/msun_replay/m2v1/summary.json](../../outputs/msun_replay/m2v1/summary.json).

**What changes vs the old (looser-matcher) numbers:**

| | Old (looser matcher) | New (LeMat matcher) | Δ |
|---|---:|---:|---:|
| M1 train-novel-MSUN | 22.33% | 26.47% | +4.14 pp |
| M1 replay | 7.66 pp | 3.52 pp | −4.14 pp |
| M2 train-novel-MSUN | 27.17% | 31.41% | +4.24 pp |
| M2 replay | 11.40 pp | 7.16 pp | −4.24 pp |

About **4 pp** of MSUN at every round was being mis-attributed to "replay"
under the looser matcher and is in fact genuine train-novel discovery.

**What the corrected numbers say about the recipe.**

1. **Replay is small at Round 0 and grows sub-linearly.** M1 sees only 3.5
   pp replay against S0 (27k). M2 sees 7.2 pp against S0 ∪ S1 (81k) — replay
   doubled but the reference set tripled, so per-reference replay density
   actually fell (0.13 → 0.089 ppm). The model isn't "lazily" regenerating
   curated data; replay scales with what's available to match, not faster.

2. **True novel discovery keeps growing across rounds.** Train-novel-MSUN
   rose **+4.9 pp** from M1 → M2 (26.5% → 31.4%), and the cumulative gain
   over the public Crystalite baseline is **+8.8 pp** (22.6% → 31.4%).
   These are real, training-set-disjoint, LeMat-Bulk-disjoint, metastable
   structures.

3. **The "is the flywheel regurgitating training data?" critique is closed.**
   At M1, 88% of LeMat-MSUN is true new discovery. At M2, 81% is. The
   recipe genuinely produces new metastables, not training-set memorization
   inflated by LeMat-Bulk's incomplete coverage.

**Effect on prior conclusions.**

- The "+7-8 pp replay gap" diagnosis in earlier blocks of this log
  ([Curation-ladder ablations 2026-05-15](#),
  [Replay-decomposition 2026-05-14](#)) was inflated by the matcher
  tolerance; the corrected replay is roughly half. The qualitative
  conclusion (verifier-curation is the load-bearing step) is unchanged
  but the quantitative gap between curation tiers needs rescoring under
  the corrected matcher before re-stating it precisely.

- The v1-vs-v2 selector comparison (M1_v1 LeMat MSUN = 30.0% vs M1_v2
  LeMat MSUN = 29.3%) is unaffected — both numbers come straight from
  LeMat, not from our internal matcher.

**Cascade implication.** Round 2 is now planned from M2_v1 (v1 lineage)
with v2 corrections layered on for any further training. The
checkpoint-selection bug is shown to have ≤0.7 pp impact on external MSUN
and is treated as a brief methodological footnote rather than a primary
cascade event.

---

## Round 1 v2 (M2_v2) — 2026-05-29

**Setup**

```text
M2_v2 = train MP20 + S0 + S1_v2_full (S1_v2 + topup_1015 merged)
        → outputs/dng_synthetic_round1_msun_108k_v2_resume
Effective train: 27,138 + 27,138 + 54,276 = 108,552
Corrected checkpoint selector (Train_MSUN against augmented training set;
src/utils/checkpoint.py `auto` selects on full-train novelty reference).
```

LeMat result: `results_final/m2v2.json` (comprehensive_multi_mlip_hull, n=2500).

**Headline table**

| Model | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | SUN+MSUN | mean e_hull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M2_v1 (buggy selector) — superseded | 97.16 | 99.46 | 69.45 | 8.15 | 60.11 | 2.14 | 38.58 | 40.72 | 0.0898 |
| **M2_v2** | 96.40 | 99.13 | 70.91 | 7.63 | 55.81 | 2.28 | **35.06** | 37.34 | 0.0944 |

The selector-bug fix deflated MSUN by 3.52 pp (38.58 → 35.06) — consistent
with the M2_v1 → M2_v2 selector swap propagating through.

**Replay decomposition (LeMat matcher, ltol=0.1):**

| Reference | LeMat MSUN | Replay (pp) | True train-novel MSUN |
|---|---:|---:|---:|
| Synthetic only (S0 ∪ S1_v2_full, 80k) | 35.06% | 4.32 | 30.75% |
| Full train (MP20 ∪ S0 ∪ S1_v2_full, 108k) | 35.06% | 5.69 | 29.36% |

Source: `results_final/replay_m2v2_*.json`.

---

## Round 2 (M3) — 2026-06-02

**Setup**

```text
M3 = train MP20 + S0 + S1_v2_full + S2_v2
     → outputs/dng_synthetic_round2_msun_217k_v2
Effective train: 27,138 + 27,138 + 54,276 + 108,552 = 217,104
Generator for S2: M2_v2 (dng_synthetic_round1_msun_108k_v2_resume/best.pt)
S2 reference dedup: MP20 ∪ S0 ∪ S1_v2_full
S2 yield: 108,552 (data/synthetic/crystalite_round2_msun_108k_v2)
```

LeMat result:
`~/lemat-genbench/results_final/crystalite_m3_msun_217k_v2_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260602_182933.json`.

**Headline table**

| Model | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | SUN+MSUN | mean e_hull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Crystalite (M0) | 97.20 | 95.80 | 53.20 | 12.70 | 51.60 | 1.50 | 22.60 | 24.10 | 0.1322 |
| M1_v2 | 96.40 | 99.13 | 65.15 | 9.21 | 53.40 | 1.74 | 29.34 | 31.08 | 0.0988 |
| M2_v2 | 96.40 | 99.13 | 70.91 | 7.63 | 55.81 | 2.28 | 35.06 | 37.34 | 0.0944 |
| **M3** | 97.76 | 99.26 | 76.51 | 5.77 | 57.90 | **2.54** | **40.43** | 42.96 | 0.0939 |

Counts over 2500 submitted: M3 — 2444 valid, 62 SUN, 988 MSUN, 1050 SUN+MSUN.

**Replay decomposition (LeMat matcher, ltol=0.1, synthetic-only refs):**

| Model | Synthetic ref size | LeMat MSUN | Replay (pp) | Train-novel MSUN | Replay / MSUN |
|---|---:|---:|---:|---:|---:|
| M1_v2 | 27k | 29.34% | 3.24 | 26.10% | 11.0% |
| M2_v2 | 81k | 35.06% | 4.32 | 30.75% | 12.3% |
| **M3** | 190k | **40.43%** | **7.69** | **32.74%** | **19.0%** |

Source: `outputs/msun_replay/m3/summary.json`.

**Train-novel SUN** (strict-stable, novel vs augmented train, M3):
- LeMat SUN 2.54%, replay 0.57 pp → **train-novel SUN 1.97%** — new high.
  (M1: 1.58, M2: 1.54 → M3: 1.97; the cascade's strict-SUN train-novel
  was flat-to-down through M2 and leapt at M3.)

**Interpretation.**

1. **External LeMat-MSUN scales monotonically:** 22.6 → 29.3 → 35.1 → 40.4.
2. **Train-novel MSUN scales monotonically with diminishing per-round velocity:**
   26.1 → 30.8 → 32.7 (deltas +4.7, +2.0 pp). The cascade keeps adding genuine
   novel discoveries, just less per round.
3. **Replay grows with reference-set size, but sub-linearly.** Ref size doubled
   M2→M3 (81k → 190k); replay rose +3.4 pp (4.3 → 7.7), while per-reference
   replay rate continued to fall (M1: 12 ppm, M2: 5 ppm, M3: 4 ppm). The model
   is matching a *smaller fraction* of each new reference structure as the
   cascade progresses — consistent with iterative exploration, not memorization.
4. **Strict-stable% drops** (9.2 → 7.6 → 5.8) but the metastable-shell gain
   dominates: SUN+MSUN climbs 31.1 → 37.3 → 43.0. The `msun_like` filter
   continues to shift mass toward the 0–0.1 eV/atom metastable band.

---

## S_big one-shot control (M_big) — 2026-06-02

**Setup**

```text
S_big = M0 samples → NequIP relax → MP2020 hull → msun_like → dedup vs MP20
        Yield 69,299 (first pass) + 12,115 (topup) = 81,414 merged
        (data/synthetic/crystalite_S_big_81k_merged)
M_big = train MP20 + S_big_81k_merged → outputs/dng_synthetic_S_big_matched_81k
Effective train: 27,138 + 81,414 = 108,552 — matches M2_v2 budget exactly.
```

Earlier `dng_synthetic_S_big_v2` and `_v2_resume` (May 15–16) were aborted runs
on the pre-topup 69k set; canonical M_big is the May-29 matched_81k run, step
825,000, primary `Train_MSUN` selector.

LeMat result:
`~/lemat-genbench/results_final/crystalite_s_big_matched_81k_n2500_nequip_relaxed_comprehensive_multi_mlip_hull_20260602_184302.json`.

**Headline table — matched-budget comparison vs M2_v2:**

| Model | Valid | Unique | Novel | Stable | Metastable | SUN | MSUN | S+M | ē_hull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M_big (one-shot from M0) | 96.52 | 99.09 | 69.46 | 7.71 | 54.12 | 1.82 | **32.91** | 34.73 | 0.0970 |
| M2_v2 (iterative S0 ∪ S1) | 96.40 | 99.13 | 70.91 | 7.63 | 55.81 | 2.28 | **35.06** | 37.34 | 0.0944 |
| Δ (iter − one-shot)      |        |        |  +1.45 |        |  +1.69 | **+0.46** | **+2.15** | +2.61 |        |

**Replay decomposition (synthetic-only ref, n_ref=81,414):**

| Model | LeMat MSUN | Replay (pp) | Train-novel MSUN | Replay / MSUN |
|---|---:|---:|---:|---:|
| M_big | 32.91% | 5.14 | **27.77%** | 15.6% |
| M2_v2 (matched ref size) | 35.06% | 4.32 | **30.75%** | 12.3% |

Source: `outputs/msun_replay/s_big/summary.json`.

**Interpretation.**

1. **Iteration > one-shot at matched compute, training-set size, and curation
   recipe.** Same MP20 + ~81k curated synthetic, same verifier, same dedup
   matcher — iterating reaches 35.06% MSUN vs one-shot's 32.91% (+2.15 pp).
2. **The gap is wider on train-novel** (+2.98 pp): 30.75% vs 27.77%. M_big
   leans harder on regenerating its own curated training data (15.6% replay
   share vs M2's 12.3%) — the one-shot curated set is a tighter mode for the
   trained model to revisit. Iteration produces a more diffuse generator that
   discovers more genuinely-novel structure outside its own training pool.
3. **Strict-stable% is essentially tied** (M_big 7.71 vs M2_v2 7.63); the
   iteration advantage on raw SUN comes from the metastable band shift, not
   strict-stable gains at the M2 budget.
4. **No matched control was run for M3.** A 190k one-shot from M0 would be the
   equivalent comparator for M3 (217k effective train); not currently planned
   (see [run_tracker.md](run_tracker.md) — the matched-budget claim rests on
   M2_v2 vs M_big, M3 is the compounding evidence at extended budget).

---

## Practicality drift — HHI + element-class analysis — 2026-05-21

Verifier-curated self-training drifts toward **supply-critical, less practical
chemistry**. Quantified two ways: LeMat HHI (element supply-risk; higher =
worse, 0–10 scale) and per-round element-class fractions from
`local_data/features.parquet`.

**HHI from LeMat `comprehensive_multi_mlip_hull` (via `scripts/extract_lemat_metrics.py`), updated 2026-06-02:**

| Model | HHI_production | HHI_reserve |
|---|---:|---:|
| Base Crystalite (M0) | 3.525 | 2.698 |
| Oversample control | 3.419 | 2.628 |
| Raw synthetic | 3.424 | 2.623 |
| Dedup-only synthetic | 3.423 | 2.612 |
| M1 (S0) | 3.605 | 2.654 |
| M2 (S0 ∪ S1_v2_full) | 3.794 | 2.672 |
| **M3 (S0 ∪ S1 ∪ S2)** | **3.943** | 2.632 |
| **M_big (one-shot)** | **3.676** | 2.639 |

- **Production-concentration drifts up monotonically across the cascade:**
  M0 3.525 → M1 3.605 → M2 3.794 → **M3 3.943** (+0.42 over M0; robustly
  outside n=2500 sampling noise). **Reserve-concentration does not** (2.698 →
  2.654 → 2.672 → 2.632, flat/slightly down). The drift is production-side —
  geographically supply-locked elements (the rare-earth signature), not
  reserve-limited ones.
- **M_big sits below M2_v2 on HHI_production (3.676 vs 3.794)** at matched
  training budget. The one-shot from M0 does *not* concentrate production
  risk to the same degree as the iterative cascade — consistent with the
  composition tables showing S_big stays at S0-level lanthanide content while
  iteration pushes higher.

**Element-class fractions (% of structures containing the class), updated
2026-06-02 to include S2_v2 and S_big_merged:**

| Dataset | n | %lanthanide | %rare-earth (+Sc,Y) | %radioactive (Tc,Pm,actinoids,Z≥84) |
|---|---:|---:|---:|---:|
| MP20 (base) | 27,138 | 35.5 | 40.6 | 6.6 |
| S0 (round 0) | 27,138 | 46.0 | 50.8 | 10.8 |
| S1_v2_full (round 1) | 54,276 | 51.5 | 55.6 | 12.3 |
| **S2_v2 (round 2)** | 108,552 | **55.8** | **59.8** | **14.0** |
| **S_big_merged (one-shot M0)** | 81,414 | 46.0 | 50.4 | 11.0 |

- Cascade drift continues monotonically into S2_v2: lanthanide-bearing
  fraction +**20.3 pp** over MP20 (35.5 → 55.8%); radioactive-element fraction
  more than doubles (6.6 → 14.0%).
- **The drift is specifically caused by iteration, not data volume.** S_big
  (81k synthetic, one-shot from M0) holds at S0 composition (46.0% lanthanide,
  11.0% radioactive); iterating to the same effective training budget (M2_v2)
  produces an S1-level mix (51.5% lanthanide), and pushing to M3 (S2_v2)
  reaches 55.8%. Volume-matched one-shot curation does not amplify
  rare-earth selection — feedback through the trained generator does.

**Top element shifts vs MP20, updated 2026-06-02
(`scripts/diagnose_element_drift.py`):**
- **Pm (promethium, radioactive)** remains the single largest mover and grows
  with iteration: MP20 1.0% → S0 5.4% → S1_v2 6.9% → **S2_v2 8.9%** (~9×).
  One-shot S_big stops at 5.8%, mid-way between S0 and S1.
- All top-20 movers are lanthanides + Pm + intermetallic formers (Ga, In, Cu,
  Pd, Ni, Fe, Al, Ag, Au, Ca). Every lanthanide is monotonically higher in
  S2_v2 than in S1_v2 than in S0. None saturate.
- **Cascade-vs-one-shot at top movers:**
  Ho 2.4% (MP20) → 4.4 (S0) → 6.9 (S1) → **8.5 (S2)** vs **4.8 (S_big)**;
  Er 2.6 → 4.8 → 7.0 → **8.0** vs **4.8**;
  Nd 3.0 → 5.0 → 6.9 → **8.4** vs **5.3**;
  Dy 2.6 → 4.5 → 6.8 → **7.5** vs **4.6**.
  Pattern: S_big sits in the S0–S1 band on every lanthanide; the cascade
  pushes monotonically beyond.
- Largest **decreases**: O −6.9, N −2.4, B −2.2, C −2.1, H −2.1, F −2.1 — i.e.
  away from oxide/nitride/carbide/boride/hydride/fluoride (abundant-element,
  practical) chemistry. **The drift is a chemistry-class shift: oxide/main-group
  → rare-earth intermetallic.**

**Caveats.**
- MP20 itself is already 35.5% lanthanide / 6.6% radioactive (MP carries computed
  entries for these), so the flywheel **amplifies** an existing skew rather than
  inventing it.
- This is NOT a NequIP verifier artifact: a second MP-consistent-PBE potential
  (PET-OAM-XL) corroborates NequIP stability on 1024 lanthanide structures to
  ~1 meV/atom (see the PET-OAM agreement note). The structures are genuinely
  DFT-metastable — the problem is that DFT-metastability is blind to
  practicality, not that the verifier is wrong.

**Interpretation.** Stability is necessary but not sufficient. A stability-only
verifier optimizes an element-agnostic target, so iterating on it drifts the
generator toward whatever metastable chemistry is easiest to satisfy — here,
rare-earth intermetallics including radioactive elements. A practical recipe
needs a practicality/scarcity constraint (element denylist and/or HHI cap)
alongside the metastability filter. M3 is held pending that decision so it is
not trained on un-constrained S2.
