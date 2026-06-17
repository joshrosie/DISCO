# Follow-up Backlog

Open ablations and extensions. Each item lists the question, the minimum-viable
test, and whether it's a confound control vs. a story extension. Promote
scheduled items to [run_tracker.md](run_tracker.md); prune when shipped or
killed.

## Selection-strictness ablations

Round 0's filter is `msun_like` (e_hull ≤ 0.1). The headline result is novel
metastable discovery; strict stable yield dropped 2.2 pp. Hypothesis: training
on the metastable survivor band teaches the model to live in the shell rather
than push to the hull. Two follow-ups isolate this:

- **`stable_like` filter.** Rebuild S0 with `e_hull ≤ 0` and a tighter dedup;
  retrain M1 with matched count. Question: does stricter filtering recover SUN
  while preserving the MSUN gain, or does it collapse novelty?
- **`low_hull_topk` + diversity caps.** Take the lowest-`e_hull` survivors
  with caps on per-formula and per-spacegroup duplication. Question: is sharper
  selection better than threshold selection at matched count?

## Composition-arity confound — closed (2026-05-11)

The Round 0 n-ary stratified re-scoring confirmed that synthetic outperforms
the oversample-real control within every composition-arity bucket (binary
+5.5 pp, ternary +4.7, quaternary +9.7, 5+ +2.6 SUN+MSUN). The arity shift
(~+4.6 pp ≥4-element vs the control) is real but does not explain the gain.
Full bucket table: [results_log.md](results_log.md) "N-ary stratified
re-scoring". The analysis script:
[scripts/analyze_nary_stratified.py](../../scripts/analyze_nary_stratified.py).

**n-ary-matched curation** (build an S0 variant where the n-ary distribution
matches MP20's, then retrain) is no longer load-bearing. Kept here only as a
possible future ablation if a reviewer pushes back on "but you didn't *match*
the arity, only stratify by it." Low priority.

## Distribution shift analysis — scheduled

Understand what the Flywheel is actually doing to the training distribution.
Script: [`scripts/analyze_flywheel_dataset_shift.py`](../../scripts/analyze_flywheel_dataset_shift.py).

Metrics tracked per dataset slice (MP20, S0, S1, S2; cumulative and marginal):
- Composition-arity distribution (n-ary stacked bar)
- Formula-space entropy (normalized Shannon)
- Element prevalence TV distance vs MP20 (atom-fraction)
- E_hull distribution (binned: stable / near / mid / far / unstable)
- Mean formula and cell atom counts

Run with current S0+S1 immediately; re-run when S2 and S_big land to track
the iterative vs one-shot shift side-by-side.
Promote to run_tracker when submitted.

## Curation ladder ablations — scheduled

Two queued ablations (see [run_tracker.md](run_tracker.md)) complete the ladder
from "no curation" to "full Flywheel":

- **Raw / no filter** (`--filter_level raw --dedup_mode none`): pure self-distillation.
  Tests whether any curation is necessary.
- **Dedup-only / no MLIP** (`--filter_level valid --dedup_mode structure`): geometry validity
  + structure novelty, but no relaxation or e_hull gate. Tests whether the physical
  verifier drives the effect vs. just "novel non-duplicate structures."

Both use M0 at 27k, matching Round 0 count exactly.

## Compute-comparison: iterative vs one-shot

Two scientifically distinct modes share the same raw recipe:

```text
iterative:  M0 → S0 → M1 → S1 → M2  (current path)
one-shot:   M0 → S_big              (mine a large pool from the original model)
```

- **One-shot S_big from M0 at matched budget.** Generate ~108k candidates
  from M0, msun_like + structure dedup, train M1' on MP20 + S_big. Compare
  to iterative (M2 trained on MP20 + S0 + S1) at the same total |synthetic|.
  Question: is the gain from progressive bootstrapping or from data mining?

## Verifier strength

- **Stronger / ensemble verifier for curation.** Swap NequIP for MACE-MP, or
  curate with NequIP + MACE intersection. Headline claim still external
  (LeMat). Question: does a more accurate curation verifier amplify the
  effect, or saturate it?

## Training-distribution transfer

- **Alex-MP20 base.** Train M0' on Alex-MP-20 instead of MP-20; repeat
  Round 0. Question: does the recipe generalize, or is MP20-specific?

## Online / semi-online extension

The offline rounds are the simplest instantiation of a broader online
verifier-guided self-improvement loop. Full per-step relaxation is
infeasible; a buffered semi-online loop is the realistic version:

```text
generate batch → cheap prescreen → async relax/filter → update buffer
train/fine-tune on real + buffer
repeat
```

- **Replay-buffer prototype.** Implement an async buffer that the trainer
  reads from; relaxation runs on a separate queue with its own GPU. Cost
  bounded by the prescreener throughput. Likely a separate paper if the core
  iterative result lands cleanly first.
- **Distilled / cheap surrogate prescreeners.** Distill NequIP to a smaller
  ranker for shortlisting; reserve full relaxation for the shortlist.

### Ensemble-MLIP filtering against the fixed DFT hull (per Ivor 2026-05-14)

The original sketch here was an "online flywheel with self-updating
local MLIP hull" — generate, relax, accept, push survivors onto a
locally-grown hull, gate next round's candidates against that hull. This
is **not** the right direction, per discussion with Ivor (materials
scientist consult, 2026-05-14):

  - MLIP errors are systematic, not random. Each MLIP has training-set
    biases and characteristic failure modes.
  - Modifying the hull with structures the *same* MLIP scored compounds
    that MLIP's bias: the hull comes to reflect what one MLIP thinks
    is low-energy, not what's physically low-energy. The loop appears
    to "improve" but is just self-reinforcing one MLIP's worldview.
  - No DFT-trustworthy frontier-pushing happens.

The right direction is the opposite: **keep the hull fixed (the
MP-DFT-derived PPD we already use), but ensemble multiple independent
MLIPs at the filtering step.** Each MLIP relaxes a candidate and
computes `e_above_hull` against the same trusted reference; accept only
on consensus.

The hull never moves. The bar never lowers. The *certainty* about
whether a candidate clears the bar improves because we're not trusting
a single MLIP's possibly-biased reading. LeMat-GenBench's
`comprehensive_multi_mlip_hull` does exactly this for evaluation
(MACE/ORB/UMA consensus); we should mirror it at curation time.

Concrete extension path (in increasing cost):

1. **Two-MLIP consensus**: curate with NequIP-OAM-L AND EquiformerV3-OAM,
   accept only structures both agree are metastable. Infrastructure for
   EquiformerV3-OAM already exists (see `equiformer_v3_setup.md`); we
   defer it from MVP for scope reasons, not because the science is
   wrong.

2. **Three-MLIP consensus**: + MACE-MP-0. Brings curation-time
   verification in line with LeMat-GenBench's eval-time consensus.

3. **Optional online piece** (separate idea): even with ensemble
   consensus, you could run the loop continuously inside training
   rather than across discrete rounds. The piggyback on precise_every
   = 25000 is still architecturally sensible. But the hull stays
   fixed; survivors only feed back into the *training set*, not the
   hull.

### Implementation note: relax once, score many

The relaxation step is by far the most expensive part of curation
(~60s per batch of 32 with NequIP+FIRE+frechet). Single-point energy
computation on an already-relaxed structure is milliseconds. So the
right ensemble architecture is:

```
candidate
  → NequIP-OAM-L: relax → final_structure   (the expensive step)
  → final_structure ─→ NequIP single-point E   → e_hull_nequip
                    ─→ EquiformerV3 single-point E → e_hull_eqv3
                    ─→ MACE single-point E       → e_hull_mace
  → consensus: accept iff all (or quorum) report e_hull ≤ 0.1
```

Adding scorers is near-marginal cost. Three MLIPs is ~5% slower than
one in this design, not 3× slower.

### Implementation note: environment isolation

Different MLIPs have conflicting Python dependencies (CUDA versions,
e3nn versions, torch versions) that don't fit in one env. We already
handle this for EquiformerV3 via a subprocess wrapper (see
[src/eval/cfg_eval.py](../../src/eval/cfg_eval.py)
`EquiformerV3FormEnergyOracle` and
[equiformer_v3_setup.md](equiformer_v3_setup.md)). Same pattern works
for MACE: each MLIP lives in its own venv, the orchestrator passes
relaxed structures (CIFs or pickled Structures via stdin) and receives
energies (via stdout/JSON). Per-call overhead is dominated by Python
startup, so batched RPC (one call → N structures → N energies) keeps
it cheap.

Architectural pieces for the simplest two-MLIP version:
  - `_build_row_oracle` accepts a list of MLIPs instead of one; orchestrates
    a single relax with the primary MLIP and N single-point scoring calls.
  - Consensus rule (all-must-pass vs quorum) is a config flag.
  - Per-candidate metadata records each MLIP's verdict so we can
    decompose effects post-hoc.
  - Or, simpler interim: keep two `make_synthetic_dataset` runs (one per
    MLIP) and use `StructureMatcher` to take the intersection of
    accepted sets.

Pitch as follow-up paper: "Ensemble-verifier flywheel — single-MLIP
curation compounds verifier bias; consensus curation aligns with
benchmark-time evaluation and removes a systematic confound."

## Appendix candidate: train-novel MSUN as a stricter benchmark

When the original Round 0 / Round 1 trainings (M1_v1, M2_v1) were eval'd
under LeMat's MP20-only novelty reference, MSUN looked like 22.6 → 30.0 →
38.6 (+16 pp over two iterations). Re-scoring the same samples with
novelty referenced against the *augmented* training set (MP20 ∪ S_0 ∪ ...)
gave 22.6 → 22.3 → 27.2 (+4.6 pp). The gap grew from 7.7 pp at round 1 to
11.4 pp at round 2: the "replay" of training-set augmentations being
counted as novel inflates apparent gains, and the inflation compounds with
iteration.

Concrete contribution worth writing up as an appendix (Option A in the
discussion notes — keep it as appendix unless the v2 train-novel numbers
turn out to be very strong, in which case promote to a main-text methods
subsection):

- Define `train-novel MSUN`: MSUN where the novelty reference is the
  generator's actual training corpus, not a fixed external set.
- Use the v1 numbers as the empirical exhibit of the divergence.
- Re-state the v2 numbers under both metrics in the main table.
- Cut the engineering story (checkpoint-selection bug, retrain cascade) to
  a footnote.

Implementation cost: writing only — the data already exists (LeMat eval
JSONs for M1_v1 and M2_v1 under `outputs/external_eval/`, the chart code
under `scripts/plot_flywheel_msun_scaling.py`).

Possible extension: stratify the replay gap by composition arity or e_hull
band using `scripts/analyze_nary_stratified.py` / e-hull stratification, to
show *what kind* of structure gets replayed most. If replay concentrates
in high-symmetry rocksalts or known dominant formulas, that's a sharper
observation about mode-reinforcement under self-training.

## Methodology side-notes (not blocking)

- SMACT validity reporting: keep for parity with prior crystal-gen work, but
  do not use as a load-bearing metric in the abstract. Note in limitations.
- DFT spot-check of headline SUN samples: if compute permits, ≤20 candidates
  via VASP at MP settings would strengthen the external-evaluation argument.
  Out of scope unless a collaborator volunteers DFT cycles.
