#!/bin/bash
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --job-name=msun_replay
#SBATCH --output=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/synthetic_aug_msun/err-%x.%A.err

# Decompose LeMat-MSUN into "replay from training augmentation" vs
# "truly novel vs (LeMat-Bulk ∪ training_synth)" using pymatgen
# StructureMatcher(ltol=0.1) — the exact comparator LeMat uses.
#
# Required env vars:
#   CIF_DIR          — relaxed_cifs directory submitted to LeMat
#   LEMAT_JSON       — path to the downloaded LeMat result JSON (UMA preferred)
#   REFERENCE_ROOTS  — colon-separated synthetic roots (each has raw/train.csv).
#                      MP20 already covered by LeMat-Bulk; only S_r needed.
#   RUN_TAG          — short tag (e.g. m1v1, m2v1) for output naming
#
# Example — M1_v1:
#   CIF_DIR=outputs/external_eval/crystalite_synthetic_aug_n2500_nequip_relaxed/relaxed_cifs
#   LEMAT_JSON=outputs/external_eval/lemat_results/m1v1_uma.json
#   REFERENCE_ROOTS=data/synthetic/crystalite_round0_msun_27k
#   RUN_TAG=m1v1
#
# Before sbatch, upload the LeMat JSON from your local machine:
#   mkdir -p $HOME/atom-reps/outputs/external_eval/lemat_results
#   scp ~/Downloads/results_final/crystalite_synthetic_aug_n2500_nequip_relaxed_uma_*.json \
#       snellius:$HOME/atom-reps/outputs/external_eval/lemat_results/m1v1_uma.json

set -euo pipefail

cd "$HOME/atom-reps"
mkdir -p logs/synthetic_aug_msun

CIF_DIR="${CIF_DIR:?Set CIF_DIR}"
LEMAT_JSON="${LEMAT_JSON:?Set LEMAT_JSON}"
REFERENCE_ROOTS="${REFERENCE_ROOTS:?Set REFERENCE_ROOTS}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG}"

OUT_DIR="${OUT_DIR:-outputs/msun_replay/$RUN_TAG}"
mkdir -p "$OUT_DIR"

SUMMARY_JSON="$OUT_DIR/summary.json"
PER_SAMPLE_JSON="$OUT_DIR/per_sample.json"

echo "[replay] run_tag=$RUN_TAG"
echo "[replay] cif_dir=$CIF_DIR"
echo "[replay] lemat_json=$LEMAT_JSON"
echo "[replay] reference_roots=$REFERENCE_ROOTS"
echo "[replay] out_dir=$OUT_DIR"

uv run python scripts/score_msun_replay.py \
    --lemat_json "$LEMAT_JSON" \
    --cif_dir "$CIF_DIR" \
    --reference_roots "$REFERENCE_ROOTS" \
    --output_json "$SUMMARY_JSON" \
    --per_sample_json "$PER_SAMPLE_JSON"

echo "[replay] DONE"
cat "$SUMMARY_JSON"
