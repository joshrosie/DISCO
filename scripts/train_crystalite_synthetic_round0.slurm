#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=dng_synth_r0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/out-%x.%A.out
#SBATCH --error=/home/jrosenthal/atom-reps/logs/dng_synthetic_aug/err-%x.%A.err

# Train DNG/Crystalite with the MP20-sized Round 0 synthetic MSUN set.
#
# Default:
#   MP20 train + data/synthetic/crystalite_round0_msun_27k
#
# Oversampled-real control:
#   sbatch --export=ALL,AUGMENTATION=oversample_real,OUT_DIR=outputs/dng_oversample_real_27k,WANDB_NAME=dng_oversample_real_27k \
#     scripts/train_crystalite_synthetic_round0.slurm

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0

export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/atom-reps"

mkdir -p "$HOME/atom-reps/logs/dng_synthetic_aug"

export UV_CACHE_DIR="${TMPDIR:-/tmp/$USER/uv-cache-$SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

AUGMENTATION="${AUGMENTATION:-synthetic_concat}"
SYNTHETIC_DATA="${SYNTHETIC_DATA:-data/synthetic/crystalite_round0_msun_27k}"
NUM_EXTRA_SAMPLES="${NUM_EXTRA_SAMPLES:-27138}"
OUT_DIR="${OUT_DIR:-outputs/dng_synthetic_round0_msun_27k}"
WANDB_NAME="${WANDB_NAME:-dng_synthetic_round0_msun_27k}"
WANDB_PROJECT="${WANDB_PROJECT:-DNG_MP20}"

MAX_STEPS="${MAX_STEPS:-2500000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
VAL_EVERY="${VAL_EVERY:-5000}"
PRECISE_EVERY="${PRECISE_EVERY:-25000}"
PRECISE_METRICS_COUNT="${PRECISE_METRICS_COUNT:-10000}"
SAMPLE_METRICS_COUNT="${SAMPLE_METRICS_COUNT:-2048}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-2048}"
THERMO_STABILITY_BATCH="${THERMO_STABILITY_BATCH:-128}"
NO_THERMO_BEFORE_STEPS="${NO_THERMO_BEFORE_STEPS:-100}"
FULL_TRAIN_NOVELTY="${FULL_TRAIN_NOVELTY:-1}"
BEST_CKPT_SELECTOR="${BEST_CKPT_SELECTOR:-auto}"

THERMO_PPD_MP="${THERMO_PPD_MP:-/home/jrosenthal/atom-reps/mp_02072023/2023-02-07-ppd-mp.pkl}"
NEQUIP_COMPILE_PATH="${NEQUIP_COMPILE_PATH:-/home/jrosenthal/atom-reps/mlips/aot_batch.nequip.pt2}"

if [[ "$AUGMENTATION" == "synthetic_concat" ]]; then
  IFS=':' read -r -a SYNTHETIC_DATA_LIST <<< "$SYNTHETIC_DATA"
  for SYNTHETIC_ROOT in "${SYNTHETIC_DATA_LIST[@]}"; do
    if [[ -z "$SYNTHETIC_ROOT" ]]; then
      continue
    fi
    if [[ ! -e "$SYNTHETIC_ROOT/raw/train.csv" && ! -f "$SYNTHETIC_ROOT" ]]; then
      echo "Synthetic dataset not found (missing raw/train.csv): $SYNTHETIC_ROOT" >&2
      exit 2
    fi
  done
fi

AUG_ARGS=(--augmentation "$AUGMENTATION")
if [[ "$AUGMENTATION" == "synthetic_concat" ]]; then
  AUG_ARGS+=(--synthetic_data "$SYNTHETIC_DATA")
elif [[ "$AUGMENTATION" == "oversample_real" ]]; then
  AUG_ARGS+=(--num_extra_samples "$NUM_EXTRA_SAMPLES")
elif [[ "$AUGMENTATION" != "none" ]]; then
  echo "Unsupported AUGMENTATION=$AUGMENTATION" >&2
  exit 2
fi

INIT_FROM_ARGS=()
if [[ -n "${INIT_FROM:-}" ]]; then
  if [[ ! -f "$INIT_FROM" ]]; then
    echo "[train-synth] INIT_FROM checkpoint not found: $INIT_FROM" >&2
    exit 2
  fi
  INIT_FROM_ARGS=(--init_from "$INIT_FROM")
  echo "[train-synth] init_from=$INIT_FROM (model+EMA only; optimizer fresh)"
fi

WANDB_ARGS=(--wandb_name "$WANDB_NAME" --wandb_project "$WANDB_PROJECT")
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--no_wandb)
fi
FULL_TRAIN_NOVELTY_ARGS=()
if [[ "$FULL_TRAIN_NOVELTY" == "1" ]]; then
  FULL_TRAIN_NOVELTY_ARGS=(--sample_full_train_novelty)
fi

echo "[train-synth] augmentation=$AUGMENTATION"
echo "[train-synth] synthetic_data=$SYNTHETIC_DATA"
echo "[train-synth] num_extra_samples=$NUM_EXTRA_SAMPLES"
echo "[train-synth] output_dir=$OUT_DIR"
echo "[train-synth] max_steps=$MAX_STEPS batch_size=$BATCH_SIZE"
echo "[train-synth] thermo_ppd_mp=$THERMO_PPD_MP"
echo "[train-synth] nequip=$NEQUIP_COMPILE_PATH"
echo "[train-synth] full_train_novelty=$FULL_TRAIN_NOVELTY"
echo "[train-synth] best_ckpt_selector=$BEST_CKPT_SELECTOR"

srun uv run python src/train_crystalite.py \
  --dataset_name mp20 \
  --data_root data/mp20 \
  --metrics_data_root data/mp20 \
  --metrics_dataset_name mp20 \
  --bf16 \
  --nmax 20 \
  --d_model 512 \
  --n_layers 14 \
  --n_heads 16 \
  --dropout 0.0 \
  --attn_dropout 0.0 \
  --type_encoding chem_pca_v2_16 \
  --coord_embed_mode fourier \
  --coord_n_freqs 32 \
  --coord_head_mode direct \
  --use_edge_bias \
  --edge_bias_hidden_dim 256 \
  --edge_bias_n_freqs 12 \
  --edge_bias_n_rbf 32 \
  --pbc_radius 1 \
  --use_noise_gate \
  --batch_size "$BATCH_SIZE" \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 0 \
  --lr_warmup_steps 1000 \
  --ema_decay 0.9999 \
  --max_steps "$MAX_STEPS" \
  --loss_weights 1 50 5 \
  --coord_loss_mode frac_mse \
  --edm_P_mean -1.2 \
  --edm_P_std 1.2 \
  --sigma_data_type 0.3 \
  --sigma_data_coord 0.3 \
  --sigma_data_lattice 0.3 \
  --sigma_min 0.002 \
  --sigma_max 80.0 \
  --rho 7.0 \
  --S_churn 60.0 \
  --S_min 0.0 \
  --S_max 999.0 \
  --S_noise 1.003 \
  --aa_frac_max_scale 0.0 \
  --aa_rho_types 0.0 \
  --aa_rho_coords 10.0 \
  --aa_rho_lattice 10.0 \
  --lattice_embed_mode mlp \
  --lattice_repr ltri \
  --sample_num_steps 150 \
  --atom_count_strategy empirical \
  --sample_mode ema \
  --sample_seed 123 \
  --sample_every 0 \
  --sample_metrics_count "$SAMPLE_METRICS_COUNT" \
  --sample_chunk_size "$SAMPLE_CHUNK_SIZE" \
  --sample_vis_count 5 \
  --sample_compute_novelty \
  --sample_novelty_limit -1 \
  "${FULL_TRAIN_NOVELTY_ARGS[@]}" \
  --precise_every "$PRECISE_EVERY" \
  --precise_metrics_count "$PRECISE_METRICS_COUNT" \
  --precise_sun_k 2048 \
  --log_every 50 \
  --val_every "$VAL_EVERY" \
  --val_batches 10 \
  --ckpt_every "$CKPT_EVERY" \
  --ckpt_latest_only \
  --best_ckpt \
  --best_ckpt_selector "$BEST_CKPT_SELECTOR" \
  --thermo_stability_check \
  --thermo_stability_device cuda \
  --thermo_ppd_mp "$THERMO_PPD_MP" \
  --thermo_mlip nequip \
  --nequip_compile_path "$NEQUIP_COMPILE_PATH" \
  --nequip_relax_mode batch \
  --nequip_optimizer FIRE \
  --nequip_cell_filter frechet \
  --nequip_fmax 0.005 \
  --nequip_max_force_abort 1000000 \
  --thermo_stability_batch "$THERMO_STABILITY_BATCH" \
  --thermo_stability_count 0 \
  --no_thermo_before_steps "$NO_THERMO_BEFORE_STEPS" \
  --thermo_ehull_method mp2020_like \
  --thermo_relax_steps 200 \
  "${AUG_ARGS[@]}" \
  "${INIT_FROM_ARGS[@]}" \
  --output_dir "$OUT_DIR" \
  "${WANDB_ARGS[@]}"
