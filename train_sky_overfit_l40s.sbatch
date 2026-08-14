#!/usr/bin/env bash
#SBATCH --job-name=dggt-sky-overfit
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=/data/user_data/junyizh3/projects/dggt
CONDA_ROOT=/data/user_data/junyizh3/miniconda3
TRAIN_LOG_DIR="$PROJECT_DIR/logs/sky_overfit_000"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dggt

cd "$PROJECT_DIR"
mkdir -p "$TRAIN_LOG_DIR"

python -c 'import tensorboard' || {
    echo "TensorBoard is missing. Install it in the dggt environment first." >&2
    exit 1
}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Overfit sample: scene=000, start=0, length=8"
echo "TensorBoard port: 6007 (compute-node loopback only)"

tensorboard \
    --logdir "$TRAIN_LOG_DIR/tensorboard" \
    --host 127.0.0.1 \
    --port 6007 \
    > "$TRAIN_LOG_DIR/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

cleanup() {
    kill "$TENSORBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    train.py \
    --raw_waymo_dir /data/datasets/waymo/waymo-open-dataset-v2.0.1/training \
    --sky_mask_cache_dir /scratch/junyizh3/waymo_sky_masks \
    --scene_names 000 \
    --raw_sequence_lengths 8 \
    --raw_fixed_start 0 \
    --ckpt_path pretrained/model_latest_waymo.pt \
    --log_dir logs/sky_overfit_000 \
    --sky_model cubemap \
    --batch_size 1 \
    --max_epoch 1000 \
    --save_image 50 \
    --log_every 1 \
    --save_ckpt 50 \
    2>&1 | tee "$TRAIN_LOG_DIR/train.log"
