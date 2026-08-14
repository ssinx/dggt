#!/usr/bin/env bash
#SBATCH --job-name=dggt-sky
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
# A fresh directory per Slurm submission keeps checkpoints, images, metrics,
# TensorBoard events, and train.log from replacing an earlier run.
RUN_NAME="sky_only_${SLURM_JOB_ID}"
TRAIN_LOG_DIR="$PROJECT_DIR/logs/$RUN_NAME"
TENSORBOARD_PORT=$((6006 + SLURM_JOB_ID % 1000))
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
SKY_MASK_CACHE_DIR=/scratch/junyizh3/waymo_sky_masks

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dggt

cd "$PROJECT_DIR"
mkdir -p "$TRAIN_LOG_DIR"

python -c 'import tensorboard' || {
    echo "TensorBoard is missing. Install it before submission with:" >&2
    echo "/data/user_data/junyizh3/miniconda3/envs/dggt/bin/pip install tensorboard" >&2
    exit 1
}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Run directory: $TRAIN_LOG_DIR"
echo "TensorBoard logdir: $TRAIN_LOG_DIR/tensorboard"
echo "TensorBoard port: $TENSORBOARD_PORT (compute-node loopback only)"

tensorboard \
    --logdir "$TRAIN_LOG_DIR/tensorboard" \
    --host 127.0.0.1 \
    --port "$TENSORBOARD_PORT" \
    > "$TRAIN_LOG_DIR/tensorboard.log" 2>&1 &
TENSORBOARD_PID=$!

cleanup() {
    kill "$TENSORBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

torchrun \
    --nproc_per_node=1 \
    --master_port="$MASTER_PORT" \
    train.py \
    --raw_waymo_dir /data/datasets/waymo/waymo-open-dataset-v2.0.1/training \
    --sky_mask_cache_dir "$SKY_MASK_CACHE_DIR" \
    --raw_sequence_lengths 4 8 12 \
    --ckpt_path pretrained/model_latest_waymo.pt \
    --log_dir "$TRAIN_LOG_DIR" \
    --sky_model cubemap \
    --batch_size 1 \
    --max_epoch 5 \
    --log_every 10 \
    --save_image 50 \
    --save_ckpt 50 \
    2>&1 | tee "$TRAIN_LOG_DIR/train.log"
