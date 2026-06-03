#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bioneo2-1}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_ROOT/droplet_spreading_modeling}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"
CONFIG_DIR="${CONFIG_DIR:-configs/generated_falling_impact}"
LOG_DIR_NAME="${LOG_DIR_NAME:-impact_logs}"
MEM_FRACTION="${MEM_FRACTION:-0.08}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/$LOG_DIR_NAME'"

mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.json' | sort)

gpu=0
for cfg in "${configs[@]}"; do
  name="$(basename "$cfg" .json)"
  session="impact_${name}"
  out_dir="$REMOTE_ROOT/experiment_${name}"
  log_file="$REMOTE_ROOT/$LOG_DIR_NAME/${name}.log"
  remote_cfg="$REMOTE_REPO/${cfg}"

  ssh "$REMOTE_HOST" "tmux has-session -t '$session' 2>/dev/null && tmux kill-session -t '$session' || true"

  ssh "$REMOTE_HOST" "tmux new-session -d -s '$session' \
    'cd \"$REMOTE_REPO\" && \
     export PYTHONPATH=src && \
     export CUDA_VISIBLE_DEVICES=$gpu && \
     export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
     export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION && \
     export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 JAX_ENABLE_X64=1 && \
     \"$REMOTE_PYTHON\" main.py --config \"$remote_cfg\" --output \"$out_dir\" > \"$log_file\" 2>&1'"

  gpu=$((1 - gpu))
done

echo "Launched ${#configs[@]} falling-impact sessions on $REMOTE_HOST"
