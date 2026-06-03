#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-spark-b92e}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/naburov/droplet_spreading_modeling}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/naburov/venvs/droplet-sim/bin/python}"
LOCAL_DOWNLOAD_DIR="${LOCAL_DOWNLOAD_DIR:-$HOME/Downloads}"
GIF_NAME="${GIF_NAME:-joint_plot_animation.gif}"
GIF_DURATION_MS="${GIF_DURATION_MS:-10}"
FFMPEG_FPS="${FFMPEG_FPS:-10}"
FFMPEG_CRF="${FFMPEG_CRF:-20}"

usage() {
  cat <<'EOF'
Usage:
  ./create_jointplot_media_from_spark.sh
  ./create_jointplot_media_from_spark.sh <remote_experiment[:local_prefix]> [...]

Examples:
  ./create_jointplot_media_from_spark.sh
  ./create_jointplot_media_from_spark.sh \
    experiment_ca120_:static_ca120 \
    experiment_ca60_:static_ca60 \
    experiment_ca120_2d_20260313_001408:static_ca120_2d

Defaults (when no args are given):
  experiment_ca120_:static_ca120
  experiment_ca60_:static_ca60
  experiment_ca120_2d_20260313_001408:static_ca120_2d
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$LOCAL_DOWNLOAD_DIR"

declare -a TARGETS=()
if [[ "$#" -eq 0 ]]; then
  TARGETS+=(
    "experiment_ca120_:static_ca120"
    "experiment_ca60_:static_ca60"
    "experiment_ca120_2d_20260313_001408:static_ca120_2d"
  )
else
  TARGETS=("$@")
fi

echo "Syncing GIF helper to ${REMOTE_HOST}..."
scp -o BatchMode=yes create_gif_from_experiment.py "${REMOTE_HOST}:${REMOTE_ROOT}/create_gif_from_experiment.py"

for target in "${TARGETS[@]}"; do
  remote_experiment="${target%%:*}"
  local_prefix="${target#*:}"
  if [[ "$local_prefix" == "$target" ]]; then
    local_prefix="$remote_experiment"
  fi

  echo
  echo "=== ${remote_experiment} -> ${local_prefix} ==="

  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '${REMOTE_ROOT}' && '${REMOTE_PYTHON}' create_gif_from_experiment.py '${remote_experiment}' '${GIF_NAME}' '${GIF_DURATION_MS}'"

  local_gif="${LOCAL_DOWNLOAD_DIR}/${local_prefix}_joint_plot_animation.gif"
  local_mp4="${LOCAL_DOWNLOAD_DIR}/${local_prefix}_joint_plot_animation.mp4"

  scp -o BatchMode=yes \
    "${REMOTE_HOST}:${REMOTE_ROOT}/${remote_experiment}/${GIF_NAME}" \
    "$local_gif"

  ffmpeg -y -i "$local_gif" \
    -movflags +faststart \
    -pix_fmt yuv420p \
    -vf "fps=${FFMPEG_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos" \
    -c:v libx264 \
    -crf "${FFMPEG_CRF}" \
    "$local_mp4" >/tmp/"$(basename "$local_mp4")".log 2>&1

  ls -lh "$local_gif" "$local_mp4"
done

echo
echo "Done. Media files are in ${LOCAL_DOWNLOAD_DIR}"
