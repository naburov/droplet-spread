#!/usr/bin/env bash
# Progress summary + download latest joint plots, GIF/MP4, telemetry for running experiments.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bioneo2-1}"
REMOTE_DROP="${REMOTE_DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
REMOTE_PY="${REMOTE_PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/Downloads/running_grid_media_$(date +%Y%m%d)}"
GIF_MS="${GIF_MS:-80}"
FFMPEG_FPS="${FFMPEG_FPS:-8}"

mkdir -p "$LOCAL_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Collect running experiment dirs on ${REMOTE_HOST} ==="
EXPERIMENTS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && EXPERIMENTS+=("$line")
done < <(
  ssh -o BatchMode=yes "$REMOTE_HOST" "ps aux | grep '[m]ain.py --resume' | sed -n 's/.*--resume \\([^ ]*\\).*/\\1/p' | sort -u"
)

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  echo "No --resume processes found."
  exit 1
fi

echo "Found ${#EXPERIMENTS[@]} experiments"
PROGRESS_CSV="$LOCAL_DIR/progress_summary.csv"
echo "experiment,mitigated,step,time,phi_min,phi_max,st_max,latest_ckpt,latest_plot,n_plots" > "$PROGRESS_CSV"

scp -o BatchMode=yes "$REPO_LOCAL/create_gif_from_experiment.py" \
  "${REMOTE_HOST}:${REPO}/create_gif_from_experiment.py"

for exp_dir in "${EXPERIMENTS[@]}"; do
  name="$(basename "$exp_dir")"
  echo ""
  echo "=== ${name} ==="

  mitigated=""
  ssh -o BatchMode=yes "$REMOTE_HOST" "test -f '${exp_dir}/resume_mitigated.log'" && mitigated="yes"

  # Progress row
  ssh -o BatchMode=yes "$REMOTE_HOST" "PY='$REMOTE_PY'; REPO='$REPO'
name='$name'
exp='$exp_dir'
mit='$mitigated'
if [ -f \"\$exp/statistics.csv\" ]; then
  tail1=\$(tail -1 \"\$exp/statistics.csv\")
  step=\$(echo \"\$tail1\" | cut -d, -f1)
  t=\$(echo \"\$tail1\" | cut -d, -f2)
  pmin=\$(echo \"\$tail1\" | cut -d, -f4)
  pmax=\$(echo \"\$tail1\" | cut -d, -f5)
  st=\$(echo \"\$tail1\" | cut -d, -f18)
else
  step=t=pmin=pmax=st=NA
fi
ckpt=\$(ls -1 \"\$exp/checkpoints\"/checkpoint_*.npz 2>/dev/null | tail -1 | xargs basename 2>/dev/null || echo none)
plot=\$(ls -1 \"\$exp/visualization\"/joint_plot_step_*.png 2>/dev/null | tail -1 | xargs basename 2>/dev/null || echo none)
nplots=\$(ls -1 \"\$exp/visualization\"/joint_plot_step_*.png 2>/dev/null | wc -l | tr -d ' ')
echo \"\$name,\$mit,\$step,\$t,\$pmin,\$pmax,\$st,\$ckpt,\$plot,\$nplots\"
cd \"\$REPO\" && \"\$PY\" create_gif_from_experiment.py \"\$exp\" joint_plot_animation.gif \"$GIF_MS\" 2>/dev/null || true
" >> "$PROGRESS_CSV"

  local_exp="$LOCAL_DIR/$name"
  mkdir -p "$local_exp/visualization"

  # Latest joint plot + all plots for mp4 (cap at 40 frames if many)
  ssh -o BatchMode=yes "$REMOTE_HOST" "ls -1 '${exp_dir}/visualization'/joint_plot_step_*.png 2>/dev/null | sort -t_ -k3 -n" \
    | tail -40 > "$local_exp/_plot_list.txt" || true

  if [[ -s "$local_exp/_plot_list.txt" ]]; then
    while IFS= read -r remote_png; do
      base=$(basename "$remote_png")
      scp -o BatchMode=yes "${REMOTE_HOST}:${remote_png}" "$local_exp/visualization/$base" 2>/dev/null || true
    done < "$local_exp/_plot_list.txt"
  fi

  latest_local=$(ls -1 "$local_exp/visualization"/joint_plot_step_*.png 2>/dev/null | sort -t_ -k3 -n | tail -1)
  if [[ -n "${latest_local:-}" ]]; then
    cp "$latest_local" "$LOCAL_DIR/${name}_latest.png"
  fi

  scp -o BatchMode=yes "${REMOTE_HOST}:${exp_dir}/joint_plot_animation.gif" \
    "$local_exp/joint_plot_animation.gif" 2>/dev/null || true

  if [[ -f "$local_exp/joint_plot_animation.gif" ]] && command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -i "$local_exp/joint_plot_animation.gif" \
      -movflags +faststart -pix_fmt yuv420p \
      -vf "fps=${FFMPEG_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2" \
      -c:v libx264 -crf 22 \
      "$LOCAL_DIR/${name}_preview.mp4" || true
  elif ls "$local_exp/visualization"/joint_plot_step_*.png >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -framerate "$FFMPEG_FPS" \
      -pattern_type glob -i "$local_exp/visualization/joint_plot_step_*.png" \
      -movflags +faststart -pix_fmt yuv420p \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
      -c:v libx264 -crf 22 \
      "$LOCAL_DIR/${name}_preview.mp4" || true
  fi

  scp -o BatchMode=yes \
    "${REMOTE_HOST}:${exp_dir}/statistics.csv" \
    "$local_exp/statistics.csv" 2>/dev/null || true

  for telem in statistics_plots.png boundary_statistics_plots.png ppe_updates_plots.png; do
    scp -o BatchMode=yes \
      "${REMOTE_HOST}:${exp_dir}/visualization/${telem}" \
      "$local_exp/visualization/${telem}" 2>/dev/null || true
  done

  if [[ "$mitigated" == "yes" ]]; then
    scp -o BatchMode=yes "${REMOTE_HOST}:${exp_dir}/resume_mitigated.log" \
      "$local_exp/resume_mitigated.log" 2>/dev/null || true
  fi
done

echo ""
echo "Progress table: $PROGRESS_CSV"
echo "Media directory: $LOCAL_DIR"
ls -lh "$LOCAL_DIR"/*.mp4 2>/dev/null | head -20 || true
