#!/usr/bin/env bash
# For each running experiment on bioneo2-1: build GIF on server, download to ~/Downloads, make MP4 locally.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bioneo2-1}"
REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
REMOTE_PY="${REMOTE_PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"
GIF_MS="${GIF_MS:-100}"
FFMPEG_FPS="${FFMPEG_FPS:-8}"
MAX_FRAMES="${MAX_FRAMES:-120}"

STAMP="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-$HOME/Downloads/bioneo_experiment_animations_${STAMP}}"
GIF_DIR="$OUT_ROOT/gifs"
MP4_DIR="$OUT_ROOT/mp4"
mkdir -p "$GIF_DIR" "$MP4_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Output: $OUT_ROOT"

mapfile -t EXPERIMENTS < <(
  ssh -n -o BatchMode=yes "$REMOTE_HOST" \
    "ps aux | grep '[m]ain.py --resume' | sed -n 's/.*--resume \\([^ ]*\\).*/\\1/p' | sort -u"
)

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  echo "No running --resume experiments found."
  exit 1
fi

echo "Found ${#EXPERIMENTS[@]} running experiments"

scp -o BatchMode=yes "$REPO_LOCAL/create_gif_from_experiment.py" "${REMOTE_HOST}:${REPO}/create_gif_from_experiment.py"

MANIFEST="$OUT_ROOT/manifest.csv"
echo "experiment,n_png,gif_mb,mp4_mb,status" > "$MANIFEST"

for exp_dir in "${EXPERIMENTS[@]}"; do
  name="$(basename "$exp_dir")"
  short="${name#experiment_}"
  safe="${short//\//_}"
  echo ""
  echo "=== $short ==="

  nplots="$(ssh -n -o BatchMode=yes "$REMOTE_HOST" \
    "ls -1 '${exp_dir}/visualization'/joint_plot_step_*.png 2>/dev/null | wc -l | tr -d ' '" || echo 0)"

  if [[ "$nplots" -lt 2 ]]; then
    echo "SKIP: fewer than 2 joint plots ($nplots)"
    echo "$short,$nplots,0,0,skip_few_frames" >> "$MANIFEST"
    continue
  fi

  remote_gif="${exp_dir}/joint_plot_animation.gif"
  ssh -n -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REPO' && '$REMOTE_PY' create_gif_from_experiment.py '$exp_dir' joint_plot_animation.gif '$GIF_MS' '$MAX_FRAMES'" \
    || true
  if ! ssh -n -o BatchMode=yes "$REMOTE_HOST" "test -s '${remote_gif}'"; then
    echo "FAIL: remote GIF missing or empty"
    echo "$short,$nplots,0,0,gif_build_failed" >> "$MANIFEST"
    continue
  fi

  local_gif="$GIF_DIR/${safe}.gif"
  if ! scp -o BatchMode=yes "${REMOTE_HOST}:${exp_dir}/joint_plot_animation.gif" "$local_gif"; then
    echo "FAIL: download GIF"
    echo "$short,$nplots,0,0,gif_download_failed" >> "$MANIFEST"
    continue
  fi

  gif_mb="$(python3 -c "import os; print(f'{os.path.getsize(\"$local_gif\")/1024/1024:.2f}')")"
  local_mp4="$MP4_DIR/${safe}.mp4"
  mp4_ok=no
  if command -v ffmpeg >/dev/null 2>&1; then
    if ffmpeg -y -loglevel error -i "$local_gif" \
      -movflags +faststart -pix_fmt yuv420p \
      -vf "fps=${FFMPEG_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2" \
      -c:v libx264 -crf 20 \
      "$local_mp4"; then
      mp4_ok=yes
    fi
  else
    echo "WARN: ffmpeg not installed"
  fi

  mp4_mb="0"
  [[ "$mp4_ok" == yes ]] && mp4_mb="$(python3 -c "import os; print(f'{os.path.getsize(\"$local_mp4\")/1024/1024:.2f}')")"

  echo "GIF: $local_gif (${gif_mb} MB)"
  [[ "$mp4_ok" == yes ]] && echo "MP4: $local_mp4 (${mp4_mb} MB)"
  echo "$short,$nplots,$gif_mb,$mp4_mb,ok" >> "$MANIFEST"
done

echo ""
echo "Done."
echo "GIFs:  $GIF_DIR"
echo "MP4s:  $MP4_DIR"
echo "Manifest: $MANIFEST"
