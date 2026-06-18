#!/usr/bin/env bash
# Retry download + MP4 for experiments listed as gif_build_failed in manifest.csv
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-bioneo2-1}"
DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
FFMPEG_FPS="${FFMPEG_FPS:-8}"

OUT_ROOT="${1:?Usage: $0 <downloads_bundle_dir>}"
MANIFEST="$OUT_ROOT/manifest.csv"
GIF_DIR="$OUT_ROOT/gifs"
MP4_DIR="$OUT_ROOT/mp4"

while IFS=, read -r short _ _ _ status || [[ -n "${short:-}" ]]; do
  status="${status//$'\r'/}"
  [[ "$status" == "gif_build_failed" ]] || continue
  safe="${short}"
  exp_dir="${DROP}/experiment_${short}"
  remote_gif="${exp_dir}/joint_plot_animation.gif"
  local_gif="$GIF_DIR/${safe}.gif"
  local_mp4="$MP4_DIR/${safe}.mp4"

  echo "=== retry $short ==="
  if ! ssh -n -o BatchMode=yes "$REMOTE_HOST" "test -s '${remote_gif}'"; then
    echo "  still no remote GIF"
    continue
  fi
  scp -o BatchMode=yes "${REMOTE_HOST}:${remote_gif}" "$local_gif"
  ffmpeg -y -loglevel error -i "$local_gif" \
    -movflags +faststart -pix_fmt yuv420p \
    -vf "fps=${FFMPEG_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -crf 20 \
    "$local_mp4"
  gif_mb="$(python3 -c "import os; print(f'{os.path.getsize(\"$local_gif\")/1024/1024:.2f}')")"
  mp4_mb="$(python3 -c "import os; print(f'{os.path.getsize(\"$local_mp4\")/1024/1024:.2f}')")"
  nplots="$(ssh -n -o BatchMode=yes "$REMOTE_HOST" "ls -1 '${exp_dir}/visualization'/joint_plot_step_*.png 2>/dev/null | wc -l | tr -d ' '")"
  # rewrite manifest line (append ok row; user can sort by hand)
  echo "$short,$nplots,$gif_mb,$mp4_mb,ok_retry" >> "$OUT_ROOT/manifest_retry.csv"
  echo "  OK gif=${gif_mb}MB mp4=${mp4_mb}MB"
done < <(tail -n +2 "$MANIFEST")
