#!/usr/bin/env bash
# Stop every simulation on bioneo2-1 and resume all production jobs with patched configs (ckpt every 100 steps).
set -euo pipefail

REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
PY="${PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"

export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.05}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$REPO"
echo "=== Patching repo configs (no_delta, soft mask, checkpoint_interval=100) ==="
"$PY" scripts/patch_chainsaw_mitigation_configs.py \
  configs/generated_sliding_long_realpaper_longrun \
  configs/generated_sliding_terrain_realpaper_longrun \
  configs/generated_falling_impact

echo "=== Stopping all tmux sessions and main.py processes ==="
while IFS= read -r session; do
  [[ -n "$session" ]] || continue
  tmux kill-session -t "$session" 2>/dev/null || true
done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
pkill -f "main.py" 2>/dev/null || true
sleep 2
pkill -9 -f "main.py" 2>/dev/null || true

shopt -s nullglob
CONFIG_DIRS=(
  "$REPO/configs/generated_sliding_long_realpaper_longrun"
  "$REPO/configs/generated_sliding_terrain_realpaper_longrun"
  "$REPO/configs/generated_falling_impact"
)

resolve_exp_dir() {
  local stem="$1"
  local suffixes=("" "_")
  local prefixes=(
    "experiment_${stem}"
    "experiment_longrun_fullwall_cons_${stem}"
    "experiment_terrain_longrun_fullwall_cons_${stem}"
    "experiment_terrain_fullwall_cons_${stem}"
  )
  local p suf d
  for p in "${prefixes[@]}"; do
    for suf in "${suffixes[@]}"; do
      d="$DROP/${p}${suf}"
      if [[ -d "$d" ]]; then
        echo "$d"
        return 0
      fi
    done
  done
  return 1
}

session_name_for() {
  local exp_dir="$1"
  local base
  base="$(basename "$exp_dir")"
  base="${base#experiment_}"
  echo "res_${base}"
}

launched=0
skipped=0

for cfg_dir in "${CONFIG_DIRS[@]}"; do
  for cfg in "$cfg_dir"/*.json; do
    stem="$(basename "$cfg" .json)"
    if ! exp_dir="$(resolve_exp_dir "$stem")"; then
      echo "SKIP (no experiment dir): $stem"
      skipped=$((skipped + 1))
      continue
    fi
    if [[ ! -d "$exp_dir/checkpoints" ]] || [[ -z "$(ls -A "$exp_dir/checkpoints"/checkpoint_*.npz 2>/dev/null)" ]]; then
      echo "SKIP (no checkpoints): $exp_dir"
      skipped=$((skipped + 1))
      continue
    fi
    latest="$(ls -1t "$exp_dir/checkpoints"/checkpoint_*.npz 2>/dev/null | head -1)"
    session="$(session_name_for "$exp_dir")"
    cp "$cfg" "$exp_dir/simulation_parameters.json"
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" bash -lc \
      "export PYTHONPATH=$REPO/src JAX_PLATFORMS=$JAX_PLATFORMS XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE; cd $REPO && $PY -u main.py --resume $exp_dir 2>&1 | tee -a $exp_dir/resume_ckpt100.log"
    echo "LAUNCHED $session <- $(basename "$latest") ($exp_dir)"
    launched=$((launched + 1))
  done
done

echo "=== Done: launched=$launched skipped=$skipped ==="
echo "Active sessions:"
tmux list-sessions 2>/dev/null || echo "(none)"
