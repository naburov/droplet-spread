#!/usr/bin/env bash
# Quarantine striping checkpoints and resume stopped longrun jobs with explicit ghost delta + soft contact mask.
set -euo pipefail

REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
PY="${PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"

export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.05}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$REPO"
"$PY" scripts/patch_chainsaw_mitigation_configs.py \
  configs/generated_sliding_long_realpaper_longrun \
  configs/generated_sliding_terrain_realpaper_longrun

# Format: exp_dir|tmux_session|config_json|space-separated quarantine steps
CASE_LINES=(
  "experiment_longrun_fullwall_cons_paper_wang_ca120_r0p12_u1p5_|res_wang_ca120_r0p12|configs/generated_sliding_long_realpaper_longrun/paper_wang_ca120_r0p12_u1p5.json|5000 7500 10000 12500 15000 17500 20000"
  "experiment_terrain_longrun_fullwall_cons_terrain_groove_amp0p02_ca90_r0p15_u1p5_|res_terrain_groove02|configs/generated_sliding_terrain_realpaper_longrun/terrain_groove_amp0p02_ca90_r0p15_u1p5.json|15000 17500"
  "experiment_terrain_longrun_fullwall_cons_terrain_groove_amp0p04_ca120_r0p15_u1p5_|res_terrain_groove04_ca120|configs/generated_sliding_terrain_realpaper_longrun/terrain_groove_amp0p04_ca120_r0p15_u1p5.json|12500 15000 17500"
  "experiment_terrain_longrun_fullwall_cons_terrain_groove_amp0p04_ca90_r0p15_u1p5_|res_terrain_groove04_ca90|configs/generated_sliding_terrain_realpaper_longrun/terrain_groove_amp0p04_ca90_r0p15_u1p5.json|12500 15000 17500"
  "experiment_terrain_longrun_fullwall_cons_terrain_incline_5deg_ca90_r0p15_|res_terrain_incline5|configs/generated_sliding_terrain_realpaper_longrun/terrain_incline_5deg_ca90_r0p15.json|12500 15000 17500 20000 22500"
  "experiment_terrain_longrun_fullwall_cons_terrain_incline_10deg_ca90_r0p15_|res_terrain_incline10|configs/generated_sliding_terrain_realpaper_longrun/terrain_incline_10deg_ca90_r0p15.json|15000 17500"
  "experiment_terrain_longrun_fullwall_cons_terrain_incline_15deg_ca90_r0p15_|res_terrain_incline15|configs/generated_sliding_terrain_realpaper_longrun/terrain_incline_15deg_ca90_r0p15.json|12500 15000 17500"
)

for line in "${CASE_LINES[@]}"; do
  IFS='|' read -r exp session config quarantine_steps <<< "$line"
  exp_dir="$DROP/$exp"
  ckpt_dir="$exp_dir/checkpoints"
  bad_dir="$exp_dir/checkpoints_quarantined_striping"
  mkdir -p "$bad_dir"
  for step in $quarantine_steps; do
    f="$ckpt_dir/checkpoint_$(printf '%06d' "$step").npz"
    if [[ -f "$f" ]]; then
      mv "$f" "$bad_dir/"
      echo "quarantined $(basename "$f") for $exp"
    fi
  done
  latest=$(ls -1t "$ckpt_dir"/checkpoint_*.npz 2>/dev/null | head -1 || true)
  if [[ -z "$latest" ]]; then
    echo "SKIP $exp: no checkpoints left after quarantine"
    continue
  fi
  echo "resume from $(basename "$latest") for $exp"
  cp "$REPO/$config" "$exp_dir/simulation_parameters.json"
  echo "updated simulation_parameters.json for $exp from $config"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" bash -lc \
    "export PYTHONPATH=$REPO/src JAX_PLATFORMS=$JAX_PLATFORMS XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE; cd $REPO && $PY -u main.py --resume $exp_dir 2>&1 | tee -a $exp_dir/resume_explicit_delta.log"
  echo "resumed tmux session $session"
done

echo "Done."
