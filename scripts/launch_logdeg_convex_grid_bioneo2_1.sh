#!/usr/bin/env bash
# Launch the logdeg_convex production grid on bioneo2-1 (run ON the remote host).
# Scheme: flory_huggins potential + log_entropy convex split + degenerate mobility.
# Families: slide (long_base), impact (falling impact), papers (paper_wang + paper_air_shear).
# Existing polydeg_convex_*/logdeg_convex_* tmux sessions are killed; previous output dirs are archived.
set -euo pipefail

REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
PY="${PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"
MEM_FRACTION="${MEM_FRACTION:-0.05}"
NUM_GPUS="${NUM_GPUS:-8}"
ARCHIVE_TS="$(date +%Y%m%d_%H%M%S)"

cd "$REPO"

configs=(
  # slide
  configs/generated_sliding_long_realpaper_longrun/long_base_ca120_slip0p005_u1p0.json
  configs/generated_sliding_long_realpaper_longrun/long_base_ca120_slip0p05_u1p0.json
  # impact
  configs/generated_falling_impact/impact_h0p13_ca120_inertial.json
  configs/generated_falling_impact/impact_h0p13_ca60_inertial.json
  # papers
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca50_r0p12_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca50_r0p18_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca90_r0p12_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca90_r0p18_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca120_r0p12_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_wang_ca120_r0p18_u1p5.json
  configs/generated_sliding_long_realpaper_longrun/paper_air_shear_u0p5_ca90_r0p15.json
  configs/generated_sliding_long_realpaper_longrun/paper_air_shear_u1p0_ca90_r0p15.json
  configs/generated_sliding_long_realpaper_longrun/paper_air_shear_u1p5_ca90_r0p15.json
  configs/generated_sliding_long_realpaper_longrun/paper_air_shear_u2p0_ca90_r0p15.json
)

echo "=== Stopping existing polydeg_convex/logdeg_convex sessions ==="
while IFS= read -r session; do
  [[ "$session" == polydeg_convex_* || "$session" == logdeg_convex_* ]] || continue
  tmux kill-session -t "$session" && echo "killed $session"
done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
sleep 3

gpu=0
launched=0
for cfg in "${configs[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "MISSING CONFIG: $cfg" >&2
    exit 1
  fi
  stem="$(basename "$cfg" .json)"
  session="logdeg_convex_${stem}"
  out_dir="$DROP/experiment_logdeg_convex_${stem}"

  if [[ -d "$out_dir" ]]; then
    mv "$out_dir" "${out_dir}_old_${ARCHIVE_TS}"
    echo "archived ${out_dir} -> ${out_dir}_old_${ARCHIVE_TS}"
  fi
  mkdir -p "$out_dir"

  tmux new-session -d -s "$session" bash -lc "\
export PYTHONPATH=$REPO/src JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; \
cd $REPO && $PY -u main.py --config $cfg --output $out_dir 2>&1 | tee $out_dir/run.log"

  echo "LAUNCHED $session (gpu $gpu) <- $cfg"
  gpu=$(( (gpu + 1) % NUM_GPUS ))
  launched=$((launched + 1))
done

echo "=== Launched $launched sessions ==="
tmux list-sessions
