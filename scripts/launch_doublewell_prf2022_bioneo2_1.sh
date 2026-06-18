#!/usr/bin/env bash
# Launch Chahine PRF 2022-inspired air-shear sweep with polynomial double-well CH.
# Run on bioneo2-1 remote host.
set -euo pipefail

REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"
DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
PY="${PY:-/home/jovyan/naburov/venvs/vlm-cu126/bin/python}"
MEM_FRACTION="${MEM_FRACTION:-0.08}"
NUM_GPUS="${NUM_GPUS:-2}"
ARCHIVE_TS="$(date +%Y%m%d_%H%M%S)"

cd "$REPO"

configs=(
  configs/generated_sliding_prf2022_doublewell/doublewell_prf2022_air_shear_u0p5_ca90_r0p15.json
  configs/generated_sliding_prf2022_doublewell/doublewell_prf2022_air_shear_u1p0_ca90_r0p15.json
  configs/generated_sliding_prf2022_doublewell/doublewell_prf2022_air_shear_u1p5_ca90_r0p15.json
  configs/generated_sliding_prf2022_doublewell/doublewell_prf2022_air_shear_u2p0_ca90_r0p15.json
)

echo "=== Stopping existing doublewell_prf2022 sessions ==="
while IFS= read -r session; do
  [[ "$session" == doublewell_prf2022_* ]] || continue
  tmux kill-session -t "$session" && echo "killed $session"
done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)

gpu=0
launched=0
for cfg in "${configs[@]}"; do
  [[ -f "$cfg" ]] || { echo "MISSING CONFIG: $cfg" >&2; exit 1; }
  stem="$(basename "$cfg" .json)"
  session="doublewell_prf2022_${stem#doublewell_prf2022_}"
  out_dir="$DROP/experiment_doublewell_prf2022_${stem#doublewell_prf2022_}"

  if [[ -d "$out_dir" ]]; then
    mv "$out_dir" "${out_dir}_old_${ARCHIVE_TS}"
    echo "archived ${out_dir} -> ${out_dir}_old_${ARCHIVE_TS}"
  fi
  mkdir -p "$out_dir"

  tmux new-session -d -s "$session" bash -lc "\
export PYTHONPATH=$REPO/src JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$gpu \
XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4; \
cd $REPO && $PY -u main.py --config $cfg --output $out_dir 2>&1 | tee $out_dir/run.log"

  echo "LAUNCHED $session (gpu $gpu) -> $out_dir"
  gpu=$(( (gpu + 1) % NUM_GPUS ))
  launched=$((launched + 1))
done

echo "=== Launched $launched doublewell PRF2022 sessions ==="
tmux list-sessions 2>/dev/null || true
