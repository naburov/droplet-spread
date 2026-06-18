#!/usr/bin/env bash
# Compare accelerated chainsaw onset: full_wall analytic vs wall_energy mitigation.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

GRID="${GRID:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/_chainsaw_fast}"

"$PY" "$REPO/scripts/generate_chainsaw_ab_configs.py" --fast-only

VARIANTS=(
  fast_analytic_fullwall
  fast_analytic_masked
  fast_wall_energy
  fast_no_cox_voinov
  fast_analytic_fullwall_no_surface_tension_bc_overwrite
  fast_analytic_fullwall_no_curvature_smoothing
)

for variant in "${VARIANTS[@]}"; do
  cfg="$REPO/configs/debug/chainsaw_ab/chainsaw_ab_${variant}_${GRID}.json"
  out="$OUT_ROOT/${variant}_${GRID}"
  mkdir -p "$out"
  echo "=== $variant -> $out ==="
  "$PY" -u "$REPO/main.py" --config "$cfg" --output "$out" 2>&1 | tee "$out/run.log"
done

echo "Strip95 timeline:"
for d in "$OUT_ROOT"/*_${GRID}; do
  echo "--- $(basename "$d") ---"
  "$PY" "$REPO/diagnostics/chainsaw_striping_score.py" --checkpoints "$d/checkpoints" 2>/dev/null || true
done

echo ""
echo "Lead-lag table (ghost vs phi0 vs chainsaw):"
"$PY" "$REPO/diagnostics/summarize_chainsaw_ghost_leadlag.py" "$OUT_ROOT"
