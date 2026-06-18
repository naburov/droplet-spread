#!/usr/bin/env bash
# Stage-by-stage phi0_alt tracing + surgical A/B toggles.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

GRID="${GRID:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/_chainsaw_stage_ab}"

"$PY" "$REPO/scripts/generate_chainsaw_ab_configs.py" 2>&1 | grep -E "stage|wrote.*stage" || true

VARIANTS=(
  fast_analytic_fullwall_stage
  fast_analytic_fullwall_freeze_wall_after_solve
  fast_analytic_fullwall_zero_phase_advection
  fast_analytic_fullwall_skip_ch_diffusion
  fast_analytic_fullwall_no_conserve_phi_sum
  fast_analytic_fullwall_explicit_monolithic
)

for variant in "${VARIANTS[@]}"; do
  cfg="$REPO/configs/debug/chainsaw_ab/chainsaw_ab_${variant}_${GRID}.json"
  out="$OUT_ROOT/${variant}_${GRID}"
  mkdir -p "$out"
  echo "=== $variant -> $out ==="
  "$PY" -u "$REPO/main.py" --config "$cfg" --output "$out" 2>&1 | tee "$out/run.log"
done

echo ""
"$PY" "$REPO/diagnostics/summarize_phase_stage_onset.py" "$OUT_ROOT"

echo ""
echo "strip95:"
for d in "$OUT_ROOT"/*_${GRID}; do
  echo "--- $(basename "$d") ---"
  "$PY" "$REPO/diagnostics/chainsaw_striping_score.py" --checkpoints "$d/checkpoints" 2>/dev/null || true
done
