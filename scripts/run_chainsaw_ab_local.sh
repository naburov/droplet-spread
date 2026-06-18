#!/usr/bin/env bash
# Run chainsaw A/B matrix locally (64^2 by default for speed).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

GRID="${GRID:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/_chainsaw_ab}"

"$PY" "$REPO/scripts/generate_chainsaw_ab_configs.py"

for variant in ref A_no_cox B_wall_energy C_no_smooth_kappa D_strict_ppe; do
  cfg="$REPO/configs/debug/chainsaw_ab/chainsaw_ab_${variant}_${GRID}.json"
  out="$OUT_ROOT/${variant}_${GRID}"
  mkdir -p "$out"
  echo "=== $variant (grid ${GRID}) -> $out ==="
  "$PY" -u "$REPO/main.py" --config "$cfg" --output "$out" 2>&1 | tee "$out/run.log"
done

echo "Done. Summarize with:"
echo "  $PY $REPO/diagnostics/summarize_chainsaw_ab.py $OUT_ROOT"
