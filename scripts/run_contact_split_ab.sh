#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

GRID="${GRID:-64}"
OUT_ROOT="${OUT_ROOT:-$REPO/diagnostics/_contact_split_ab}"

"$PY" "$REPO/scripts/generate_chainsaw_ab_configs.py" 2>&1 | grep fast_split || true

VARIANTS=(
  fast_split_explicit_delta
  fast_split_no_delta
  fast_split_filtered_delta
  fast_split_damped_delta_beta0
  fast_split_damped_delta_beta025
  fast_split_damped_delta_beta05
  fast_split_implicit_wall_energy
  fast_split_explicit_ghost
)

for variant in "${VARIANTS[@]}"; do
  cfg="$REPO/configs/debug/chainsaw_ab/chainsaw_ab_${variant}_${GRID}.json"
  out="$OUT_ROOT/${variant}_${GRID}"
  mkdir -p "$out"
  echo "=== $variant ==="
  "$PY" -u "$REPO/main.py" --config "$cfg" --output "$out" 2>&1 | tee "$out/run.log"
done

"$PY" "$REPO/diagnostics/summarize_contact_split_ab.py" "$OUT_ROOT" --step 400
"$PY" "$REPO/diagnostics/summarize_contact_split_ab.py" "$OUT_ROOT" --step 800
