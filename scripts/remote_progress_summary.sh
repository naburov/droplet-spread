#!/usr/bin/env bash
# Print progress table for production experiment dirs on bioneo2-1.
set -euo pipefail

DROP="${DROP:-/home/jovyan/shares/SR003.nfs2/naburov/drop}"
REPO="${REPO:-/home/jovyan/shares/SR003.nfs2/naburov/drop/droplet_spreading_modeling}"

printf "%-55s %8s %12s %8s %10s %8s %s\n" "experiment" "step" "t/t_max" "strip95" "ckpt" "plots" "status"
printf "%s\n" "$(printf '%.0s-' {1..120})"

for exp in "$DROP"/experiment_{longrun,terrain,impact}*; do
  [[ -d "$exp" ]] || continue
  name="$(basename "$exp")"
  name="${name#experiment_}"
  [[ -d "$exp/checkpoints" ]] || continue

  step="—"
  tfrac="—"
  if [[ -f "$exp/statistics.csv" ]]; then
    read -r step t tmax <<< "$(python3 - <<PY
import csv, json
exp = "$exp"
rows = list(csv.DictReader(open(f"{exp}/statistics.csv")))
if not rows:
    print("— — —")
    raise SystemExit
r = rows[-1]
step = r.get("step", r.get("Step", ""))
t = float(r.get("time", r.get("t", 0)))
try:
    cfg = json.load(open(f"{exp}/simulation_parameters.json"))
    tmax = float(cfg["time_params"]["t_max"])
except Exception:
    tmax = float("nan")
print(step, t, tmax)
PY
)"
    if [[ "$tmax" != "nan" && "$tmax" != "0" ]]; then
      tfrac="$(python3 -c "print(f'{float(\"$t\")/float(\"$tmax\"):.3f}')")"
    fi
  fi

  ckpt="$(ls -1t "$exp/checkpoints"/checkpoint_*.npz 2>/dev/null | head -1 | xargs -n1 basename 2>/dev/null || echo —)"
  nplots="$(ls -1 "$exp/visualization"/joint_plot_step_*.png 2>/dev/null | wc -l | tr -d ' ')"

  strip95="—"
  if [[ -f "$REPO/diagnostics/chainsaw_striping_score.py" && "$ckpt" != "—" ]]; then
    strip95="$(cd "$REPO" && PYTHONPATH=src python3 diagnostics/chainsaw_striping_score.py --checkpoints "$exp/checkpoints" 2>/dev/null | tail -1 | awk '{print $2}' || echo —)"
  fi

  status="stopped"
  if pgrep -f "main.py --resume $exp" >/dev/null 2>&1; then
    status="running"
  elif [[ -f "$exp/resume_ckpt100.log" ]] && tail -1 "$exp/resume_ckpt100.log" | grep -q ModuleNotFoundError; then
    status="crashed(import)"
  elif [[ -f "$exp/resume_ckpt100.log" ]] && tail -5 "$exp/resume_ckpt100.log" | grep -q "Simulation completed"; then
    status="done"
  fi

  short="${name:0:55}"
  printf "%-55s %8s %12s %8s %10s %8s %s\n" "$short" "$step" "$tfrac" "$strip95" "$ckpt" "$nplots" "$status"
done
