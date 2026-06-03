#!/usr/bin/env python3
"""
Parameter sweep near instability onset (restart from a checkpoint).

Runs monitor_restart_bottom_instability.py for each scenario and summarizes:
  - end max_div / row1_max
  - growth slope of row1_max over run
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Scenario:
    name: str
    epsilon: float
    ppe_under_relax: float
    ppe_max_iterations: int
    cfl_number: float
    capillary_cfl_number: float


def _read_monitor(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _lin_slope(y: List[float]) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = 0.5 * (n - 1)
    y_mean = sum(y) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(y))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep onset params for bottom divergence")
    p.add_argument("--base_config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--stop_max_div", type=float, default=2500.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.base_config, "r") as f:
        base = json.load(f)

    scenarios = [
        Scenario("baseline", 0.020, 1.00, 2500, 0.10, 1.00),
        Scenario("eps_0p03", 0.030, 1.00, 2500, 0.10, 1.00),
        Scenario("eps_0p035", 0.035, 1.00, 2500, 0.10, 1.00),
        Scenario("eps03_ppe08", 0.030, 0.80, 1200, 0.10, 1.00),
        Scenario("eps03_ppe08_cfl", 0.030, 0.80, 1200, 0.08, 0.70),
        Scenario("eps035_ppe08_cfl", 0.035, 0.80, 1200, 0.08, 0.70),
    ]

    results = []
    monitor_script = os.path.join(os.path.dirname(__file__), "monitor_restart_bottom_instability.py")

    for sc in scenarios:
        sc_dir = os.path.join(args.output_dir, sc.name)
        os.makedirs(sc_dir, exist_ok=True)
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("restart", {})["restart_from"] = args.checkpoint
        cfg.setdefault("physical_params", {})["epsilon"] = sc.epsilon
        cfg.setdefault("solver_params", {}).setdefault("ppe", {})["under_relaxation"] = sc.ppe_under_relax
        cfg["solver_params"]["ppe"]["max_iterations"] = sc.ppe_max_iterations
        cfg.setdefault("time_params", {})["cfl_number"] = sc.cfl_number
        cfg["time_params"]["capillary_cfl_number"] = sc.capillary_cfl_number

        cfg_path = os.path.join(sc_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cmd = [
            sys.executable,
            monitor_script,
            "--config",
            cfg_path,
            "--checkpoint",
            args.checkpoint,
            "--steps",
            str(args.steps),
            "--output_dir",
            sc_dir,
            "--stop_max_div",
            str(args.stop_max_div),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, check=True, env=env)

        rows = _read_monitor(os.path.join(sc_dir, "monitor.csv"))
        row1 = [float(r["row1_max"]) for r in rows]
        maxd = [float(r["max_div"]) for r in rows]
        dts = [float(r["dt"]) for r in rows]
        result = {
            "scenario": sc.name,
            "steps_done": len(rows),
            "end_step": int(rows[-1]["step"]),
            "end_max_div": maxd[-1],
            "end_row1_max": row1[-1],
            "peak_max_div": max(maxd),
            "peak_row1_max": max(row1),
            "row1_slope_per_step": _lin_slope(row1),
            "avg_dt": sum(dts) / len(dts),
            "epsilon": sc.epsilon,
            "ppe_under_relax": sc.ppe_under_relax,
            "ppe_max_iterations": sc.ppe_max_iterations,
            "cfl_number": sc.cfl_number,
            "capillary_cfl_number": sc.capillary_cfl_number,
        }
        results.append(result)

    # Rank by lower peak row1 and lower slope
    results_sorted = sorted(results, key=lambda r: (r["peak_row1_max"], r["row1_slope_per_step"], r["end_row1_max"]))

    out_csv = os.path.join(args.output_dir, "sweep_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        w.writerows(results_sorted)

    print("\n=== Sweep summary (best first) ===")
    for r in results_sorted:
        print(
            f"{r['scenario']:18s} "
            f"peak_row1={r['peak_row1_max']:.3e} "
            f"slope={r['row1_slope_per_step']:.3e} "
            f"end_row1={r['end_row1_max']:.3e} "
            f"avg_dt={r['avg_dt']:.3e}"
        )
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

