#!/usr/bin/env python3
"""
A/B test for top-strip compatibility constraint in staggered BCs.

Runs `analyze_ppe_iteration_strips.py` twice on the same checkpoint:
  1) strip_compatibility_top = false
  2) strip_compatibility_top = true

and prints a compact comparison from the last recorded PPE iteration.
"""

import argparse
import csv
import json
import os
import subprocess
import sys


def _read_last_row(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test top strip compatibility")
    p.add_argument("--config", required=True, help="Base config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--output_dir", required=True, help="Output directory for A/B")
    p.add_argument("--max_iters", type=int, default=60)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--strip_width", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    analyzer = os.path.join(os.path.dirname(__file__), "analyze_ppe_iteration_strips.py")
    results = {}

    for mode in ("off", "on"):
        cfg = json.loads(json.dumps(base_cfg))
        vel = cfg.setdefault("boundary_conditions", {}).setdefault("velocity", {})
        vel["strip_compatibility_top"] = (mode == "on")
        vel["strip_compatibility_width"] = int(args.strip_width)

        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        cfg_path = os.path.join(mode_dir, f"config_strip_{mode}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cmd = [
            sys.executable,
            analyzer,
            "--config",
            cfg_path,
            "--checkpoint",
            args.checkpoint,
            "--output_dir",
            mode_dir,
            "--max_iters",
            str(args.max_iters),
        ]
        if args.dt is not None:
            cmd.extend(["--dt", str(args.dt)])

        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, check=True, env=env)

        last = _read_last_row(os.path.join(mode_dir, "ppe_iteration_metrics.csv"))
        results[mode] = last

    def f(mode: str, key: str) -> float:
        v = results.get(mode, {}).get(key, "nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    print("\n=== Strip compatibility A/B (last recorded PPE iter) ===")
    for key in ("max_div", "max_div_interior", "tl5_max", "tl5_mean", "hot_abs_div", "hot_du", "hot_dv"):
        print(
            f"{key:18s} off={f('off', key):12.6g}  on={f('on', key):12.6g}"
        )
    print(
        "hotspot idx:",
        f"off=({results.get('off', {}).get('hot_i')},{results.get('off', {}).get('hot_j')})",
        f"on=({results.get('on', {}).get('hot_i')},{results.get('on', {}).get('hot_j')})",
    )


if __name__ == "__main__":
    main()

