#!/usr/bin/env python3
"""
A/B test contact-angle relaxation alpha on a checkpoint.

Runs short main-simulation replays from the same checkpoint with different
phase-field bottom contact-angle relaxation values and compares divergence.
"""

import argparse
import csv
import json
import os
import subprocess
import sys


def _read_last_metrics(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main() -> None:
    p = argparse.ArgumentParser(description="A/B test contact-angle relaxation alpha")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--alpha_off", type=float, default=0.5)
    p.add_argument("--alpha_on", type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    results = {}
    for mode, alpha in (("off", args.alpha_off), ("on", args.alpha_on)):
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("restart", {})["restart_from"] = args.checkpoint
        cfg.setdefault("boundary_conditions", {}).setdefault("phase_field", {})[
            "contact_angle_relaxation"
        ] = float(alpha)
        cfg.setdefault("time_params", {})["t_max"] = float(base_cfg["time_params"]["dt"]) * float(args.steps) * 1.2
        cfg.setdefault("time_params", {})["checkpoint_interval"] = max(5, int(args.steps // 2))

        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        cfg_path = os.path.join(mode_dir, f"cfg_{mode}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "main.py"),
            "--config",
            cfg_path,
            "--output",
            mode_dir,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        subprocess.run(cmd, check=True, env=env)

        telemetry_csv = os.path.join(mode_dir, "telemetry.csv")
        results[mode] = _read_last_metrics(telemetry_csv)

    def f(mode: str, key: str) -> float:
        try:
            return float(results[mode].get(key, "nan"))
        except Exception:
            return float("nan")

    print("\n=== Contact-angle relaxation A/B ===")
    print(f"alpha_off={args.alpha_off} alpha_on={args.alpha_on}")
    for key in ("max_div", "mean_div", "ppe_applied", "ppe_iterations", "time", "step"):
        print(f"{key:14s} off={f('off', key):12.6g}  on={f('on', key):12.6g}")


if __name__ == "__main__":
    main()

