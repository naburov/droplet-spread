#!/usr/bin/env python3
"""Table: chainsaw strip95 on latest ckpt + mass drift for running experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np


def striping_score(phi: np.ndarray) -> float:
    mask = np.abs(phi) < 0.9
    if not mask.any():
        return 0.0
    d2 = phi[2:, :] - 2.0 * phi[1:-1, :] + phi[:-2, :]
    m = mask[1:-1, :]
    values = np.abs(d2[m])
    return float(np.percentile(values, 95)) if values.size else 0.0


def load_rows(stats_path: str) -> list[dict[str, str]]:
    with open(stats_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mass_drift(rows: list[dict[str, str]]) -> tuple[float, float]:
    """Mass drift from the first statistics row (experiment start) to the latest."""
    if not rows:
        return float("nan"), float("nan")
    base = float(rows[0]["droplet_mass"])
    massN = float(rows[-1]["droplet_mass"])
    drift_abs = massN - base
    drift_rel = drift_abs / base if abs(base) > 1e-14 else float("nan")
    return drift_abs, drift_rel


def analyze_experiment(exp_dir: str) -> dict[str, object]:
    name = os.path.basename(exp_dir.rstrip("/"))
    stats_path = os.path.join(exp_dir, "statistics.csv")
    row: dict[str, object] = {
        "run_name": name,
        "chainsaw_metric": float("nan"),
        "mass_drift_abs": float("nan"),
        "mass_drift_rel": float("nan"),
    }
    if not os.path.isfile(stats_path):
        return row

    rows = load_rows(stats_path)
    if not rows:
        return row

    drift_abs, drift_rel = mass_drift(rows)
    row["mass_drift_abs"] = drift_abs
    row["mass_drift_rel"] = drift_rel

    ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "checkpoint_*.npz")))
    if ckpts:
        row["chainsaw_metric"] = striping_score(np.load(ckpts[-1])["phi"])

    return row


def discover_running(drop_root: str) -> list[str]:
    out = subprocess.check_output(
        "ps aux | grep '[m]ain.py --resume' | sed -n 's/.*--resume \\([^ ]*\\).*/\\1/p' | sort -u",
        shell=True,
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-root", default=os.environ.get("DROP", "/home/jovyan/shares/SR003.nfs2/naburov/drop"))
    parser.add_argument("--experiment", action="append", help="Explicit experiment directory")
    args = parser.parse_args()

    if args.experiment:
        experiments = args.experiment
    else:
        experiments = discover_running(args.drop_root)

    results = [analyze_experiment(exp) for exp in experiments]
    results.sort(key=lambda r: str(r["run_name"]))

    print(f"{'run_name':<72} | {'chainsaw_metric':>15} | {'mass_drift':>12}")
    print("-" * 105)
    for r in results:
        name = str(r["run_name"])
        strip = r["chainsaw_metric"]
        strip_s = f"{strip:.4f}" if isinstance(strip, float) and np.isfinite(strip) else "NA"
        d_abs = r["mass_drift_abs"]
        d_rel = r["mass_drift_rel"]
        if isinstance(d_abs, float) and np.isfinite(d_abs):
            mass_s = f"{d_abs:+.3e} ({d_rel:+.3%})" if isinstance(d_rel, float) and np.isfinite(d_rel) else f"{d_abs:+.3e}"
        else:
            mass_s = "NA"
        print(f"{name:<72} | {strip_s:>15} | {mass_s:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
