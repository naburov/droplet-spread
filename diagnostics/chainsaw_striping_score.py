#!/usr/bin/env python3
"""Quantify x-direction interface rippling (chainsaw) from checkpoints or statistics."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np


def striping_score(phi: np.ndarray) -> float:
    """95th percentile of |d²φ/dx²| on the interface band (|φ| < 0.9)."""
    mask = np.abs(phi) < 0.9
    if not mask.any():
        return 0.0
    d2 = phi[2:, :] - 2.0 * phi[1:-1, :] + phi[:-2, :]
    m = mask[1:-1, :]
    values = np.abs(d2[m])
    return float(np.percentile(values, 95)) if values.size else 0.0


def scan_checkpoints(checkpoint_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    print(f"{'step':>8}  {'strip95':>10}  {'phi_min':>10}  {'phi_max':>10}")
    for path in paths:
        match = re.search(r"checkpoint_(\d+)", os.path.basename(path))
        if not match:
            continue
        step = int(match.group(1))
        data = np.load(path)
        phi = data["phi"]
        print(
            f"{step:8d}  {striping_score(phi):10.4f}  {phi.min():10.4f}  {phi.max():10.4f}"
        )


def scan_statistics(stats_path: str, columns: tuple[str, ...]) -> None:
    with open(stats_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty statistics file: {stats_path}")
    print(f"{'step':>8}  " + "  ".join(f"{c:>14}" for c in columns))
    for row in rows[:: max(1, len(rows) // 20)]:
        print(f"{int(row['step']):8d}  " + "  ".join(f"{float(row[c]):14.6g}" for c in columns))
    last = rows[-1]
    print("--- last row ---")
    for c in columns:
        print(f"  {c}: {last[c]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Directory containing checkpoint_*.npz files",
    )
    parser.add_argument(
        "--statistics",
        type=str,
        help="Path to statistics.csv (prints subsampled telemetry)",
    )
    args = parser.parse_args()
    if bool(args.checkpoints) == bool(args.statistics):
        parser.error("Provide exactly one of --checkpoints or --statistics")

    if args.checkpoints:
        scan_checkpoints(args.checkpoints)
    else:
        scan_statistics(
            args.statistics,
            (
                "phi_min",
                "phi_max",
                "surface_tension_max",
                "divergence_max",
                "droplet_mass",
                "dt",
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
