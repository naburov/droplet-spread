#!/usr/bin/env python3
"""
Continue simulation from a checkpoint and monitor bottom-strip instability.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
from numerics.staggered_mac import divergence as mac_divergence


def _contact_span(phi: np.ndarray, pad: int = 2):
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor bottom divergence from restart")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stop_max_div", type=float, default=5000.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "monitor.csv")

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    sim = TwoPhaseSimulation(cfg, output_dir=args.output_dir)

    rows = []
    for n in range(int(args.steps)):
        sim.step()

        u = np.array(sim.state.u_face)
        v = np.array(sim.state.v_face)
        phi = np.array(sim.state.phi)
        div = np.array(mac_divergence(u, v, sim.state.dx, sim.state.dy))
        ad = np.abs(div)
        i_hot, j_hot = np.unravel_index(np.argmax(ad), ad.shape)

        span = _contact_span(phi, pad=2)
        contact_row1 = np.nan
        contact_band = np.nan
        if span is not None:
            lo, hi = span
            contact_row1 = float(np.max(ad[lo:hi + 1, 1]))
            contact_band = float(np.max(ad[lo:hi + 1, :4]))

        row = {
            "n": n,
            "step": int(sim.state.step),
            "time": float(sim.state.t),
            "dt": float(sim.state.dt),
            "max_div": float(np.max(ad)),
            "max_div_interior": float(np.max(ad[1:-1, 1:-1])),
            "mean_div": float(np.mean(ad)),
            "row1_max": float(np.max(ad[:, 1])),
            "row0_max": float(np.max(ad[:, 0])),
            "u_bottom_max": float(np.max(np.abs(u[:, 0]))),
            "u_j1_max": float(np.max(np.abs(u[:, 1]))),
            "v_j1_max": float(np.max(np.abs(v[:, 1]))),
            "hot_i": int(i_hot),
            "hot_j": int(j_hot),
            "contact_row1_max": contact_row1,
            "contact_band_max": contact_band,
        }
        rows.append(row)
        print(
            f"n={n:04d} step={row['step']} dt={row['dt']:.3e} "
            f"max_div={row['max_div']:.3e} row1={row['row1_max']:.3e} "
            f"u0={row['u_bottom_max']:.3e} hot=({row['hot_i']},{row['hot_j']})"
        )

        if row["max_div"] >= float(args.stop_max_div):
            print(f"Stopping early: max_div reached {row['max_div']:.3e}")
            break

        # When driving the simulation manually (outside BaseSimulation.run),
        # advance step counter explicitly so checkpoint/diagnostic cadence and
        # adaptive logic that depend on step index remain meaningful.
        sim.state.step += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    print("Saved monitor:", out_csv)


if __name__ == "__main__":
    main()

