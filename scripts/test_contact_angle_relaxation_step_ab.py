#!/usr/bin/env python3
"""
Fast A/B test for contact-angle relaxation using a single physics step.
"""

import argparse
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
        return (0, phi.shape[0] - 1)
    return (max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad))


def _run_one(cfg: dict, out_dir: str):
    sim = TwoPhaseSimulation(cfg, output_dir=out_dir)
    sim.step()
    u = np.array(sim.state.u_face)
    v = np.array(sim.state.v_face)
    phi = np.array(sim.state.phi)
    div = np.array(mac_divergence(u, v, sim.state.dx, sim.state.dy))
    ad = np.abs(div)
    i_lo, i_hi = _contact_span(phi, pad=2)
    return {
        "max_div": float(np.max(ad)),
        "mean_div": float(np.mean(ad)),
        "max_div_interior": float(np.max(ad[1:-1, 1:-1])),
        "row1_max": float(np.max(ad[:, 1])),
        "contact_row1_max": float(np.max(ad[i_lo:i_hi + 1, 1])),
        "contact_band_max": float(np.max(ad[i_lo:i_hi + 1, :4])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="One-step A/B for contact-angle relaxation")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha_off", type=float, default=0.5)
    p.add_argument("--alpha_on", type=float, default=0.2)
    args = p.parse_args()

    with open(args.config, "r") as f:
        base = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for mode, alpha in (("off", args.alpha_off), ("on", args.alpha_on)):
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("restart", {})["restart_from"] = args.checkpoint
        cfg.setdefault("boundary_conditions", {}).setdefault("phase_field", {})[
            "contact_angle_relaxation"
        ] = float(alpha)
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        results[mode] = _run_one(cfg, mode_dir)

    print("\n=== Contact-angle relaxation one-step A/B ===")
    print(f"alpha_off={args.alpha_off} alpha_on={args.alpha_on}")
    for key in ("max_div", "max_div_interior", "mean_div", "row1_max", "contact_row1_max", "contact_band_max"):
        print(f"{key:18s} off={results['off'][key]:12.6g}  on={results['on'][key]:12.6g}")


if __name__ == "__main__":
    main()

