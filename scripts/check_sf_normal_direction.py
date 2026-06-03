#!/usr/bin/env python3
"""
Diagnose interface-normal direction and SF projection near the wall.

Outputs:
  - normal_direction_bottom_strip.csv
  - normal_direction_summary.json
"""

import argparse
import csv
import json
import os
import sys
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from numerics.finite_differences import jax_gradient, jax_norm
from physics.surface_tension import jax_curvature, jax_curvature_smooth
from simulation.two_phase import TwoPhaseSimulation


def _contact_pair(phi: np.ndarray) -> Optional[Tuple[int, int]]:
    mask = ((phi[:, 0] * phi[:, 1]) < 0.0) | (np.abs(phi[:, 0]) < 0.5) | (np.abs(phi[:, 1]) < 0.5)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return None
    return int(idx.min()), int(idx.max())


def _safe_unit(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = np.sqrt(vx * vx + vy * vy)
    inv = 1.0 / np.maximum(mag, eps)
    return vx * inv, vy * inv, mag


def main() -> None:
    p = argparse.ArgumentParser(description="Check SF vs interface normal direction near wall")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bottom_rows", type=int, default=4)
    p.add_argument("--phi_interface", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "normal_direction_bottom_strip.csv")
    json_path = os.path.join(args.output_dir, "normal_direction_summary.json")

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    sim = TwoPhaseSimulation(cfg, output_dir=args.output_dir)
    phi = np.array(sim.state.phi)
    sf = np.array(sim.state.compute_surface_tension())
    dx, dy = float(sim.state.dx), float(sim.state.dy)
    f1 = sim.state.geometry.f_1_grid

    use_comp = bool(getattr(sim.state.surface_tension_solver, "use_composition_field", False))
    smooth = bool(getattr(sim.state.surface_tension_solver, "smooth_curvature", True))
    radius = int(getattr(sim.state.surface_tension_solver, "smoothing_radius", 1))
    phase = 0.5 * (phi + 1.0) if use_comp else phi

    grad = np.array(jax_gradient(jnp.array(phase), dx, dy, f1))
    nx, ny, grad_mag = _safe_unit(grad[..., 0], grad[..., 1])
    sfx, sfy = sf[..., 0], sf[..., 1]
    sf_n = sfx * nx + sfy * ny
    sf_t = sfx * (-ny) + sfy * nx

    if smooth:
        kappa = np.array(jax_curvature_smooth(jnp.array(phase), dx, dy, f1, smoothing_radius=radius))
    else:
        kappa = np.array(jax_curvature(jnp.array(phase), dx, dy, f1))

    nx_raw, ny_raw, _ = _safe_unit(
        np.array(jax_gradient(jnp.array(phi), dx, dy, f1))[..., 0],
        np.array(jax_gradient(jnp.array(phi), dx, dy, f1))[..., 1],
    )
    sf_n_raw = sfx * nx_raw + sfy * ny_raw

    Ny = phi.shape[1]
    bottom_rows = max(1, min(int(args.bottom_rows), Ny))
    interface_mask = np.abs(phi) < float(args.phi_interface)
    strip_mask = np.zeros_like(interface_mask, dtype=bool)
    strip_mask[:, :bottom_rows] = True
    mask = interface_mask & strip_mask

    rows = []
    ii, jj = np.where(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "phi": float(phi[i, j]),
                "phase_used": float(phase[i, j]),
                "grad_mag": float(grad_mag[i, j]),
                "n_x": float(nx[i, j]),
                "n_y": float(ny[i, j]),
                "kappa": float(kappa[i, j]),
                "sf_x": float(sfx[i, j]),
                "sf_y": float(sfy[i, j]),
                "sf_mag": float(np.hypot(sfx[i, j], sfy[i, j])),
                "sf_dot_n": float(sf_n[i, j]),
                "sf_dot_t": float(sf_t[i, j]),
                "sf_dot_n_rawphi_normal": float(sf_n_raw[i, j]),
            }
        )

    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "phi_convention": "phi<0 liquid, phi>0 gas; n=grad(phase)/|grad(phase)| points liquid->gas",
        "phase_used_for_normal": "c=(phi+1)/2" if use_comp else "phi",
        "num_points_bottom_strip": int(len(rows)),
        "mean_sf_dot_n": float(np.mean([r["sf_dot_n"] for r in rows])) if rows else float("nan"),
        "mean_abs_sf_dot_n": float(np.mean(np.abs([r["sf_dot_n"] for r in rows]))) if rows else float("nan"),
        "mean_abs_sf_dot_t": float(np.mean(np.abs([r["sf_dot_t"] for r in rows]))) if rows else float("nan"),
        "frac_sf_along_plus_n": float(np.mean(np.array([r["sf_dot_n"] for r in rows]) > 0.0)) if rows else float("nan"),
        "frac_sf_along_minus_n": float(np.mean(np.array([r["sf_dot_n"] for r in rows]) < 0.0)) if rows else float("nan"),
    }

    contacts = _contact_pair(phi)
    if contacts is not None:
        left_i, right_i = contacts
        cp = {}
        for label, i in (("left_contact", left_i), ("right_contact", right_i)):
            j_star = int(np.argmin(np.abs(phi[i, :bottom_rows])))
            cp[label] = {
                "i": int(i),
                "j": int(j_star),
                "phi": float(phi[i, j_star]),
                "n_x": float(nx[i, j_star]),
                "n_y": float(ny[i, j_star]),
                "kappa": float(kappa[i, j_star]),
                "sf_x": float(sfx[i, j_star]),
                "sf_y": float(sfy[i, j_star]),
                "sf_dot_n": float(sf_n[i, j_star]),
                "sf_dot_t": float(sf_t[i, j_star]),
            }
        summary["contact_points"] = cp

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Bottom-strip points:", summary["num_points_bottom_strip"])
    print("mean(SF·n):", summary["mean_sf_dot_n"])
    print("frac(SF along +n):", summary["frac_sf_along_plus_n"])
    if "contact_points" in summary:
        for name, d in summary["contact_points"].items():
            print(
                f"{name}: i={d['i']} j={d['j']} n=({d['n_x']:.3e},{d['n_y']:.3e}) "
                f"SF=({d['sf_x']:.3e},{d['sf_y']:.3e}) SF·n={d['sf_dot_n']:.3e}"
            )


if __name__ == "__main__":
    main()

