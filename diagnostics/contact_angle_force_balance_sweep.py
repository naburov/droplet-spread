#!/usr/bin/env python3
"""
Sweep contact angle on one checkpoint and report near-contact force balance.

Goal: verify sign consistency with physical expectation:
  - theta < 90 deg  -> hydrophilic tendency
  - theta > 90 deg  -> hydrophobic tendency
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from numerics.finite_differences import jax_gradient
from physics.properties import jax_calculate_density
from physics.surface_tension import (
    jax_apply_surface_tension_boundary_conditions,
    jax_surface_tension_force,
)
from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition
from simulation.geometry import Geometry


def _f1_grid_from_config(cfg: Dict, nx: int, ny: int, dx: float, lx: float) -> np.ndarray:
    geom_cfg = cfg.get("initial_conditions", {}).get("geometry", {}) or cfg.get("geometry", {})
    gtype = str(geom_cfg.get("type", "flat")).lower()
    if gtype == "tilted":
        deg = float(geom_cfg.get("degree", 10.0))
        slope = np.tan(np.deg2rad(deg))
        return np.full((nx, ny), slope, dtype=np.float64)
    if gtype == "hump":
        amp = float(geom_cfg.get("amplitude", 0.1))
        sigma = float(geom_cfg.get("sigma", 0.2))
        center = float(geom_cfg.get("center_x", lx / 2.0))
        x = np.arange(nx, dtype=np.float64) * dx
        f = amp * np.exp(-((x - center) ** 2) / (2.0 * sigma * sigma))
        f1 = -f * (x - center) / (sigma * sigma)
        return np.repeat(f1[:, None], ny, axis=1)
    return np.zeros((nx, ny), dtype=np.float64)


def _pick_checkpoint(exp_dir: Path, step: int | None) -> Tuple[int, Path]:
    cdir = exp_dir / "checkpoints"
    cands = sorted(cdir.glob("checkpoint_*.npz"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {cdir}")
    if step is None:
        p = cands[-1]
        s = int(p.stem.split("_")[1])
        return s, p
    p = cdir / f"checkpoint_{step:06d}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint missing: {p}")
    return int(step), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config JSON path")
    ap.add_argument("--experiment-dir", required=True, help="Experiment with checkpoints")
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step (default latest)")
    ap.add_argument("--angles", nargs="*", type=float, default=[60.0, 90.0, 120.0])
    ap.add_argument("--contact-window", type=int, default=3, help="Half-window around each contact index")
    ap.add_argument("--out-dir", default="diagnostics/contact_angle_force_balance_sweep")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    exp = Path(args.experiment_dir)
    step, cp = _pick_checkpoint(exp, args.step)
    z = np.load(cp)
    phi = np.array(z["phi"], dtype=np.float64)
    P = np.array(z["P"], dtype=np.float64)
    nx, ny = phi.shape

    gp = cfg["grid_params"]
    pp = cfg["physical_params"]
    st_cfg = pp.get("surface_tension", {})
    dx = float(gp["Lx"]) / float(gp["Nx"])
    dy = float(gp["Ly"]) / float(gp["Ny"])
    f1 = _f1_grid_from_config(cfg, nx, ny, dx, float(gp["Lx"]))

    # Contact footprint from near-wall liquid mask.
    near = phi[:, :3]
    liquid = np.any(near < 0.0, axis=1)
    xs = np.where(liquid)[0]
    if xs.size == 0:
        raise RuntimeError("No near-wall droplet footprint found.")
    left = int(xs.min())
    right = int(xs.max())
    w = max(int(args.contact_window), 1)
    left_band = slice(max(0, left - w), min(nx, left + w + 1))
    right_band = slice(max(0, right - w), min(nx, right + w + 1))
    mask = np.zeros(nx, dtype=bool)
    mask[left_band] = True
    mask[right_band] = True

    # Pressure gradient and density are angle-independent.
    grad_p = np.array(jax_gradient(jnp.array(P), dx, dy, jnp.array(f1)), dtype=np.float64)
    rho = np.array(
        jax_calculate_density(jnp.array(phi), float(pp["rho1"]), float(pp["rho2"])),
        dtype=np.float64,
    )
    rho = np.maximum(rho, 1e-12)
    inv_rho = 1.0 / rho
    a_pg_y = -grad_p[:, :, 1] * inv_rho
    g = float(pp.get("g", -1.0))
    fr = float(pp.get("Fr", 1.0))
    g_term = g / max(fr, 1e-12)

    sf_raw = np.array(
        jax_surface_tension_force(
            jnp.array(phi),
            float(pp["epsilon"]),
            float(pp["We1"]),
            float(pp["We2"]),
            dx,
            dy,
            jnp.array(f1),
            smooth_curvature=bool(st_cfg.get("smooth_curvature", True)),
            smoothing_radius=int(st_cfg.get("smoothing_radius", 1)),
            use_composition_field=bool(st_cfg.get("use_composition_field", True)),
            composition_force_scale=float(st_cfg.get("composition_force_scale", 1.0)),
            weber_interpolation=str(st_cfg.get("weber_interpolation", "constant_liquid")),
        ),
        dtype=np.float64,
    )

    out = Path(args.out_dir) / exp.name / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    line_data = {}
    # Build one reusable flat-geometry object for phase-BC application.
    geom = Geometry.flat(nx, ny)
    for theta in args.angles:
        sf = np.array(
            jax_apply_surface_tension_boundary_conditions(
                jnp.array(sf_raw),
                jnp.array(phi),
                contact_angle=float(theta),
                f_1_surface=jnp.array(f1[:, 0]) if np.any(np.abs(f1[:, 0]) > 1e-12) else None,
            ),
            dtype=np.float64,
        )
        a_sf_y = -sf[:, :, 1] * inv_rho
        a_res_y = a_sf_y + a_pg_y + g_term
        fy_bottom = sf[:, 0, 1]

        # Phase-BC side: apply contact-angle BC and measure apparent angle.
        ca_bc = ContactAngleBoundaryCondition(
            contact_angle=float(theta),
            method=str(cfg.get("boundary_conditions", {}).get("phase_field", {}).get("contact_angle_method", "simple")),
            epsilon=float(pp.get("epsilon", 0.02)),
            use_ice_aware=False,
            use_geometry_aware=False,
            use_cox_voinov=False,
            contact_angle_relaxation=float(
                cfg.get("boundary_conditions", {}).get("phase_field", {}).get("contact_angle_relaxation", 1.0)
            ),
        )
        phi_bc = np.array(
            ca_bc.apply(jnp.array(phi), dx, dy, geometry=geom, psi=None, U=None, bottom_velocity_bc="no_slip"),
            dtype=np.float64,
        )
        grad_bc = np.array(jax_gradient(jnp.array(phi_bc), dx, dy, jnp.array(f1)), dtype=np.float64)
        grad_norm_bc = np.maximum(np.sqrt(grad_bc[:, 1, 0] ** 2 + grad_bc[:, 1, 1] ** 2), 1e-12)
        lhs_bc = (phi_bc[:, 1] - phi_bc[:, 0]) / dy
        cos_app = np.clip(lhs_bc / grad_norm_bc, -1.0, 1.0)
        theta_app_deg = np.degrees(np.arccos(cos_app))
        theta_app_mean = float(np.mean(theta_app_deg[mask]))

        row = {
            "theta_deg": float(theta),
            "fy_left_mean": float(np.mean(fy_bottom[left_band])),
            "fy_right_mean": float(np.mean(fy_bottom[right_band])),
            "a_sf_y_cl_mean": float(np.mean(a_sf_y[mask, 0])),
            "a_pg_y_cl_mean": float(np.mean(a_pg_y[mask, 0])),
            "g_term": float(g_term),
            "a_res_y_cl_mean": float(np.mean(a_res_y[mask, 0])),
            "theta_app_phasebc_mean_deg": theta_app_mean,
        }
        rows.append(row)
        line_data[theta] = fy_bottom

    csv_path = out / "force_balance_sweep.csv"
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Plot signed bottom Fy for each angle.
    i = np.arange(nx)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    for theta in args.angles:
        ax.plot(i, line_data[theta], linewidth=1.8, label=f"{theta:.0f} deg")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.6)
    ax.axvline(left, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(right, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Signed bottom F_sigma_y near contact line")
    ax.set_xlabel("i")
    ax.set_ylabel("F_sigma_y at j=0")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(out / "bottom_signed_fy_sweep.png", dpi=180)
    plt.close(fig)

    summary = {
        "config": str(Path(args.config)),
        "experiment": exp.name,
        "checkpoint_step": step,
        "contact_left_i": left,
        "contact_right_i": right,
        "contact_window": w,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

