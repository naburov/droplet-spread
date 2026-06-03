#!/usr/bin/env python3
"""
Check consistency between contact-angle phase BC and bottom surface-tension BC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from numerics.finite_differences import jax_gradient
from physics.surface_tension import (
    jax_apply_surface_tension_boundary_conditions,
    jax_surface_tension_force,
)


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
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step; default latest")
    ap.add_argument("--out-dir", default="diagnostics/contact_angle_bc_force_consistency")
    args = ap.parse_args()

    exp = Path(args.experiment_dir)
    cfg = json.loads((exp / "simulation_parameters.json").read_text())
    step, cp = _pick_checkpoint(exp, args.step)
    z = np.load(cp)
    phi = np.array(z["phi"], dtype=np.float64)
    nx, ny = phi.shape

    lx = float(cfg["grid_params"]["Lx"])
    dx = lx / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])
    pp = cfg["physical_params"]
    st_cfg = pp.get("surface_tension", {})
    theta_deg = float(pp.get("contact_angle", 90.0))
    theta = np.deg2rad(theta_deg)

    f1 = _f1_grid_from_config(cfg, nx, ny, dx, lx)
    grad = np.array(jax_gradient(jnp.array(phi), dx, dy, jnp.array(f1)), dtype=np.float64)
    grad_x_j1 = grad[:, 1, 0]
    grad_y_j1 = grad[:, 1, 1]
    grad_norm_j1 = np.maximum(np.sqrt(grad_x_j1 * grad_x_j1 + grad_y_j1 * grad_y_j1), 1e-12)

    # Phase BC consistency in liquid-side convention:
    # (phi1-phi0)/dy ?= cos(theta_liq) |grad(phi)| at j=1
    lhs = (phi[:, 1] - phi[:, 0]) / dy
    rhs = np.cos(theta) * grad_norm_j1
    phase_res = lhs - rhs

    # Interface-near-wall mask for meaningful angle checks.
    mask_if = np.abs(phi[:, 1]) < 0.95
    if not np.any(mask_if):
        mask_if = np.ones(nx, dtype=bool)

    cos_app = np.clip(lhs / grad_norm_j1, -1.0, 1.0)
    theta_app_deg = np.degrees(np.arccos(cos_app))

    # Surface-tension consistency
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
    apply_over = bool(st_cfg.get("apply_boundary_overwrite", True))
    if apply_over:
        sf_bc = np.array(
            jax_apply_surface_tension_boundary_conditions(
                jnp.array(sf_raw),
                jnp.array(phi),
                contact_angle=theta_deg,
                f_1_surface=jnp.array(f1[:, 0]) if np.any(np.abs(f1[:, 0]) > 1e-12) else None,
            ),
            dtype=np.float64,
        )
    else:
        sf_bc = sf_raw.copy()

    # Reconstruct expected bottom force according to implementation branch.
    if np.any(np.abs(f1[:, 0]) > 1e-12):
        nf = np.sqrt(1.0 + f1[:, 0] ** 2)
        nx_s = -f1[:, 0] / nf
        ny_s = 1.0 / nf
        sf_x0 = sf_raw[:, 0, 0]
        sf_y0 = sf_raw[:, 0, 1]
        sf_n = sf_x0 * nx_s + sf_y0 * ny_s
        sf_t = sf_x0 * (-ny_s) + sf_y0 * nx_s
        sf_n_adj = sf_n * np.cos(theta)
        sf_x_expected = sf_n_adj * nx_s - sf_t * ny_s
        sf_y_expected = sf_n_adj * ny_s + sf_t * nx_s
    else:
        sf_x_expected = sf_raw[:, 1, 0]
        sf_y_expected = sf_raw[:, 1, 1] * np.cos(theta)

    fx_res = sf_bc[:, 0, 0] - sf_x_expected
    fy_res = sf_bc[:, 0, 1] - sf_y_expected

    summary = {
        "experiment": exp.name,
        "step": step,
        "theta_target_deg": theta_deg,
        "phase_bc_res_abs_mean_all": float(np.mean(np.abs(phase_res))),
        "phase_bc_res_abs_mean_interface": float(np.mean(np.abs(phase_res[mask_if]))),
        "phase_bc_res_abs_max_interface": float(np.max(np.abs(phase_res[mask_if]))),
        "theta_app_deg_mean_interface": float(np.mean(theta_app_deg[mask_if])),
        "theta_app_deg_std_interface": float(np.std(theta_app_deg[mask_if])),
        "theta_app_deg_p10_interface": float(np.percentile(theta_app_deg[mask_if], 10)),
        "theta_app_deg_p90_interface": float(np.percentile(theta_app_deg[mask_if], 90)),
        "sf_bc_overwrite_enabled": apply_over,
        "sf_bottom_fx_res_abs_mean": float(np.mean(np.abs(fx_res))),
        "sf_bottom_fy_res_abs_mean": float(np.mean(np.abs(fy_res))),
        "sf_bottom_fx_res_abs_max": float(np.max(np.abs(fx_res))),
        "sf_bottom_fy_res_abs_max": float(np.max(np.abs(fy_res))),
    }

    out = Path(args.out_dir) / exp.name / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)

    i = np.arange(nx)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax[0, 0].plot(i, lhs, label="lhs=(phi1-phi0)/dy", linewidth=1.5)
    ax[0, 0].plot(i, rhs, label="rhs=-cos(theta)|grad|", linewidth=1.5)
    ax[0, 0].set_title("Phase contact-angle BC relation")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(i, phase_res, color="crimson", linewidth=1.4)
    ax[0, 1].axhline(0.0, color="k", linestyle="--", alpha=0.6)
    ax[0, 1].set_title("Phase BC residual (lhs-rhs)")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(i, sf_bc[:, 0, 1], label="sf_bc_y bottom", linewidth=1.4)
    ax[1, 0].plot(i, sf_y_expected, label="sf_y expected", linewidth=1.4)
    ax[1, 0].set_title("Bottom normal force consistency")
    ax[1, 0].legend()
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].hist(theta_app_deg[mask_if], bins=30, alpha=0.8, color="steelblue")
    ax[1, 1].axvline(theta_deg, color="red", linestyle="--", linewidth=2, label=f"target {theta_deg:.1f} deg")
    ax[1, 1].set_title("Apparent theta distribution (near-wall interface)")
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Contact-angle BC / bottom-force consistency: {exp.name} step {step}")
    fig.savefig(out / "consistency_plots.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        out / "consistency_fields.npz",
        phi=phi,
        lhs=lhs,
        rhs=rhs,
        phase_res=phase_res,
        theta_app_deg=theta_app_deg,
        mask_if=mask_if,
        sf_raw=sf_raw,
        sf_bc=sf_bc,
        sf_x_expected=sf_x_expected,
        sf_y_expected=sf_y_expected,
        fx_res=fx_res,
        fy_res=fy_res,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

