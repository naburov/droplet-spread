#!/usr/bin/env python3
"""
Debug contact-angle phase BC application in isolation.

Given one phi field (from checkpoint), this script applies ContactAngleBoundaryCondition
for selected angles and quantifies:
  - how bottom row phi changes,
  - how well (phi[:,1]-phi[:,0])/dy matches -cos(theta)*|grad phi|,
  - apparent angle distribution near the contact-line region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition
from numerics.finite_differences import jax_gradient
from physics.properties import calculate_density
from simulation.geometry import Geometry


def _pick_checkpoint(experiment_dir: Path, step: int | None) -> Tuple[int, Path]:
    cdir = experiment_dir / "checkpoints"
    cands = sorted(cdir.glob("checkpoint_*.npz"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {cdir}")
    if step is None:
        p = cands[-1]
        s = int(p.stem.split("_")[1])
        return s, p
    p = cdir / f"checkpoint_{step:06d}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return int(step), p


def _contact_band_mask(phi: np.ndarray, j_rows: int = 3) -> np.ndarray:
    near = phi[:, : max(j_rows, 1)]
    return np.any(near < 0.0, axis=1)


def _evaluate(
    phi: np.ndarray,
    theta_deg: float,
    dx: float,
    dy: float,
    f1_grid: np.ndarray,
    method: str,
    alpha: float,
    flip_interface_normal: bool = False,
):
    theta_apply_deg = 180.0 - float(theta_deg) if flip_interface_normal else float(theta_deg)
    geom = Geometry.flat(phi.shape[0], phi.shape[1])
    bc = ContactAngleBoundaryCondition(
        contact_angle=theta_apply_deg,
        method=method,
        epsilon=0.02,
        use_ice_aware=False,
        use_geometry_aware=False,
        use_cox_voinov=False,
        contact_angle_relaxation=float(alpha),
    )
    phi_bc = np.array(
        bc.apply(jnp.array(phi), dx, dy, geometry=geom, psi=None, U=None, bottom_velocity_bc="no_slip"),
        dtype=np.float64,
    )
    grad = np.array(jax_gradient(jnp.array(phi_bc), dx, dy, jnp.array(f1_grid)), dtype=np.float64)
    gx = grad[:, 1, 0]
    gy = grad[:, 1, 1]
    gnorm = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-12)
    lhs = (phi_bc[:, 1] - phi_bc[:, 0]) / dy
    rhs = np.cos(np.deg2rad(theta_apply_deg)) * gnorm
    res = lhs - rhs
    theta_app = np.degrees(np.arccos(np.clip(lhs / gnorm, -1.0, 1.0)))
    return phi_bc, lhs, rhs, res, theta_app, theta_apply_deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--angles", nargs="*", type=float, default=[60.0, 90.0, 120.0])
    ap.add_argument("--first-rows", type=int, default=6)
    ap.add_argument("--flip-interface-normal", action="store_true")
    ap.add_argument("--out-dir", default="diagnostics/debug_contact_angle_bc_application")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    exp = Path(args.experiment_dir)
    step, cp = _pick_checkpoint(exp, args.step)
    z = np.load(cp)
    phi = np.array(z["phi"], dtype=np.float64)
    nx, ny = phi.shape
    dx = float(cfg["grid_params"]["Lx"]) / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])
    method = str(cfg.get("boundary_conditions", {}).get("phase_field", {}).get("contact_angle_method", "simple"))
    alpha = float(cfg.get("boundary_conditions", {}).get("phase_field", {}).get("contact_angle_relaxation", 1.0))
    f1 = np.zeros((nx, ny), dtype=np.float64)
    mask = _contact_band_mask(phi, j_rows=3)
    if not np.any(mask):
        mask = np.ones(nx, dtype=bool)

    out = Path(args.out_dir) / exp.name / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    name_suffix = "_flipped_normal" if bool(args.flip_interface_normal) else ""

    results: Dict[float, Dict[str, float]] = {}
    rho_by_theta: Dict[float, np.ndarray] = {}
    phi_by_theta: Dict[float, np.ndarray] = {}
    i = np.arange(nx)
    fig, ax = plt.subplots(3, len(args.angles), figsize=(5 * len(args.angles), 10), constrained_layout=True)
    if len(args.angles) == 1:
        ax = np.array(ax).reshape(3, 1)

    for col, theta in enumerate(args.angles):
        phi_bc, lhs, rhs, res, theta_app, theta_apply_deg = _evaluate(
            phi,
            theta,
            dx,
            dy,
            f1,
            method,
            alpha,
            flip_interface_normal=bool(args.flip_interface_normal),
        )

        results[float(theta)] = {
            "theta_target_deg": float(theta),
            "theta_applied_deg": float(theta_apply_deg),
            "theta_app_mean_deg_contact_band": float(np.mean(theta_app[mask])),
            "theta_app_std_deg_contact_band": float(np.std(theta_app[mask])),
            "bc_res_abs_mean_contact_band": float(np.mean(np.abs(res[mask]))),
            "bc_res_abs_max_contact_band": float(np.max(np.abs(res[mask]))),
            "phi_bottom_change_l2": float(np.sqrt(np.mean((phi_bc[:, 0] - phi[:, 0]) ** 2))),
        }
        rho_by_theta[float(theta)] = calculate_density(phi_bc, float(cfg["physical_params"]["rho1"]), float(cfg["physical_params"]["rho2"]))
        phi_by_theta[float(theta)] = phi_bc

        ax[0, col].plot(i, phi[:, 0], label="phi_bottom before", linewidth=1.2)
        ax[0, col].plot(i, phi_bc[:, 0], label="phi_bottom after", linewidth=1.2)
        ax[0, col].set_title(f"theta={theta:.0f} deg: bottom phi")
        ax[0, col].grid(alpha=0.25)
        ax[0, col].legend(fontsize=8)

        ax[1, col].plot(i, lhs, label="lhs", linewidth=1.2)
        ax[1, col].plot(i, rhs, label="rhs", linewidth=1.2)
        ax[1, col].plot(i, res, label="res", linewidth=1.2, alpha=0.8)
        ax[1, col].set_title("BC equation terms")
        ax[1, col].grid(alpha=0.25)
        ax[1, col].legend(fontsize=8)

        ax[2, col].plot(i[mask], theta_app[mask], ".", markersize=2)
        ax[2, col].axhline(theta, color="red", linestyle="--", linewidth=1.5, label="target")
        ax[2, col].set_ylim(0, 180)
        ax[2, col].set_title("Apparent theta in contact band")
        ax[2, col].grid(alpha=0.25)
        ax[2, col].legend(fontsize=8)

        # Save dedicated near-wall heatmaps for first rows: before / after / delta
        nrows = int(np.clip(args.first_rows, 1, ny))
        phi0 = phi[:, :nrows]
        phi1 = phi_bc[:, :nrows]
        dphi = phi1 - phi0
        vmax = float(max(np.max(np.abs(phi0)), np.max(np.abs(phi1)), 1e-12))
        dmax = float(max(np.max(np.abs(dphi)), 1e-12))

        fig_h, ax_h = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
        im0 = ax_h[0].imshow(phi0.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax_h[0].set_title("phi before (first rows)")
        ax_h[0].set_xlabel("i")
        ax_h[0].set_ylabel("j")
        plt.colorbar(im0, ax=ax_h[0], fraction=0.046, pad=0.04)

        im1 = ax_h[1].imshow(phi1.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax_h[1].set_title("phi after BC (first rows)")
        ax_h[1].set_xlabel("i")
        ax_h[1].set_ylabel("j")
        plt.colorbar(im1, ax=ax_h[1], fraction=0.046, pad=0.04)

        im2 = ax_h[2].imshow(dphi.T, origin="lower", aspect="auto", cmap="bwr", vmin=-dmax, vmax=dmax)
        ax_h[2].set_title("delta phi (after - before)")
        ax_h[2].set_xlabel("i")
        ax_h[2].set_ylabel("j")
        plt.colorbar(im2, ax=ax_h[2], fraction=0.046, pad=0.04)

        fig_h.suptitle(f"Contact-angle BC near-wall heatmap: theta={theta:.0f} deg, rows=0..{nrows-1}")
        fig_h.savefig(out / f"bc_first_rows_heatmap_theta_{int(round(theta)):03d}{name_suffix}.png", dpi=180)
        plt.close(fig_h)

        # Save phi=0 contour plots (before vs after BC)
        x = np.arange(nx)
        y = np.arange(ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        fig_c, ax_c = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        ax_c[0].contour(xx, yy, phi, levels=[0.0], colors=["tab:blue"], linewidths=1.8)
        ax_c[0].contour(xx, yy, phi_bc, levels=[0.0], colors=["tab:red"], linewidths=1.8, linestyles="--")
        ax_c[0].axhline(0.0, color="k", linewidth=1.0)
        ax_c[0].set_title("phi=0 contour (full domain)")
        ax_c[0].set_xlabel("i")
        ax_c[0].set_ylabel("j")
        ax_c[0].set_xlim(0, nx - 1)
        ax_c[0].set_ylim(0, ny - 1)
        ax_c[0].legend(["before", "after BC"], loc="upper right", fontsize=8)
        ax_c[0].grid(alpha=0.2)

        ax_c[1].contour(xx, yy, phi, levels=[0.0], colors=["tab:blue"], linewidths=1.8)
        ax_c[1].contour(xx, yy, phi_bc, levels=[0.0], colors=["tab:red"], linewidths=1.8, linestyles="--")
        ax_c[1].axhline(0.0, color="k", linewidth=1.0)
        ax_c[1].set_title(f"phi=0 contour (near wall, first {nrows} rows)")
        ax_c[1].set_xlabel("i")
        ax_c[1].set_ylabel("j")
        ax_c[1].set_xlim(0, nx - 1)
        ax_c[1].set_ylim(0, nrows - 1)
        ax_c[1].legend(["before", "after BC"], loc="upper right", fontsize=8)
        ax_c[1].grid(alpha=0.2)

        fig_c.suptitle(f"Contact-angle BC contour debug: theta={theta:.0f} deg")
        fig_c.savefig(out / f"bc_phi0_contour_theta_{int(round(theta)):03d}{name_suffix}.png", dpi=180)
        plt.close(fig_c)

    # Combined density heatmaps for all requested contact angles
    fig_rho, ax_rho = plt.subplots(1, len(args.angles), figsize=(5.2 * len(args.angles), 4.2), constrained_layout=True)
    if len(args.angles) == 1:
        ax_rho = np.array([ax_rho])
    all_rho = [rho_by_theta[float(t)] for t in args.angles]
    vmin = float(min(np.min(r) for r in all_rho))
    vmax = float(max(np.max(r) for r in all_rho))
    for k, theta in enumerate(args.angles):
        rho = rho_by_theta[float(theta)]
        phi_bc = phi_by_theta[float(theta)]
        im = ax_rho[k].imshow(rho.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_rho[k].contour(phi_bc.T, levels=[0.0], colors=["white"], linewidths=1.0, linestyles=":")
        theta_liq = results[float(theta)]["theta_app_mean_deg_contact_band"]
        theta_gas = 180.0 - theta_liq
        ax_rho[k].set_title(
            f"Density, target CA_liq={theta:.0f} deg\n"
            f"measured CA_liq={theta_liq:.1f} deg (CA_gas={theta_gas:.1f} deg)"
        )
        ax_rho[k].set_xlabel("i")
        ax_rho[k].set_ylabel("j")
    cbar = fig_rho.colorbar(im, ax=ax_rho.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("rho")
    fig_rho.suptitle(f"Step {step}: density heatmaps for CA sweep")
    fig_rho.savefig(out / f"density_heatmaps_by_ca{name_suffix}.png", dpi=180)
    plt.close(fig_rho)

    fig.suptitle(f"Contact-angle BC debug: {exp.name} step {step} (method={method}, alpha={alpha})")
    fig.savefig(out / f"bc_debug_panels{name_suffix}.png", dpi=180)
    plt.close(fig)

    payload = {
        "config": str(Path(args.config)),
        "experiment": exp.name,
        "step": step,
        "method": method,
        "contact_angle_relaxation": alpha,
        "flip_interface_normal": bool(args.flip_interface_normal),
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"saved: {out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

