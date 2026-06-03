#!/usr/bin/env python3
"""
Per-cell capillary-contributor heatmaps from a checkpoint.

Builds maps for the main ingredients of capillary forcing:
  1) curvature magnitude |kappa|
  2) normalization amplifier 1/max(|grad(phase)|, eps_norm)
  3) interface localization |grad(phase)|^2
  4) Weber scaling 1/We

Also writes a compact 3-panel figure with:
  - |kappa|
  - 1/max(|grad|, eps_norm)
  - |grad|^2 / We
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Allow imports from src when run from repo root.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from numerics.finite_differences import jax_gradient, jax_norm
from physics.surface_tension import jax_curvature, jax_curvature_smooth


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


def _we_map(phi: np.ndarray, we1: float, we2: float, mode: str) -> np.ndarray:
    c = 0.5 * (phi + 1.0)
    if mode == "harmonic":
        return 1.0 / (((1.0 - c) / we2) + (c / we1))
    if mode == "arithmetic":
        return (1.0 - c) * we2 + c * we1
    if mode == "constant_liquid":
        return np.full_like(phi, float(we2))
    raise ValueError(f"Unknown weber mode: {mode}")


def _plot_map(ax, field: np.ndarray, title: str, cmap: str = "magma", clip_q: float = 99.0) -> None:
    finite = np.isfinite(field)
    vmax = np.percentile(field[finite], clip_q) if np.any(finite) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _smooth_3x3_edge(field: np.ndarray) -> np.ndarray:
    p = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def _smooth_field(field: np.ndarray, radius: int) -> np.ndarray:
    out = np.array(field, copy=True)
    for _ in range(max(int(radius), 0)):
        out = _smooth_3x3_edge(out)
    return out


def _compute_contributors(cfg: Dict, phi: np.ndarray, eps_norm: float) -> Dict[str, np.ndarray]:
    nx, ny = phi.shape
    lx = float(cfg["grid_params"]["Lx"])
    ly = float(cfg["grid_params"]["Ly"])
    dx = lx / float(cfg["grid_params"]["Nx"])
    dy = ly / float(cfg["grid_params"]["Ny"])
    f1 = _f1_grid_from_config(cfg, nx, ny, dx, lx)

    st = cfg.get("physical_params", {}).get("surface_tension", {})
    use_composition = bool(st.get("use_composition_field", True))
    smooth_curvature = bool(st.get("smooth_curvature", True))
    smoothing_radius = int(st.get("smoothing_radius", 1))
    weber_mode = str(st.get("weber_interpolation", "constant_liquid"))
    epsilon = float(cfg["physical_params"]["epsilon"])
    we1 = float(cfg["physical_params"]["We1"])
    we2 = float(cfg["physical_params"]["We2"])
    c_scale = float(st.get("composition_force_scale", 1.0))

    phase = 0.5 * (phi + 1.0) if use_composition else phi
    phase_j = jnp.array(phase)
    f1_j = jnp.array(f1)

    if smooth_curvature:
        kappa = np.array(jax_curvature_smooth(phase_j, dx, dy, f1_j, smoothing_radius=smoothing_radius), dtype=np.float64)
    else:
        kappa = np.array(jax_curvature(phase_j, dx, dy, f1_j), dtype=np.float64)

    grad = np.array(jax_gradient(phase_j, dx, dy, f1_j), dtype=np.float64)
    grad_mag = np.array(jax_norm(grad), dtype=np.float64)
    inv_grad_mag = 1.0 / np.maximum(grad_mag, float(eps_norm))

    # | |grad| * grad | = |grad|^2
    grad_sq = grad_mag * grad_mag

    we = _we_map(phi, we1, we2, weber_mode)
    inv_we = 1.0 / np.maximum(we, 1e-12)

    coeff = c_scale * (3.0 * np.sqrt(2.0) * epsilon / 4.0)
    capillary_weight = np.abs(kappa) * grad_sq * inv_we
    sf_mag_model = coeff * capillary_weight
    grad_sq_over_we = grad_sq * inv_we

    return {
        "kappa_abs": np.abs(kappa),
        "grad_mag": grad_mag,
        "inv_grad_mag": inv_grad_mag,
        "grad_sq": grad_sq,
        "inv_we": inv_we,
        "grad_sq_over_we": grad_sq_over_we,
        "capillary_weight": capillary_weight,
        "sf_mag_model": sf_mag_model,
        "phase": phase,
        "phi": phi,
    }


def _plot_bundle(c: Dict[str, np.ndarray], ck_name: str, run_dir: Path, suffix: str = "") -> None:
    suffix = str(suffix)
    tag = f"_{suffix}" if suffix else ""
    pretty = f" ({suffix})" if suffix else ""

    fig3, ax3 = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    _plot_map(ax3[0], c["kappa_abs"], "|kappa|")
    _plot_map(ax3[1], c["inv_grad_mag"], "1 / max(|grad(phase)|, eps_norm)")
    _plot_map(ax3[2], c["grad_sq_over_we"], "|grad(phase)|^2 / We")
    fig3.suptitle(f"Capillary Contributors (3-panel){pretty} - {ck_name}")
    fig3.savefig(run_dir / f"contributors_3panel{tag}.png", dpi=180)
    plt.close(fig3)

    fig5, ax5 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    _plot_map(ax5[0, 0], c["kappa_abs"], "|kappa|")
    _plot_map(ax5[0, 1], c["inv_grad_mag"], "1 / max(|grad(phase)|, eps_norm)")
    _plot_map(ax5[0, 2], c["grad_sq"], "|grad(phase)|^2")
    _plot_map(ax5[1, 0], c["inv_we"], "1 / We")
    _plot_map(ax5[1, 1], c["capillary_weight"], "|kappa| * |grad|^2 / We")
    _plot_map(ax5[1, 2], c["sf_mag_model"], "modeled |F_sigma| prefactor*weight")
    fig5.suptitle(f"Capillary Contributors (full){pretty} - {ck_name}")
    fig5.savefig(run_dir / f"contributors_full{tag}.png", dpi=180)
    plt.close(fig5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to simulation_parameters.json")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_XXXXXX.npz")
    ap.add_argument("--out-dir", default="diagnostics/capillary_contributors", help="Output directory")
    ap.add_argument("--eps-norm", type=float, default=1e-6, help="Floor for |grad(phase)| in normalization map")
    ap.add_argument(
        "--plot-smoothing-radius",
        type=int,
        default=1,
        help="Extra 3x3 smoothing passes for a second (smoothed) plot set.",
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    d = np.load(args.checkpoint)
    phi = np.array(d["phi"], dtype=np.float64)

    out_root = Path(args.out_dir)
    ck_name = Path(args.checkpoint).stem
    run_dir = out_root / ck_name
    run_dir.mkdir(parents=True, exist_ok=True)

    c = _compute_contributors(cfg, phi, args.eps_norm)

    # Save raw per-cell fields.
    np.savez_compressed(
        run_dir / "contributors.npz",
        **c,
    )

    # Raw plots.
    _plot_bundle(c, ck_name, run_dir, suffix="")

    # Smoothed plots (for visual diagnosis only).
    smooth_r = max(int(args.plot_smoothing_radius), 0)
    if smooth_r > 0:
        cs = {}
        for k, v in c.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and k not in ("phi", "phase"):
                cs[k] = _smooth_field(v, smooth_r)
            else:
                cs[k] = v
        np.savez_compressed(
            run_dir / f"contributors_smoothed_r{smooth_r}.npz",
            **cs,
        )
        _plot_bundle(cs, ck_name, run_dir, suffix=f"smoothed_r{smooth_r}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "eps_norm": float(args.eps_norm),
        "plot_smoothing_radius": int(args.plot_smoothing_radius),
        "weber_mode": str(cfg.get("physical_params", {}).get("surface_tension", {}).get("weber_interpolation", "constant_liquid")),
        "kappa_abs_p99": float(np.percentile(c["kappa_abs"], 99.0)),
        "inv_grad_mag_p99": float(np.percentile(c["inv_grad_mag"], 99.0)),
        "grad_sq_over_we_p99": float(np.percentile(c["grad_sq_over_we"], 99.0)),
        "capillary_weight_p99": float(np.percentile(c["capillary_weight"], 99.0)),
        "sf_mag_model_p99": float(np.percentile(c["sf_mag_model"], 99.0)),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved: {run_dir / 'contributors_3panel.png'}")
    print(f"saved: {run_dir / 'contributors_full.png'}")
    if max(int(args.plot_smoothing_radius), 0) > 0:
        sr = max(int(args.plot_smoothing_radius), 0)
        print(f"saved: {run_dir / f'contributors_3panel_smoothed_r{sr}.png'}")
        print(f"saved: {run_dir / f'contributors_full_smoothed_r{sr}.png'}")
        print(f"saved: {run_dir / f'contributors_smoothed_r{sr}.npz'}")
    print(f"saved: {run_dir / 'contributors.npz'}")
    print(f"saved: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

