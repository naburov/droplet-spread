#!/usr/bin/env python3
"""
Term-by-term heatmaps in interface area for force-balance/PPE equations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from numerics.finite_differences import jax_divergence, jax_gradient, jax_norm
from physics.pressure import compute_hydrostatic_pressure
from physics.properties import jax_calculate_density
from physics.surface_tension import (
    jax_apply_surface_tension_boundary_conditions,
    jax_curvature,
    jax_curvature_smooth,
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


def _smooth_3x3_edge(field: np.ndarray, radius: int) -> np.ndarray:
    out = np.array(field, copy=True)
    for _ in range(max(int(radius), 0)):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return out


def _bbox_from_mask(mask: np.ndarray, pad: int = 5) -> Tuple[slice, slice]:
    ii, jj = np.where(mask)
    nx, ny = mask.shape
    if ii.size == 0:
        return slice(0, nx), slice(0, min(ny, max(8, ny // 3)))
    i0 = max(0, int(ii.min()) - pad)
    i1 = min(nx, int(ii.max()) + pad + 1)
    j0 = max(0, int(jj.min()) - pad)
    j1 = min(ny, int(jj.max()) + pad + 1)
    return slice(i0, i1), slice(j0, j1)


def _mask_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(field, copy=True)
    out[~mask] = np.nan
    return out


def _plot(ax, f: np.ndarray, title: str, cmap: str = "coolwarm", q: float = 99.0, signed: bool = True) -> None:
    finite = np.isfinite(f)
    if not np.any(finite):
        v = 1.0
    else:
        v = float(np.percentile(np.abs(f[finite]), q)) if signed else float(np.percentile(f[finite], q))
        if not np.isfinite(v) or v <= 0.0:
            v = 1.0
    if signed:
        im = ax.imshow(f.T, origin="lower", cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    else:
        im = ax.imshow(f.T, origin="lower", cmap="magma", vmin=0.0, vmax=v, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _grad_faces(p: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = p.shape
    dpdx = np.zeros((nx + 1, ny), dtype=np.float64)
    dpdy = np.zeros((nx, ny + 1), dtype=np.float64)
    dpdx[1:nx, :] = (p[1:, :] - p[:-1, :]) / max(dx, 1e-12)
    dpdy[:, 1:ny] = (p[:, 1:] - p[:, :-1]) / max(dy, 1e-12)
    dpdx[0, :] = dpdx[1, :]
    dpdx[nx, :] = dpdx[nx - 1, :]
    dpdy[:, 0] = dpdy[:, 1]
    dpdy[:, ny] = dpdy[:, ny - 1]
    return dpdx, dpdy


def _inv_rho_faces(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = rho.shape
    inv = 1.0 / np.maximum(rho, 1e-12)
    iu = np.zeros((nx + 1, ny), dtype=np.float64)
    iv = np.zeros((nx, ny + 1), dtype=np.float64)
    iu[1:nx, :] = 0.5 * (inv[1:, :] + inv[:-1, :])
    iu[0, :] = inv[0, :]
    iu[nx, :] = inv[-1, :]
    iv[:, 1:ny] = 0.5 * (inv[:, 1:] + inv[:, :-1])
    iv[:, 0] = inv[:, 0]
    iv[:, ny] = inv[:, -1]
    return iu, iv


def _lhs_varcoef_ppe(p_dyn: np.ndarray, rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    iu, iv = _inv_rho_faces(rho)
    dpdx, dpdy = _grad_faces(p_dyn, dx, dy)
    fu = iu * dpdx
    fv = iv * dpdy
    return (fu[1:, :] - fu[:-1, :]) / max(dx, 1e-12) + (fv[:, 1:] - fv[:, :-1]) / max(dy, 1e-12)


def _term_fields(cfg: Dict, phi: np.ndarray, p_total: np.ndarray) -> Dict[str, np.ndarray]:
    nx, ny = phi.shape
    lx = float(cfg["grid_params"]["Lx"])
    ly = float(cfg["grid_params"]["Ly"])
    dx = lx / float(cfg["grid_params"]["Nx"])
    dy = ly / float(cfg["grid_params"]["Ny"])
    f1 = _f1_grid_from_config(cfg, nx, ny, dx, lx)

    pp = cfg["physical_params"]
    st = pp.get("surface_tension", {})
    rho1 = float(pp["rho1"])
    rho2 = float(pp["rho2"])
    g = float(pp["g"])
    fr = float(pp["Fr"])
    atm = float(pp["atm_pressure"])
    eps = float(pp["epsilon"])
    we1 = float(pp["We1"])
    we2 = float(pp["We2"])
    contact_angle = float(pp.get("contact_angle", 90.0))
    smooth_curv = bool(st.get("smooth_curvature", True))
    srad = int(st.get("smoothing_radius", 1))
    use_comp = bool(st.get("use_composition_field", True))
    cscale = float(st.get("composition_force_scale", 1.0))
    we_mode = str(st.get("weber_interpolation", "constant_liquid"))
    apply_bc_overwrite = bool(st.get("apply_boundary_overwrite", True))
    rhs_smooth = int(st.get("capillary_rhs_smoothing_radius", 1))

    phi_j = jnp.array(phi)
    f1_j = jnp.array(f1)

    phase = 0.5 * (phi + 1.0) if use_comp else phi
    if smooth_curv:
        kappa = np.array(jax_curvature_smooth(jnp.array(phase), dx, dy, f1_j, smoothing_radius=srad), dtype=np.float64)
    else:
        kappa = np.array(jax_curvature(jnp.array(phase), dx, dy, f1_j), dtype=np.float64)
    grad_phase = np.array(jax_gradient(jnp.array(phase), dx, dy, f1_j), dtype=np.float64)
    grad_phase_mag = np.array(jax_norm(jnp.array(grad_phase)), dtype=np.float64)

    sf = np.array(
        jax_surface_tension_force(
            phi_j, eps, we1, we2, dx, dy, f1_j,
            smooth_curvature=smooth_curv,
            smoothing_radius=srad,
            use_composition_field=use_comp,
            composition_force_scale=cscale,
            weber_interpolation=we_mode,
        ),
        dtype=np.float64,
    )
    if apply_bc_overwrite:
        sf = np.array(
            jax_apply_surface_tension_boundary_conditions(
                jnp.array(sf), phi_j, contact_angle=contact_angle, f_1_surface=f1_j[:, 0]
            ),
            dtype=np.float64,
        )

    rho = np.array(jax_calculate_density(phi_j, rho1, rho2), dtype=np.float64)
    inv_rho = 1.0 / np.maximum(rho, 1e-12)

    p_h = np.array(compute_hydrostatic_pressure(jnp.array(rho), g, dy, fr, atm), dtype=np.float64)
    p_dyn = np.array(p_total, dtype=np.float64) - p_h
    grad_p_dyn = np.array(jax_gradient(jnp.array(p_dyn), dx, dy, f1_j), dtype=np.float64)
    grad_p_h = np.array(jax_gradient(jnp.array(p_h), dx, dy, f1_j), dtype=np.float64)

    a_sf = -sf * inv_rho[..., None]
    a_pg_dyn = -grad_p_dyn * inv_rho[..., None]
    a_pg_h = -grad_p_h * inv_rho[..., None]
    g_vec = np.zeros_like(a_sf)
    g_vec[..., 1] = g / max(fr, 1e-12)

    a_res_current = a_sf + a_pg_dyn + a_pg_h + g_vec
    a_res_cap = (-a_sf) + a_pg_dyn + a_pg_h + g_vec

    rhs_raw = np.array(jax_divergence(jnp.array(sf), dx, dy, f1_j), dtype=np.float64)
    rhs_sm = _smooth_3x3_edge(rhs_raw, rhs_smooth)
    lhs = _lhs_varcoef_ppe(p_dyn, rho, dx, dy)

    return {
        "phi": phi,
        "phase": phase,
        "kappa": kappa,
        "grad_phase_mag": grad_phase_mag,
        "sf_x": sf[..., 0],
        "sf_y": sf[..., 1],
        "a_sf_x": a_sf[..., 0],
        "a_sf_y": a_sf[..., 1],
        "a_pg_dyn_x": a_pg_dyn[..., 0],
        "a_pg_dyn_y": a_pg_dyn[..., 1],
        "a_pg_h_x": a_pg_h[..., 0],
        "a_pg_h_y": a_pg_h[..., 1],
        "g_y": g_vec[..., 1],
        "res_current_norm": np.linalg.norm(a_res_current, axis=-1),
        "res_cap_norm": np.linalg.norm(a_res_cap, axis=-1),
        "rhs_raw": rhs_raw,
        "rhs_smoothed": rhs_sm,
        "lhs_varcoef": lhs,
        "ppe_mismatch_raw": lhs - rhs_raw,
        "ppe_mismatch_smoothed": lhs - rhs_sm,
    }


def _plot_bundle(fields: Dict[str, np.ndarray], out_dir: Path, interface_band: float) -> None:
    phi = fields["phi"]
    m_if = np.abs(phi) < float(interface_band)
    si, sj = _bbox_from_mask(m_if, pad=6)

    # Full interface-masked maps.
    fig1, ax1 = plt.subplots(4, 4, figsize=(18, 16), constrained_layout=True)
    names = [
        ("kappa", True), ("grad_phase_mag", False), ("sf_x", True), ("sf_y", True),
        ("a_sf_x", True), ("a_sf_y", True), ("a_pg_dyn_x", True), ("a_pg_dyn_y", True),
        ("a_pg_h_x", True), ("a_pg_h_y", True), ("g_y", True), ("rhs_raw", True),
        ("rhs_smoothed", True), ("lhs_varcoef", True), ("ppe_mismatch_smoothed", True), ("res_current_norm", False),
    ]
    for k, (name, signed) in enumerate(names):
        i = k // 4
        j = k % 4
        _plot(ax1[i, j], _mask_field(fields[name], m_if), name, signed=signed)
    fig1.suptitle("Interface-masked equation terms")
    fig1.savefig(out_dir / "terms_interface_masked.png", dpi=180)
    plt.close(fig1)

    # Cropped around interface region.
    fig2, ax2 = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    names2 = [
        ("a_sf_x", True), ("a_pg_dyn_x", True), ("a_pg_h_x", True), ("res_cap_norm", False),
        ("a_sf_y", True), ("a_pg_dyn_y", True), ("a_pg_h_y", True), ("res_current_norm", False),
        ("rhs_raw", True), ("rhs_smoothed", True), ("lhs_varcoef", True), ("ppe_mismatch_smoothed", True),
    ]
    for k, (name, signed) in enumerate(names2):
        i = k // 4
        j = k % 4
        _plot(ax2[i, j], fields[name][si, sj], f"{name} [crop]", signed=signed)
    fig2.suptitle("Interface crop: term-by-term")
    fig2.savefig(out_dir / "terms_interface_crop.png", dpi=180)
    plt.close(fig2)

    np.savez_compressed(out_dir / "term_fields.npz", **fields)


def _checkpoint_list(exp_dir: Path, selected: List[str]) -> List[Path]:
    ck_dir = exp_dir / "checkpoints"
    if selected:
        out = []
        for c in selected:
            p = ck_dir / c if c.endswith(".npz") else ck_dir / f"{c}.npz"
            if p.exists():
                out.append(p)
        return out
    return sorted(ck_dir.glob("checkpoint_*.npz"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--interface-band", type=float, default=0.75)
    p.add_argument("--out-dir", default="diagnostics/interface_equation_heatmaps")
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    cfg_path = exp / "simulation_parameters.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    checkpoints = _checkpoint_list(exp, args.checkpoints)
    root_out = Path(args.out_dir) / exp.name
    root_out.mkdir(parents=True, exist_ok=True)

    for cp in checkpoints:
        d = np.load(cp)
        phi = np.array(d["phi"], dtype=np.float64)
        p_total = np.array(d["P"], dtype=np.float64)
        fields = _term_fields(cfg, phi, p_total)
        out = root_out / cp.stem
        out.mkdir(parents=True, exist_ok=True)
        _plot_bundle(fields, out, args.interface_band)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()

