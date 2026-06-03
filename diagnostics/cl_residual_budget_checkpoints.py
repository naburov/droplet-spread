#!/usr/bin/env python3
"""
Checkpoint diagnostic for contact-line residual budgets.

Self-contained script (no repo imports) that mirrors runtime equations:
  a_sf     = -surface_tension / rho
  a_pg_h   = -grad(p_hydro) / rho
  a_pg_dyn = -grad(p_dyn) / rho, where p_dyn = P - p_hydro
  g_vec    = (0, g/Fr)
  a_res    = a_sf + a_pg_h + a_pg_dyn + g_vec
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class CheckpointSummary:
    checkpoint: str
    weber_mode: str
    pressure_route: str
    cl_cells: int
    cl_liquid_cells: int
    cl_gas_cells: int
    res_current_norm_mean_cl: float
    res_current_norm_mean_cl_liquid: float
    res_current_norm_mean_cl_gas: float
    res_current_n_mean_cl: float
    res_current_t_mean_cl: float
    res_current_n_abs_mean_cl: float
    res_current_t_abs_mean_cl: float
    res_cap_norm_mean_cl: float
    res_cap_norm_mean_cl_liquid: float
    res_cap_norm_mean_cl_gas: float
    res_cap_n_mean_cl: float
    res_cap_t_mean_cl: float
    res_cap_n_abs_mean_cl: float
    res_cap_t_abs_mean_cl: float
    res_current_to_cap_ratio: float
    gas_liq_ratio_current: float
    gas_liq_ratio_cap: float
    sf_norm_mean_cl: float
    pg_dyn_norm_mean_cl: float
    pg_h_norm_mean_cl: float
    g_norm: float
    sf_to_pg_dyn_ratio: float
    corner_share_res_current_l1: float
    boundary_share_res_current_l1: float
    cl_strip_share_res_current_l1: float
    interior_share_res_current_l1: float
    corner_share_res_cap_l1: float
    boundary_share_res_cap_l1: float
    cl_strip_share_res_cap_l1: float
    interior_share_res_cap_l1: float
    top1_region: str
    top2_region: str
    top3_region: str


def _jax_dx(f, h):
    df = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * h)
    df = df.at[0, :].set((f[1, :] - f[0, :]) / h)
    df = df.at[-1, :].set((f[-1, :] - f[-2, :]) / h)
    return df


def _jax_dy(f, h):
    df = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * h)
    df = df.at[:, 0].set((f[:, 1] - f[:, 0]) / h)
    df = df.at[:, -1].set((f[:, -1] - f[:, -2]) / h)
    return df


def _jax_gradient(f, dx, dy, f1_grid):
    phi_x = _jax_dx(f, dx)
    phi_eta = _jax_dy(f, dy)
    grad_x = phi_x - f1_grid * phi_eta
    grad_y = phi_eta
    return jnp.stack([grad_x, grad_y], axis=-1)


def _jax_divergence(v, dx, dy, f1_grid):
    u = v[..., 0]
    w = v[..., 1]
    du_dx = _jax_dx(u, dx)
    dw_dy = _jax_dy(w, dy)
    u_eta = _jax_dy(u, dy)
    return du_dx - f1_grid * u_eta + dw_dy


def _jax_norm(v):
    return jnp.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)


def _curvature(phi_for_curv, dx, dy, f1_grid):
    grad_phi = _jax_gradient(phi_for_curv, dx, dy, f1_grid)
    mag = jnp.maximum(_jax_norm(grad_phi), 1e-6)
    n = grad_phi / mag[..., None]
    return _jax_divergence(n, dx, dy, f1_grid)


def _smooth_3x3(curv):
    sm = curv.copy()
    sm = sm + jnp.roll(curv, 1, axis=0) + jnp.roll(curv, -1, axis=0)
    sm = sm + jnp.roll(curv, 1, axis=1) + jnp.roll(curv, -1, axis=1)
    sm = sm + jnp.roll(jnp.roll(curv, 1, axis=0), 1, axis=1)
    sm = sm + jnp.roll(jnp.roll(curv, 1, axis=0), -1, axis=1)
    sm = sm + jnp.roll(jnp.roll(curv, -1, axis=0), 1, axis=1)
    sm = sm + jnp.roll(jnp.roll(curv, -1, axis=0), -1, axis=1)
    return sm / 9.0


def _surface_tension(phi, cfg, dx, dy, f1_grid, f1_surface, weber_override=None):
    pp = cfg["physical_params"]
    st_cfg = pp.get("surface_tension", {})
    epsilon = float(pp["epsilon"])
    we1 = float(pp["We1"])
    we2 = float(pp["We2"])
    contact_angle = float(pp["contact_angle"])
    smooth = bool(st_cfg.get("smooth_curvature", True))
    radius = int(st_cfg.get("smoothing_radius", 1))
    use_comp = bool(st_cfg.get("use_composition_field", True))
    comp_scale = float(st_cfg.get("composition_force_scale", 1.0))
    we_mode = str(weber_override) if weber_override is not None else str(st_cfg.get("weber_interpolation", "constant_liquid"))

    phi_j = jnp.array(phi)
    phase = 0.5 * (phi_j + 1.0) if use_comp else phi_j
    kappa = _curvature(phase, dx, dy, f1_grid)
    if smooth:
        for _ in range(max(radius, 0)):
            kappa = _smooth_3x3(kappa)

    grad_phase = _jax_gradient(phase, dx, dy, f1_grid)
    norm_grad = _jax_norm(grad_phase)

    c = 0.5 * (phi_j + 1.0)
    if we_mode == "harmonic":
        we = 1.0 / (((1.0 - c) / we2) + (c / we1))
    elif we_mode == "arithmetic":
        we = (1.0 - c) * we2 + c * we1
    elif we_mode == "constant_liquid":
        we = jnp.full_like(phi_j, float(we2))
    else:
        raise ValueError(f"Unsupported weber_interpolation={we_mode}")

    coeff = comp_scale * (3.0 * jnp.sqrt(2.0) * epsilon / 4.0)
    sf = coeff * kappa[..., None] * norm_grad[..., None] * grad_phase / jnp.maximum(we[..., None], 1e-12)

    # Match runtime surface-tension BC internal projection.
    theta = (180.0 - contact_angle) * jnp.pi / 180.0
    if f1_surface is not None:
        nf = jnp.sqrt(1.0 + f1_surface**2)
        nx = -f1_surface / nf
        ny = 1.0 / nf
        sfx = sf[:, 0, 0]
        sfy = sf[:, 0, 1]
        sf_n = sfx * nx + sfy * ny
        sf_t = sfx * (-ny) + sfy * nx
        sf_n_adj = sf_n * jnp.cos(theta)
        sf = sf.at[:, 0, 0].set(sf_n_adj * nx - sf_t * ny)
        sf = sf.at[:, 0, 1].set(sf_n_adj * ny + sf_t * nx)
    else:
        sf = sf.at[:, 0, 1].set(sf[:, 1, 1] * jnp.cos(theta))
        sf = sf.at[:, 0, 0].set(sf[:, 1, 0])

    sf = sf.at[:, -1, 0].set(sf[:, -2, 0])
    sf = sf.at[:, -1, 1].set(sf[:, -2, 1])
    sf = sf.at[0, :, 0].set(sf[1, :, 0])
    sf = sf.at[0, :, 1].set(sf[1, :, 1])
    sf = sf.at[-1, :, 0].set(sf[-2, :, 0])
    sf = sf.at[-1, :, 1].set(sf[-2, :, 1])
    return np.array(sf)


def _density(phi, rho1, rho2):
    c = 0.5 * (phi + 1.0)
    return (1.0 - c) * rho2 + c * rho1


def _hydrostatic_pressure(rho, g, dy, fr, atm):
    nx, ny = rho.shape
    p = np.zeros_like(rho)
    p[:, -1] = atm
    for j in range(ny - 2, -1, -1):
        p[:, j] = p[:, j + 1] - rho[:, j] * g * dy / fr
    return p


def _f1_grid_from_config(cfg, nx, ny, dx, lx):
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


def _grad_p_faces(p, dx, dy):
    nx, ny = p.shape
    dpdx = np.zeros((nx + 1, ny), dtype=np.float64)
    dpdy = np.zeros((nx, ny + 1), dtype=np.float64)
    dpdx[1:nx, :] = (p[1:, :] - p[:-1, :]) / dx
    dpdy[:, 1:ny] = (p[:, 1:] - p[:, :-1]) / dy
    return dpdx, dpdy


def _inv_rho_faces(rho):
    nx, ny = rho.shape
    inv = 1.0 / np.maximum(rho, 1e-12)
    inv_u = np.zeros((nx + 1, ny), dtype=np.float64)
    inv_v = np.zeros((nx, ny + 1), dtype=np.float64)
    inv_u[1:nx, :] = 0.5 * (inv[1:, :] + inv[:-1, :])
    inv_u[0, :] = inv[0, :]
    inv_u[nx, :] = inv[-1, :]
    inv_v[:, 1:ny] = 0.5 * (inv[:, 1:] + inv[:, :-1])
    inv_v[:, 0] = inv[:, 0]
    inv_v[:, ny] = inv[:, -1]
    return inv_u, inv_v


def _cc_to_faces(sf):
    nx, ny, _ = sf.shape
    sf_u = np.zeros((nx + 1, ny), dtype=np.float64)
    sf_v = np.zeros((nx, ny + 1), dtype=np.float64)
    sf_u[1:nx, :] = 0.5 * (sf[1:, :, 0] + sf[:-1, :, 0])
    sf_u[0, :] = sf[0, :, 0]
    sf_u[nx, :] = sf[-1, :, 0]
    sf_v[:, 1:ny] = 0.5 * (sf[:, 1:, 1] + sf[:, :-1, 1])
    sf_v[:, 0] = sf[:, 0, 1]
    sf_v[:, ny] = sf[:, -1, 1]
    return sf_u, sf_v


def _cl_mask(phi):
    nx, ny = phi.shape
    phi_bottom = phi[:, 0]
    phi_above = phi[:, 1] if ny > 1 else phi[:, 0]
    contact_mask = ((phi_bottom * phi_above) < 0.0) | (np.abs(phi_bottom) < 0.5)
    idx = np.where(contact_mask)[0]
    if idx.size == 0:
        return np.zeros_like(phi, dtype=bool)
    i0 = max(0, int(idx.min()) - 2)
    i1 = min(nx - 1, int(idx.max()) + 2)
    region = np.zeros_like(phi, dtype=bool)
    region[i0 : i1 + 1, 0 : min(ny, 3)] = True
    interface_band = np.abs(phi) < 0.75
    m = region & interface_band
    if not np.any(m):
        m = region
    return m


def _mean_norm(v, m):
    if not np.any(m):
        return 0.0
    return float(np.mean(np.linalg.norm(v[m], axis=1)))


def _dot_mean(v, d, m, absolute=False):
    if not np.any(m):
        return 0.0
    dot = np.sum(v[m] * d[m], axis=1)
    if absolute:
        dot = np.abs(dot)
    return float(np.mean(dot))


def _region_breakdown(res_norm, cl_mask):
    nx, ny = res_norm.shape
    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    corner = ((i == 0) | (i == nx - 1)) & ((j == 0) | (j == ny - 1))
    boundary = ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)) & (~corner)
    cl_strip = cl_mask & (~corner) & (~boundary)
    interior = (~corner) & (~boundary) & (~cl_strip)
    total = float(np.sum(res_norm)) + 1e-30
    shares = {
        "corner": float(np.sum(res_norm[corner]) / total),
        "boundary": float(np.sum(res_norm[boundary]) / total),
        "cl_strip": float(np.sum(res_norm[cl_strip]) / total),
        "interior": float(np.sum(res_norm[interior]) / total),
    }
    ranked = [
        ("corner", float(np.sum(res_norm[corner]))),
        ("boundary_non_corner", float(np.sum(res_norm[boundary]))),
        ("cl_strip", float(np.sum(res_norm[cl_strip]))),
        ("interior", float(np.sum(res_norm[interior]))),
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return shares, ranked


def analyze_checkpoint(cfg, ckpt_path, out_dir, weber_override=None, pressure_route="raw"):
    data = np.load(ckpt_path)
    phi = np.array(data["phi"], dtype=np.float64)
    p_base = np.array(data["P"], dtype=np.float64)
    p_corr = np.array(data["p_corr_out"], dtype=np.float64) if "p_corr_out" in data.files else None
    if pressure_route == "with_pcorr" and p_corr is not None and p_corr.shape == p_base.shape:
        p = p_base + p_corr
    else:
        p = p_base
    nx, ny = phi.shape

    lx = float(cfg["grid_params"]["Lx"])
    ly = float(cfg["grid_params"]["Ly"])
    dx = lx / float(cfg["grid_params"]["Nx"])
    dy = ly / float(cfg["grid_params"]["Ny"])
    pp = cfg["physical_params"]
    rho1 = float(pp["rho1"])
    rho2 = float(pp["rho2"])
    g = float(pp["g"])
    fr = float(pp["Fr"])
    atm = float(pp.get("atm_pressure", 0.0))

    f1_grid = _f1_grid_from_config(cfg, nx, ny, dx, lx)
    f1_surface = f1_grid[:, 0] if np.max(np.abs(f1_grid)) > 1e-14 else None
    sf = _surface_tension(
        phi,
        cfg,
        dx,
        dy,
        jnp.array(f1_grid),
        (jnp.array(f1_surface) if f1_surface is not None else None),
        weber_override=weber_override,
    )

    rho = _density(phi, rho1, rho2)
    inv_rho = 1.0 / np.maximum(rho, 1e-12)

    p_h = _hydrostatic_pressure(rho, g, dy, fr, atm)
    p_dyn = p - p_h
    grad_h = np.array(_jax_gradient(jnp.array(p_h), dx, dy, jnp.array(f1_grid)))
    grad_dyn = np.array(_jax_gradient(jnp.array(p_dyn), dx, dy, jnp.array(f1_grid)))
    a_pg_h = -grad_h * inv_rho[..., None]
    a_pg_dyn = -grad_dyn * inv_rho[..., None]
    a_sf = -sf * inv_rho[..., None]
    g_vec = np.zeros_like(a_sf)
    g_vec[..., 1] = g / max(fr, 1e-12)
    a_res_current = a_sf + a_pg_dyn + a_pg_h + g_vec
    # Capillary-consistent balance form for pressure-capillary route.
    a_res_cap = (-a_sf) + a_pg_dyn + a_pg_h + g_vec

    grad_phi = np.array(_jax_gradient(jnp.array(phi), dx, dy, jnp.array(f1_grid)))
    nrm = np.linalg.norm(grad_phi, axis=-1, keepdims=True)
    n_hat = grad_phi / np.maximum(nrm, 1e-12)
    t_hat = np.concatenate([-n_hat[..., 1:2], n_hat[..., 0:1]], axis=-1)

    cl_mask = _cl_mask(phi)
    liq = cl_mask & (phi < 0.0)
    gas = cl_mask & (phi >= 0.0)

    inv_u, inv_v = _inv_rho_faces(rho)
    sf_u, sf_v = _cc_to_faces(sf)
    dpdx_u, dpdy_v = _grad_p_faces(p, dx, dy)
    a_sf_u = -(sf_u * inv_u)
    a_sf_v = -(sf_v * inv_v)
    a_pg_u = -(inv_u * dpdx_u)
    a_pg_v = -(inv_v * dpdy_v)
    g_v = np.full_like(a_pg_v, g / max(fr, 1e-12))
    a_res_current_u = a_sf_u + a_pg_u
    a_res_current_v = a_sf_v + a_pg_v + g_v
    a_res_cap_u = (-a_sf_u) + a_pg_u
    a_res_cap_v = (-a_sf_v) + a_pg_v + g_v

    res_current_norm = np.linalg.norm(a_res_current, axis=-1)
    res_cap_norm = np.linalg.norm(a_res_cap, axis=-1)
    shares_current, ranked = _region_breakdown(res_current_norm, cl_mask)
    shares_cap, _ = _region_breakdown(res_cap_norm, cl_mask)
    top = [name for name, _ in ranked[:3]]
    while len(top) < 3:
        top.append("n/a")

    ck_name = os.path.basename(ckpt_path).replace(".npz", "")
    out_ck = os.path.join(out_dir, f"we_{weber_override if weber_override is not None else 'config'}", f"p_{pressure_route}", ck_name)
    os.makedirs(out_ck, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_ck, "cl_cell_budget.npz"),
        phi=phi,
        rho=rho,
        cl_mask=cl_mask.astype(np.uint8),
        a_sf=a_sf,
        a_pg_dyn=a_pg_dyn,
        a_pg_h=a_pg_h,
        g_vec=g_vec,
        a_res_current=a_res_current,
        a_res_cap=a_res_cap,
        n_hat=n_hat,
        t_hat=t_hat,
        res_current_norm=res_current_norm,
        res_cap_norm=res_cap_norm,
        res_current_n=np.sum(a_res_current * n_hat, axis=-1),
        res_current_t=np.sum(a_res_current * t_hat, axis=-1),
        res_cap_n=np.sum(a_res_cap * n_hat, axis=-1),
        res_cap_t=np.sum(a_res_cap * t_hat, axis=-1),
        p_base=p_base,
        p_used=p,
        p_corr_out=p_corr if p_corr is not None else np.zeros_like(p_base),
    )
    np.savez_compressed(
        os.path.join(out_ck, "cl_face_budget.npz"),
        phi=phi,
        cl_mask=cl_mask.astype(np.uint8),
        inv_rho_u=inv_u,
        inv_rho_v=inv_v,
        sf_u=sf_u,
        sf_v=sf_v,
        a_sf_u=a_sf_u,
        a_sf_v=a_sf_v,
        a_pg_u=a_pg_u,
        a_pg_v=a_pg_v,
        g_v=g_v,
        a_res_current_u=a_res_current_u,
        a_res_current_v=a_res_current_v,
        a_res_cap_u=a_res_cap_u,
        a_res_cap_v=a_res_cap_v,
    )

    res_current_norm_mean = _mean_norm(a_res_current, cl_mask)
    res_cap_norm_mean = _mean_norm(a_res_cap, cl_mask)
    res_current_liq = _mean_norm(a_res_current, liq)
    res_current_gas = _mean_norm(a_res_current, gas)
    res_cap_liq = _mean_norm(a_res_cap, liq)
    res_cap_gas = _mean_norm(a_res_cap, gas)

    return CheckpointSummary(
        checkpoint=os.path.basename(ckpt_path),
        weber_mode=str(weber_override if weber_override is not None else cfg.get("physical_params", {}).get("surface_tension", {}).get("weber_interpolation", "constant_liquid")),
        pressure_route=str(pressure_route),
        cl_cells=int(np.sum(cl_mask)),
        cl_liquid_cells=int(np.sum(liq)),
        cl_gas_cells=int(np.sum(gas)),
        res_current_norm_mean_cl=res_current_norm_mean,
        res_current_norm_mean_cl_liquid=res_current_liq,
        res_current_norm_mean_cl_gas=res_current_gas,
        res_current_n_mean_cl=_dot_mean(a_res_current, n_hat, cl_mask, absolute=False),
        res_current_t_mean_cl=_dot_mean(a_res_current, t_hat, cl_mask, absolute=False),
        res_current_n_abs_mean_cl=_dot_mean(a_res_current, n_hat, cl_mask, absolute=True),
        res_current_t_abs_mean_cl=_dot_mean(a_res_current, t_hat, cl_mask, absolute=True),
        res_cap_norm_mean_cl=res_cap_norm_mean,
        res_cap_norm_mean_cl_liquid=res_cap_liq,
        res_cap_norm_mean_cl_gas=res_cap_gas,
        res_cap_n_mean_cl=_dot_mean(a_res_cap, n_hat, cl_mask, absolute=False),
        res_cap_t_mean_cl=_dot_mean(a_res_cap, t_hat, cl_mask, absolute=False),
        res_cap_n_abs_mean_cl=_dot_mean(a_res_cap, n_hat, cl_mask, absolute=True),
        res_cap_t_abs_mean_cl=_dot_mean(a_res_cap, t_hat, cl_mask, absolute=True),
        res_current_to_cap_ratio=float(res_current_norm_mean / max(res_cap_norm_mean, 1e-30)),
        gas_liq_ratio_current=float(res_current_gas / max(res_current_liq, 1e-30)),
        gas_liq_ratio_cap=float(res_cap_gas / max(res_cap_liq, 1e-30)),
        sf_norm_mean_cl=_mean_norm(a_sf, cl_mask),
        pg_dyn_norm_mean_cl=_mean_norm(a_pg_dyn, cl_mask),
        pg_h_norm_mean_cl=_mean_norm(a_pg_h, cl_mask),
        g_norm=abs(float(g / max(fr, 1e-12))),
        sf_to_pg_dyn_ratio=float(_mean_norm(a_sf, cl_mask) / max(_mean_norm(a_pg_dyn, cl_mask), 1e-30)),
        corner_share_res_current_l1=shares_current["corner"],
        boundary_share_res_current_l1=shares_current["boundary"],
        cl_strip_share_res_current_l1=shares_current["cl_strip"],
        interior_share_res_current_l1=shares_current["interior"],
        corner_share_res_cap_l1=shares_cap["corner"],
        boundary_share_res_cap_l1=shares_cap["boundary"],
        cl_strip_share_res_cap_l1=shares_cap["cl_strip"],
        interior_share_res_cap_l1=shares_cap["interior"],
        top1_region=top[0],
        top2_region=top[1],
        top3_region=top[2],
    )


def _load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_summary_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", default="diagnostics/cl_residual_budget")
    ap.add_argument(
        "--weber-mode-override",
        choices=["harmonic", "arithmetic", "constant_liquid"],
        default=None,
        help="Override Weber interpolation mode for replay diagnostics.",
    )
    ap.add_argument(
        "--weber-modes",
        nargs="+",
        choices=["harmonic", "arithmetic", "constant_liquid"],
        default=None,
        help="Run multiple Weber modes in one call (overrides --weber-mode-override).",
    )
    ap.add_argument(
        "--pressure-routes",
        nargs="+",
        choices=["raw", "with_pcorr"],
        default=["raw"],
        help="Pressure route A/B: raw P or P + p_corr_out if available.",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.weber_modes:
        weber_modes = list(args.weber_modes)
    elif args.weber_mode_override is not None:
        weber_modes = [args.weber_mode_override]
    else:
        weber_modes = [str(cfg.get("physical_params", {}).get("surface_tension", {}).get("weber_interpolation", "constant_liquid"))]

    rows: List[CheckpointSummary] = []
    for we_mode in weber_modes:
        for p_route in args.pressure_routes:
            for ck in args.checkpoints:
                if not os.path.exists(ck):
                    print(f"skip missing: {ck}")
                    continue
                r = analyze_checkpoint(cfg, ck, out_dir, weber_override=we_mode, pressure_route=p_route)
                rows.append(r)
                print(
                    f"{r.checkpoint} [{r.weber_mode}/{r.pressure_route}]: "
                    f"r_cur={r.res_current_norm_mean_cl:.3e}, r_cap={r.res_cap_norm_mean_cl:.3e}, "
                    f"ratio={r.res_current_to_cap_ratio:.2f}, "
                    f"gas/liq(cur)={r.gas_liq_ratio_current:.2f}"
                )

    if rows:
        csv_path = os.path.join(out_dir, "summary.csv")
        json_path = os.path.join(out_dir, "summary.json")
        compare_csv = os.path.join(out_dir, "comparison_table.csv")
        _write_summary_csv(csv_path, rows)
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        # Compact human table for quick A/B read.
        with open(compare_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "checkpoint",
                    "weber_mode",
                    "pressure_route",
                    "r_current",
                    "r_cap",
                    "r_current_to_cap_ratio",
                    "sf_to_pg_dyn_ratio",
                    "gas_liq_ratio_current",
                    "gas_liq_ratio_cap",
                    "normal_tangential_ratio_current",
                    "normal_tangential_ratio_cap",
                ]
            )
            for r in rows:
                ntr_cur = r.res_current_n_abs_mean_cl / max(r.res_current_t_abs_mean_cl, 1e-30)
                ntr_cap = r.res_cap_n_abs_mean_cl / max(r.res_cap_t_abs_mean_cl, 1e-30)
                w.writerow(
                    [
                        r.checkpoint,
                        r.weber_mode,
                        r.pressure_route,
                        r.res_current_norm_mean_cl,
                        r.res_cap_norm_mean_cl,
                        r.res_current_to_cap_ratio,
                        r.sf_to_pg_dyn_ratio,
                        r.gas_liq_ratio_current,
                        r.gas_liq_ratio_cap,
                        ntr_cur,
                        ntr_cap,
                    ]
                )
        print(f"saved: {csv_path}")
        print(f"saved: {json_path}")
        print(f"saved: {compare_csv}")


if __name__ == "__main__":
    main()

