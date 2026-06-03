#!/usr/bin/env python3
"""
Standalone A/B diagnostic for capillary-pressure coupling paths.

Route A (current-like):
  F_sigma on cell centers -> rhs = div_cc(F_sigma)

Route B (fully staggered prototype):
  F_sigma constructed on faces directly from phase field operators ->
  rhs = div_face(F_sigma_face)

For each checkpoint, script solves variable-density pressure from both RHS and compares:
  - pressure-gradient vs capillary acceleration magnitudes
  - residuals (current-sign and capillary-consistent forms)
  - jump metrics (x/y second-difference severity) of total acceleration magnitude
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import jax.numpy as jnp
import numpy as np

from numerics.finite_differences import jax_divergence, jax_gradient, jax_norm
from numerics.staggered_mac import grad_p_to_faces
from physics.pressure import (
    _build_variable_coefficient_pressure_matrix,
    _solve_variable_coefficient_pressure,
)
from physics.properties import jax_calculate_density
from physics.surface_tension import jax_curvature, jax_curvature_smooth


@dataclass
class RouteMetrics:
    checkpoint: str
    route: str
    sf_norm_mean_cl: float
    pg_norm_mean_cl: float
    res_current_norm_mean_cl: float
    res_cap_norm_mean_cl: float
    res_current_gas_liq_ratio: float
    jump_nd2x_roi: float
    jump_nd2y_roi: float
    jump_anisotropy_x_over_y: float
    sf_n_row0_abs_mean: float
    sf_n_row1_abs_mean: float
    sf_n_row2_abs_mean: float
    pg_n_row0_abs_mean: float
    pg_n_row1_abs_mean: float
    pg_n_row2_abs_mean: float
    res_n_row0_abs_mean: float
    res_n_row1_abs_mean: float
    res_n_row2_abs_mean: float
    rhs_xparity_row2: float
    resmag_xparity_row2: float


def _row_abs_normal_means(vec_cc: np.ndarray, n_hat: np.ndarray, cl_mask: np.ndarray) -> Tuple[float, float, float]:
    out = []
    ny = vec_cc.shape[1]
    rows = [0, 1, 2]
    for j in rows:
        if j >= ny:
            out.append(0.0)
            continue
        m = cl_mask & (np.arange(ny)[None, :] == j)
        if not np.any(m):
            out.append(0.0)
            continue
        dot = np.sum(vec_cc[m] * n_hat[m], axis=1)
        out.append(float(np.mean(np.abs(dot))))
    return out[0], out[1], out[2]


def _we_map(phi: np.ndarray, we1: float, we2: float, mode: str) -> np.ndarray:
    c = 0.5 * (phi + 1.0)
    if mode == "harmonic":
        return 1.0 / (((1.0 - c) / we2) + (c / we1))
    if mode == "arithmetic":
        return (1.0 - c) * we2 + c * we1
    if mode == "constant_liquid":
        return np.full_like(phi, float(we2))
    raise ValueError(f"Unknown mode: {mode}")


def _surface_tension_cc(
    phi: np.ndarray,
    dx: float,
    dy: float,
    epsilon: float,
    we1: float,
    we2: float,
    mode: str,
    smooth_curvature: bool,
    smoothing_radius: int,
    use_composition_field: bool,
    composition_force_scale: float,
) -> np.ndarray:
    f1 = jnp.zeros_like(jnp.array(phi))
    phase = 0.5 * (jnp.array(phi) + 1.0) if use_composition_field else jnp.array(phi)
    if smooth_curvature:
        kappa = jax_curvature_smooth(phase, dx, dy, f1, smoothing_radius=smoothing_radius)
    else:
        kappa = jax_curvature(phase, dx, dy, f1)
    kappa2 = jnp.stack([kappa, kappa], axis=-1)
    grad = jax_gradient(phase, dx, dy, f1)
    normg = jax_norm(grad)
    normg2 = jnp.stack([normg, normg], axis=-1)
    we = _we_map(phi, we1, we2, mode)
    we2map = np.stack([we, we], axis=-1)
    sf = (
        float(composition_force_scale)
        * (3.0 * np.sqrt(2.0) * float(epsilon) / 4.0)
        * np.array(kappa2)
        * np.array(normg2)
        * np.array(grad)
    )
    sf = sf / np.maximum(we2map, 1e-12)
    return sf


def _sf_faces_discrete_from_phase(
    phi: np.ndarray,
    dx: float,
    dy: float,
    epsilon: float,
    we1: float,
    we2: float,
    mode: str,
    smooth_curvature: bool,
    smoothing_radius: int,
    use_composition_field: bool,
    composition_force_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    phase = 0.5 * (phi + 1.0) if use_composition_field else phi
    phase_j = jnp.array(phase)
    f1 = jnp.zeros_like(phase_j)
    if smooth_curvature:
        kappa_cc = np.array(jax_curvature_smooth(phase_j, dx, dy, f1, smoothing_radius=smoothing_radius))
    else:
        kappa_cc = np.array(jax_curvature(phase_j, dx, dy, f1))

    nx, ny = phase.shape
    dcdx_u = np.zeros((nx + 1, ny), dtype=np.float64)
    dcdy_v = np.zeros((nx, ny + 1), dtype=np.float64)
    dcdx_u[1:nx, :] = (phase[1:, :] - phase[:-1, :]) / max(dx, 1e-12)
    dcdx_u[0, :] = dcdx_u[1, :]
    dcdx_u[nx, :] = dcdx_u[nx - 1, :]
    dcdy_v[:, 1:ny] = (phase[:, 1:] - phase[:, :-1]) / max(dy, 1e-12)
    dcdy_v[:, 0] = dcdy_v[:, 1]
    dcdy_v[:, ny] = dcdy_v[:, ny - 1]

    dcdy_cc = np.zeros_like(phase, dtype=np.float64)
    dcdx_cc = np.zeros_like(phase, dtype=np.float64)
    dcdy_cc[:, 1:ny - 1] = (phase[:, 2:] - phase[:, :-2]) / max(2.0 * dy, 1e-12)
    dcdx_cc[1:nx - 1, :] = (phase[2:, :] - phase[:-2, :]) / max(2.0 * dx, 1e-12)
    dcdy_cc[:, 0] = dcdy_cc[:, 1]
    dcdy_cc[:, -1] = dcdy_cc[:, -2]
    dcdx_cc[0, :] = dcdx_cc[1, :]
    dcdx_cc[-1, :] = dcdx_cc[-2, :]

    dcdy_u = np.zeros((nx + 1, ny), dtype=np.float64)
    dcdx_v = np.zeros((nx, ny + 1), dtype=np.float64)
    dcdy_u[1:nx, :] = 0.5 * (dcdy_cc[1:, :] + dcdy_cc[:-1, :])
    dcdy_u[0, :] = dcdy_cc[0, :]
    dcdy_u[nx, :] = dcdy_cc[-1, :]
    dcdx_v[:, 1:ny] = 0.5 * (dcdx_cc[:, 1:] + dcdx_cc[:, :-1])
    dcdx_v[:, 0] = dcdx_cc[:, 0]
    dcdx_v[:, ny] = dcdx_cc[:, -1]

    gradmag_u = np.sqrt(dcdx_u * dcdx_u + dcdy_u * dcdy_u)
    gradmag_v = np.sqrt(dcdx_v * dcdx_v + dcdy_v * dcdy_v)

    kappa_u = np.zeros((nx + 1, ny), dtype=np.float64)
    kappa_v = np.zeros((nx, ny + 1), dtype=np.float64)
    kappa_u[1:nx, :] = 0.5 * (kappa_cc[1:, :] + kappa_cc[:-1, :])
    kappa_u[0, :] = kappa_cc[0, :]
    kappa_u[nx, :] = kappa_cc[-1, :]
    kappa_v[:, 1:ny] = 0.5 * (kappa_cc[:, 1:] + kappa_cc[:, :-1])
    kappa_v[:, 0] = kappa_cc[:, 0]
    kappa_v[:, ny] = kappa_cc[:, -1]

    we_cc = _we_map(phi, we1, we2, mode)
    we_u = np.zeros((nx + 1, ny), dtype=np.float64)
    we_v = np.zeros((nx, ny + 1), dtype=np.float64)
    we_u[1:nx, :] = 0.5 * (we_cc[1:, :] + we_cc[:-1, :])
    we_u[0, :] = we_cc[0, :]
    we_u[nx, :] = we_cc[-1, :]
    we_v[:, 1:ny] = 0.5 * (we_cc[:, 1:] + we_cc[:, :-1])
    we_v[:, 0] = we_cc[:, 0]
    we_v[:, ny] = we_cc[:, -1]

    coeff = float(composition_force_scale) * (3.0 * np.sqrt(2.0) * float(epsilon) / 4.0)
    sf_u = coeff * kappa_u * gradmag_u * dcdx_u / np.maximum(we_u, 1e-12)
    sf_v = coeff * kappa_v * gradmag_v * dcdy_v / np.maximum(we_v, 1e-12)
    return sf_u, sf_v


def _inv_rho_faces(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _cc_to_faces(sf_cc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny, _ = sf_cc.shape
    sf_u = np.zeros((nx + 1, ny), dtype=np.float64)
    sf_v = np.zeros((nx, ny + 1), dtype=np.float64)
    sf_u[1:nx, :] = 0.5 * (sf_cc[1:, :, 0] + sf_cc[:-1, :, 0])
    sf_u[0, :] = sf_cc[0, :, 0]
    sf_u[nx, :] = sf_cc[-1, :, 0]
    sf_v[:, 1:ny] = 0.5 * (sf_cc[:, 1:, 1] + sf_cc[:, :-1, 1])
    sf_v[:, 0] = sf_cc[:, 0, 1]
    sf_v[:, ny] = sf_cc[:, -1, 1]
    return sf_u, sf_v


def _div_faces(sf_u: np.ndarray, sf_v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (sf_u[1:, :] - sf_u[:-1, :]) / dx + (sf_v[:, 1:] - sf_v[:, :-1]) / dy


def _smooth_3x3_edge(field: np.ndarray) -> np.ndarray:
    p = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def _smooth_rhs(rhs: np.ndarray, radius: int) -> np.ndarray:
    out = np.array(rhs, copy=True)
    for _ in range(max(int(radius), 0)):
        out = _smooth_3x3_edge(out)
    return out


def _xparity_row(field: np.ndarray, row: int, x0: float = 0.30, x1: float = 0.70) -> float:
    nx, ny = field.shape
    if row < 0 or row >= ny:
        return 0.0
    x = np.linspace(0.0, 1.0, nx)
    m = (x >= x0) & (x <= x1)
    if not np.any(m):
        return 0.0
    v = np.asarray(field[m, row], dtype=np.float64)
    if v.size == 0:
        return 0.0
    alt = np.where((np.arange(v.size) % 2) == 0, 1.0, -1.0)
    num = abs(np.mean(alt * v))
    den = max(np.mean(np.abs(v)), 1e-30)
    return float(num / den)


def _apply_sf_face_bcs(
    sf_u: np.ndarray,
    sf_v: np.ndarray,
    contact_angle_deg: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply diagnostic face-BC variants to SF faces on a flat wall.
    - none: leave as-is
    - copy: zero-gradient copy on all domain boundaries
    - contact_angle_copy: copy + bottom v-face scaled by cos(theta)
    """
    if mode == "none":
        return sf_u, sf_v

    su = np.array(sf_u, copy=True)
    sv = np.array(sf_v, copy=True)
    nx = sv.shape[0]
    ny = su.shape[1]

    # Side boundaries
    su[0, :] = su[1, :]
    su[nx, :] = su[nx - 1, :]
    sv[0, :] = sv[1, :]
    sv[nx - 1, :] = sv[nx - 2, :]

    # Top boundary
    su[:, ny - 1] = su[:, ny - 2]
    sv[:, ny] = sv[:, ny - 1]

    # Bottom boundary
    su[:, 0] = su[:, 1]
    if mode == "contact_angle_copy":
        # Match runtime surface-tension BC: physical liquid-side angle.
        theta = float(contact_angle_deg) * np.pi / 180.0
        sv[:, 0] = sv[:, 1] * np.cos(theta)
    else:
        sv[:, 0] = sv[:, 1]
    return su, sv


def _solve_pdyn(rhs: np.ndarray, inv_u: np.ndarray, inv_v: np.ndarray, dx: float, dy: float, bcs: Dict[str, str]) -> np.ndarray:
    A, all_neu, gauge_ij = _build_variable_coefficient_pressure_matrix(inv_u, inv_v, dx, dy, bcs)
    rhs_use = rhs.copy()
    if all_neu and gauge_ij is not None:
        gi, gj = gauge_ij
        rhs_use[gi, gj] = 0.0
    return np.array(_solve_variable_coefficient_pressure(A, rhs_use, tol=1e-10, maxiter=3000), dtype=np.float64)


def _cl_mask(phi: np.ndarray) -> np.ndarray:
    nx, ny = phi.shape
    phi_bot = phi[:, 0]
    phi_above = phi[:, 1]
    contact_mask = ((phi_bot * phi_above) < 0.0) | (np.abs(phi_bot) < 0.5)
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


def _mean_norm(v: np.ndarray, m: np.ndarray) -> float:
    if not np.any(m):
        return 0.0
    return float(np.mean(np.linalg.norm(v[m], axis=1)))


def _jump_metrics_from_cc_vector(a_cc: np.ndarray, phi: np.ndarray) -> Tuple[float, float, float]:
    mag = np.linalg.norm(a_cc, axis=-1)
    nx, ny = mag.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    ix = (x >= 0.30) & (x <= 0.70)
    iy = (y <= 0.20)
    roi = mag[np.ix_(ix, iy)]
    d2x = np.abs(np.diff(mag, 2, axis=0))
    d2y = np.abs(np.diff(mag, 2, axis=1))
    m2x = (y[None, :] <= 0.20) & (x[1:-1, None] >= 0.30) & (x[1:-1, None] <= 0.70)
    m2y = (y[1:-1][None, :] <= 0.20) & (x[:, None] >= 0.30) & (x[:, None] <= 0.70)
    nd2x = float(d2x[m2x].mean() / max(roi.mean(), 1e-30))
    nd2y = float(d2y[m2y].mean() / max(roi.mean(), 1e-30))
    anis = float(nd2x / max(nd2y, 1e-30))
    return nd2x, nd2y, anis


def _pressure_bcs_from_config(cfg: Dict) -> Dict[str, str]:
    pbc = cfg.get("boundary_conditions", {}).get("pressure", {})
    out = {}
    for b in ("left", "right", "bottom", "top"):
        raw = str(pbc.get(b, "neumann")).lower()
        out[b] = "dirichlet" if raw in ("open", "dirichlet") else "neumann"
    return out


def _route_metrics(
    checkpoint: str,
    route: str,
    phi: np.ndarray,
    rho: np.ndarray,
    p_dyn: np.ndarray,
    sf_u: np.ndarray,
    sf_v: np.ndarray,
    inv_u: np.ndarray,
    inv_v: np.ndarray,
    cl: np.ndarray,
    rhs: np.ndarray,
):
    dpdx_u, dpdy_v = grad_p_to_faces(jnp.array(p_dyn), dx, dy)
    dpdx_u = np.array(dpdx_u)
    dpdy_v = np.array(dpdy_v)
    a_pg_u = -inv_u * dpdx_u
    a_pg_v = -inv_v * dpdy_v
    a_sf_u = -inv_u * sf_u
    a_sf_v = -inv_v * sf_v

    # Map face accelerations to centers for unified diagnostics.
    a_pg = np.zeros((phi.shape[0], phi.shape[1], 2), dtype=np.float64)
    a_sf = np.zeros_like(a_pg)
    a_pg[..., 0] = 0.5 * (a_pg_u[1:, :] + a_pg_u[:-1, :])
    a_pg[..., 1] = 0.5 * (a_pg_v[:, 1:] + a_pg_v[:, :-1])
    a_sf[..., 0] = 0.5 * (a_sf_u[1:, :] + a_sf_u[:-1, :])
    a_sf[..., 1] = 0.5 * (a_sf_v[:, 1:] + a_sf_v[:, :-1])

    # Two residual definitions:
    # current-sign (to compare with current telemetry convention),
    # capillary-consistent (pressure-capillary cancellation check).
    a_res_current = a_sf + a_pg
    a_res_cap = (-a_sf) + a_pg

    liq = cl & (phi < 0.0)
    gas = cl & (phi >= 0.0)
    gas_liq_ratio = float(_mean_norm(a_res_current, gas) / max(_mean_norm(a_res_current, liq), 1e-30))
    nd2x, nd2y, anis = _jump_metrics_from_cc_vector(a_res_current, phi)

    grad_phi = np.array(jax_gradient(jnp.array(phi), dx, dy, jnp.zeros_like(jnp.array(phi))), dtype=np.float64)
    nrm = np.linalg.norm(grad_phi, axis=-1, keepdims=True)
    n_hat = grad_phi / np.maximum(nrm, 1e-12)
    sf_r0, sf_r1, sf_r2 = _row_abs_normal_means(a_sf, n_hat, cl)
    pg_r0, pg_r1, pg_r2 = _row_abs_normal_means(a_pg, n_hat, cl)
    rs_r0, rs_r1, rs_r2 = _row_abs_normal_means(a_res_current, n_hat, cl)
    res_mag = np.linalg.norm(a_res_current, axis=-1)
    rhs_xpar = _xparity_row(rhs, row=2)
    resmag_xpar = _xparity_row(res_mag, row=2)
    return RouteMetrics(
        checkpoint=checkpoint,
        route=route,
        sf_norm_mean_cl=_mean_norm(a_sf, cl),
        pg_norm_mean_cl=_mean_norm(a_pg, cl),
        res_current_norm_mean_cl=_mean_norm(a_res_current, cl),
        res_cap_norm_mean_cl=_mean_norm(a_res_cap, cl),
        res_current_gas_liq_ratio=gas_liq_ratio,
        jump_nd2x_roi=nd2x,
        jump_nd2y_roi=nd2y,
        jump_anisotropy_x_over_y=anis,
        sf_n_row0_abs_mean=sf_r0,
        sf_n_row1_abs_mean=sf_r1,
        sf_n_row2_abs_mean=sf_r2,
        pg_n_row0_abs_mean=pg_r0,
        pg_n_row1_abs_mean=pg_r1,
        pg_n_row2_abs_mean=pg_r2,
        res_n_row0_abs_mean=rs_r0,
        res_n_row1_abs_mean=rs_r1,
        res_n_row2_abs_mean=rs_r2,
        rhs_xparity_row2=rhs_xpar,
        resmag_xparity_row2=resmag_xpar,
    )


def evaluate_checkpoint(ckpt_path: str, cfg: Dict, we_mode: str, face_bc_mode: str, rhs_smoothing_radius: int) -> List[RouteMetrics]:
    d = np.load(ckpt_path)
    phi = np.array(d["phi"], dtype=np.float64)

    rho = np.array(
        jax_calculate_density(
            jnp.array(phi),
            float(cfg["physical_params"]["rho1"]),
            float(cfg["physical_params"]["rho2"]),
        )
    )
    inv_u, inv_v = _inv_rho_faces(rho)

    eps = float(cfg["physical_params"]["epsilon"])
    we1 = float(cfg["physical_params"]["We1"])
    we2 = float(cfg["physical_params"]["We2"])
    st_cfg = cfg.get("physical_params", {}).get("surface_tension", {})
    smooth = bool(st_cfg.get("smooth_curvature", True))
    srad = int(st_cfg.get("smoothing_radius", 1))
    use_c = bool(st_cfg.get("use_composition_field", True))
    c_scale = float(st_cfg.get("composition_force_scale", 1.0))
    contact_angle = float(cfg.get("physical_params", {}).get("contact_angle", 90.0))
    bcs = _pressure_bcs_from_config(cfg)

    sf_cc = _surface_tension_cc(
        phi, dx, dy, eps, we1, we2, we_mode,
        smooth_curvature=smooth,
        smoothing_radius=srad,
        use_composition_field=use_c,
        composition_force_scale=c_scale,
    )

    # Route A: current-like (cc divergence RHS)
    rhs_cc = np.array(jax_divergence(jnp.array(sf_cc), dx, dy, jnp.zeros_like(jnp.array(phi))), dtype=np.float64)
    rhs_cc = _smooth_rhs(rhs_cc, rhs_smoothing_radius)
    p_dyn_cc = _solve_pdyn(rhs_cc, inv_u, inv_v, dx, dy, bcs)
    sf_u_cc, sf_v_cc = _cc_to_faces(sf_cc)
    sf_u_cc, sf_v_cc = _apply_sf_face_bcs(sf_u_cc, sf_v_cc, contact_angle, mode=face_bc_mode)

    # Route B: fully staggered SF path (face-native SF + face divergence RHS)
    sf_u_stag, sf_v_stag = _sf_faces_discrete_from_phase(
        phi, dx, dy, eps, we1, we2, we_mode,
        smooth_curvature=smooth,
        smoothing_radius=srad,
        use_composition_field=use_c,
        composition_force_scale=c_scale,
    )
    sf_u_stag, sf_v_stag = _apply_sf_face_bcs(sf_u_stag, sf_v_stag, contact_angle, mode=face_bc_mode)
    rhs_face = _div_faces(sf_u_stag, sf_v_stag, dx, dy)
    rhs_face = _smooth_rhs(rhs_face, rhs_smoothing_radius)
    p_dyn_stag = _solve_pdyn(rhs_face, inv_u, inv_v, dx, dy, bcs)

    cl = _cl_mask(phi)
    ck_name = os.path.basename(ckpt_path)
    return [
        _route_metrics(ck_name, "current_cc_rhs", phi, rho, p_dyn_cc, sf_u_cc, sf_v_cc, inv_u, inv_v, cl, rhs_cc),
        _route_metrics(ck_name, "staggered_face_rhs", phi, rho, p_dyn_stag, sf_u_stag, sf_v_stag, inv_u, inv_v, cl, rhs_face),
    ]


def _write_csv(path: str, rows: Iterable[RouteMetrics]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument(
        "--weber-mode",
        choices=["harmonic", "arithmetic", "constant_liquid"],
        default="constant_liquid",
    )
    p.add_argument(
        "--out-csv",
        default="diagnostics/staggered_sf_path/route_compare.csv",
    )
    p.add_argument(
        "--face-bc-mode",
        choices=["none", "copy", "contact_angle_copy"],
        default="contact_angle_copy",
        help="Diagnostic BC treatment for SF face fields before div_face and evaluation.",
    )
    p.add_argument(
        "--rhs-smoothing-radius",
        type=int,
        default=0,
        help="3x3 smoothing passes on capillary RHS before pressure solve.",
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    global dx, dy
    dx = float(cfg["grid_params"]["Lx"]) / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])

    rows: List[RouteMetrics] = []
    for ck in args.checkpoints:
        if not os.path.exists(ck):
            print(f"skip missing: {ck}")
            continue
        r = evaluate_checkpoint(ck, cfg, args.weber_mode, args.face_bc_mode, args.rhs_smoothing_radius)
        rows.extend(r)
        for rr in r:
            print(
                f"{rr.checkpoint} | {rr.route:>18s} | "
                f"res_cur={rr.res_current_norm_mean_cl:.3e} | "
                f"res_cap={rr.res_cap_norm_mean_cl:.3e} | "
                f"jump(nd2x/nd2y)=({rr.jump_nd2x_roi:.3e}/{rr.jump_nd2y_roi:.3e}) | "
                f"res_n_rows=({rr.res_n_row0_abs_mean:.2e},{rr.res_n_row1_abs_mean:.2e},{rr.res_n_row2_abs_mean:.2e}) | "
                f"xpar(rhs/res@j2)=({rr.rhs_xparity_row2:.2e}/{rr.resmag_xparity_row2:.2e})"
            )

    _write_csv(args.out_csv, rows)
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()

