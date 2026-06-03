#!/usr/bin/env python3
"""
Checkpoint A/B: cell-centered vs face-balanced capillary-pressure coupling.

For each checkpoint/mode, we evaluate:
  1) legacy_cc:
       rhs = div(F_sigma_cc)
       a_sf = -(F_sigma_cc / rho_cc)
       a_pg = face-consistent -(1/rho_face) grad(P_dyn) mapped to cell centers
       residual = a_sf + a_pg
  2) balanced_face_interp:
       F_sigma on faces from centered interpolation of F_sigma_cc components
       rhs = div_face(F_sigma_face)
       a_sf(face) = -(F_sigma_face / rho_face), mapped to cell centers
       a_pg same face-consistent mapping
       residual = a_sf(face->cc) + a_pg(face->cc)
  3) balanced_face_discrete:
       F_sigma is constructed natively on faces from phase-field discrete operators
       (not from interpolated cell-centered force), then used in both momentum and PPE RHS.

The goal is to check whether using the same face operator family for capillary and
pressure terms reduces CL residuals and gas-side overforcing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
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
class CaseResult:
    checkpoint: str
    mode: str
    route: str
    sf_norm_mean: float
    pg_norm_mean: float
    res_norm_mean: float
    sf_to_pg: float
    res_norm_mean_liquid: float
    res_norm_mean_gas: float
    sf_norm_mean_liquid: float
    sf_norm_mean_gas: float
    pg_norm_mean_liquid: float
    pg_norm_mean_gas: float


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
    smooth_curvature: bool = True,
    smoothing_radius: int = 1,
    use_composition_field: bool = True,
    composition_force_scale: float = 1.0,
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


def _pressure_bcs_from_config(cfg: Dict) -> Dict[str, str]:
    pbc = cfg.get("boundary_conditions", {}).get("pressure", {})
    out = {}
    for b in ("left", "right", "bottom", "top"):
        raw = str(pbc.get(b, "neumann")).lower()
        out[b] = "dirichlet" if raw in ("open", "dirichlet") else "neumann"
    return out


def _inv_rho_faces(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = rho.shape
    inv_rho = 1.0 / np.maximum(rho, 1e-6)
    inv_u = np.zeros((nx + 1, ny), dtype=np.float64)
    inv_v = np.zeros((nx, ny + 1), dtype=np.float64)
    inv_u[1:nx, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
    inv_u[0, :] = inv_rho[0, :]
    inv_u[nx, :] = inv_rho[-1, :]
    inv_v[:, 1:ny] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    inv_v[:, 0] = inv_rho[:, 0]
    inv_v[:, ny] = inv_rho[:, -1]
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


def _sf_faces_discrete_from_phase(
    phi: np.ndarray,
    dx: float,
    dy: float,
    epsilon: float,
    we1: float,
    we2: float,
    mode: str,
    smooth_curvature: bool = True,
    smoothing_radius: int = 1,
    use_composition_field: bool = True,
    composition_force_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build capillary force directly on MAC faces from phase field.

    This is a diagnostic balanced-force prototype:
      - curvature is computed at cell centers,
      - mapped to faces,
      - multiplied by face gradient component and face gradient magnitude.
    """
    phase = 0.5 * (phi + 1.0) if use_composition_field else phi
    phase_j = jnp.array(phase)
    f1 = jnp.zeros_like(phase_j)
    if smooth_curvature:
        kappa_cc = np.array(jax_curvature_smooth(phase_j, dx, dy, f1, smoothing_radius=smoothing_radius))
    else:
        kappa_cc = np.array(jax_curvature(phase_j, dx, dy, f1))

    nx, ny = phase.shape
    # Face gradients (normal direction)
    dcdx_u = np.zeros((nx + 1, ny), dtype=np.float64)
    dcdy_v = np.zeros((nx, ny + 1), dtype=np.float64)
    dcdx_u[1:nx, :] = (phase[1:, :] - phase[:-1, :]) / max(dx, 1e-12)
    dcdx_u[0, :] = dcdx_u[1, :]
    dcdx_u[nx, :] = dcdx_u[nx - 1, :]
    dcdy_v[:, 1:ny] = (phase[:, 1:] - phase[:, :-1]) / max(dy, 1e-12)
    dcdy_v[:, 0] = dcdy_v[:, 1]
    dcdy_v[:, ny] = dcdy_v[:, ny - 1]

    # Tangential gradients mapped to faces for |grad c|
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

    # Curvature on faces
    kappa_u = np.zeros((nx + 1, ny), dtype=np.float64)
    kappa_v = np.zeros((nx, ny + 1), dtype=np.float64)
    kappa_u[1:nx, :] = 0.5 * (kappa_cc[1:, :] + kappa_cc[:-1, :])
    kappa_u[0, :] = kappa_cc[0, :]
    kappa_u[nx, :] = kappa_cc[-1, :]
    kappa_v[:, 1:ny] = 0.5 * (kappa_cc[:, 1:] + kappa_cc[:, :-1])
    kappa_v[:, 0] = kappa_cc[:, 0]
    kappa_v[:, ny] = kappa_cc[:, -1]

    # Weber map at centers then to faces
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


def _div_faces(sf_u: np.ndarray, sf_v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (sf_u[1:, :] - sf_u[:-1, :]) / dx + (sf_v[:, 1:] - sf_v[:, :-1]) / dy


def _solve_pdyn(rhs: np.ndarray, inv_u: np.ndarray, inv_v: np.ndarray, dx: float, dy: float, bcs: Dict[str, str]) -> np.ndarray:
    A, all_neu, gauge_ij = _build_variable_coefficient_pressure_matrix(inv_u, inv_v, dx, dy, bcs)
    rhs_use = rhs.copy()
    if all_neu and gauge_ij is not None:
        gi, gj = gauge_ij
        rhs_use[gi, gj] = 0.0
    return np.array(_solve_variable_coefficient_pressure(A, rhs_use, tol=1e-10, maxiter=3000), dtype=np.float64)


def _map_face_to_cc(ax_u: np.ndarray, ay_v: np.ndarray) -> np.ndarray:
    nx = ax_u.shape[0] - 1
    ny = ax_u.shape[1]
    out = np.zeros((nx, ny, 2), dtype=np.float64)
    out[..., 0] = 0.5 * (ax_u[1:, :] + ax_u[:-1, :])
    out[..., 1] = 0.5 * (ay_v[:, 1:] + ay_v[:, :-1])
    return out


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


def _evaluate_checkpoint(ckpt: str, cfg: Dict, mode: str) -> List[CaseResult]:
    d = np.load(ckpt)
    phi = np.array(d["phi"], dtype=np.float64)
    rho = np.array(
        jax_calculate_density(
            jnp.array(phi),
            float(cfg["physical_params"]["rho1"]),
            float(cfg["physical_params"]["rho2"]),
        )
    )
    dx = float(cfg["grid_params"]["Lx"]) / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])
    eps = float(cfg["physical_params"]["epsilon"])
    we1 = float(cfg["physical_params"]["We1"])
    we2 = float(cfg["physical_params"]["We2"])
    st_cfg = cfg.get("physical_params", {}).get("surface_tension", {})
    smooth = bool(st_cfg.get("smooth_curvature", True))
    srad = int(st_cfg.get("smoothing_radius", 1))
    use_c = bool(st_cfg.get("use_composition_field", True))
    c_scale = float(st_cfg.get("composition_force_scale", 1.0))
    bcs = _pressure_bcs_from_config(cfg)

    sf_cc = _surface_tension_cc(
        phi, dx, dy, eps, we1, we2, mode,
        smooth_curvature=smooth,
        smoothing_radius=srad,
        use_composition_field=use_c,
        composition_force_scale=c_scale,
    )
    inv_u, inv_v = _inv_rho_faces(rho)

    # Shared pressure acceleration on faces from solved P.
    def _pg_cc_from_p(p_dyn: np.ndarray) -> np.ndarray:
        dpdx_u, dpdy_v = grad_p_to_faces(jnp.array(p_dyn), dx, dy)
        ax_u = -inv_u * np.array(dpdx_u)
        ay_v = -inv_v * np.array(dpdy_v)
        return _map_face_to_cc(ax_u, ay_v)

    # Route 1: legacy cell-centered SF RHS
    rhs_cc = np.array(jax_divergence(jnp.array(sf_cc), dx, dy, jnp.zeros_like(jnp.array(rho))), dtype=np.float64)
    p_dyn_cc = _solve_pdyn(rhs_cc, inv_u, inv_v, dx, dy, bcs)
    a_pg_cc_route = _pg_cc_from_p(p_dyn_cc)
    a_sf_cc_route = -(sf_cc / np.maximum(rho, 1e-12)[..., None])
    res_cc_route = a_sf_cc_route + a_pg_cc_route

    # Route 2: face-balanced SF RHS from interpolated CC SF
    sf_u, sf_v = _cc_to_faces(sf_cc)
    rhs_face = _div_faces(sf_u, sf_v, dx, dy)
    p_dyn_face = _solve_pdyn(rhs_face, inv_u, inv_v, dx, dy, bcs)
    a_pg_face_route = _pg_cc_from_p(p_dyn_face)
    a_sf_face_cc = _map_face_to_cc(-inv_u * sf_u, -inv_v * sf_v)
    res_face_route = a_sf_face_cc + a_pg_face_route

    # Route 3: face-balanced SF built directly on faces from phase
    sf_u_d, sf_v_d = _sf_faces_discrete_from_phase(
        phi, dx, dy, eps, we1, we2, mode,
        smooth_curvature=smooth,
        smoothing_radius=srad,
        use_composition_field=use_c,
        composition_force_scale=c_scale,
    )
    rhs_face_d = _div_faces(sf_u_d, sf_v_d, dx, dy)
    p_dyn_face_d = _solve_pdyn(rhs_face_d, inv_u, inv_v, dx, dy, bcs)
    a_pg_face_d = _pg_cc_from_p(p_dyn_face_d)
    a_sf_face_cc_d = _map_face_to_cc(-inv_u * sf_u_d, -inv_v * sf_v_d)
    res_face_d = a_sf_face_cc_d + a_pg_face_d

    m = _cl_mask(phi)
    m_liq = m & (phi < 0.0)
    m_gas = m & (phi >= 0.0)

    def _pack(route: str, a_sf: np.ndarray, a_pg: np.ndarray, r: np.ndarray) -> CaseResult:
        sf_norm = _mean_norm(a_sf, m)
        pg_norm = _mean_norm(a_pg, m)
        return CaseResult(
            checkpoint=os.path.basename(ckpt),
            mode=mode,
            route=route,
            sf_norm_mean=sf_norm,
            pg_norm_mean=pg_norm,
            res_norm_mean=_mean_norm(r, m),
            sf_to_pg=float(sf_norm / max(pg_norm, 1e-12)),
            res_norm_mean_liquid=_mean_norm(r, m_liq),
            res_norm_mean_gas=_mean_norm(r, m_gas),
            sf_norm_mean_liquid=_mean_norm(a_sf, m_liq),
            sf_norm_mean_gas=_mean_norm(a_sf, m_gas),
            pg_norm_mean_liquid=_mean_norm(a_pg, m_liq),
            pg_norm_mean_gas=_mean_norm(a_pg, m_gas),
        )

    return [
        _pack("legacy_cc", a_sf_cc_route, a_pg_cc_route, res_cc_route),
        _pack("balanced_face_interp", a_sf_face_cc, a_pg_face_route, res_face_route),
        _pack("balanced_face_discrete", a_sf_face_cc_d, a_pg_face_d, res_face_d),
    ]


def _write_csv(path: str, rows: Iterable[CaseResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "mode",
                "route",
                "sf_norm_mean",
                "pg_norm_mean",
                "res_norm_mean",
                "sf_to_pg",
                "res_norm_mean_liquid",
                "res_norm_mean_gas",
                "sf_norm_mean_liquid",
                "sf_norm_mean_gas",
                "pg_norm_mean_liquid",
                "pg_norm_mean_gas",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.checkpoint,
                    r.mode,
                    r.route,
                    r.sf_norm_mean,
                    r.pg_norm_mean,
                    r.res_norm_mean,
                    r.sf_to_pg,
                    r.res_norm_mean_liquid,
                    r.res_norm_mean_gas,
                    r.sf_norm_mean_liquid,
                    r.sf_norm_mean_gas,
                    r.pg_norm_mean_liquid,
                    r.pg_norm_mean_gas,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config json")
    p.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint .npz files")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["constant_liquid", "harmonic", "arithmetic"],
        choices=["harmonic", "arithmetic", "constant_liquid"],
    )
    p.add_argument(
        "--out-csv",
        default="/Users/burovnikita/Desktop/Study/PhD/droplet/droplet_spreading_modeling/diagnostics/balanced_force_ab/results_faces_ab.csv",
    )
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    rows: List[CaseResult] = []
    for ck in args.checkpoints:
        if not os.path.exists(ck):
            print(f"skip missing checkpoint: {ck}")
            continue
        for mode in args.modes:
            pair = _evaluate_checkpoint(ck, cfg, mode)
            rows.extend(pair)
            for r in pair:
                print(
                    f"{r.checkpoint:>20} | {mode:>15} | {r.route:>13} | "
                    f"sf/pg={r.sf_to_pg:.4f} | "
                    f"res(liq/gas)=({r.res_norm_mean_liquid:.3e}/{r.res_norm_mean_gas:.3e})"
                )

    _write_csv(args.out_csv, rows)
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()

