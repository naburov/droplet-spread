#!/usr/bin/env python3
"""
Checkpoint-based A/B for capillary-pressure balance.

Goal:
  Compare how different Weber interpolation choices affect local capillary
  force balance near contact lines on frozen checkpoints.

What is measured (for each checkpoint and mode):
  - a_sf = -F_sigma / rho  (cell-centered)
  - P_dyn from variable-coefficient pressure solve:
      div((1/rho) grad(P_dyn)) = div(F_sigma)
  - a_pg_cc   = -(1/rho) * grad_cc(P_dyn)
  - a_pg_face = face-consistent pressure acceleration mapped back to cells
  - residuals:
      r_cc   = a_sf + a_pg_cc
      r_face = a_sf + a_pg_face

The script reports region/phase-resolved residual stats in the CL strip.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import jax.numpy as jnp

from numerics.finite_differences import jax_divergence, jax_gradient, jax_norm
from numerics.staggered_mac import grad_p_to_faces
from physics.properties import jax_calculate_density
from physics.surface_tension import jax_curvature, jax_curvature_smooth
from physics.pressure import (
    _build_variable_coefficient_pressure_matrix,
    _solve_variable_coefficient_pressure,
)


@dataclass
class CaseResult:
    checkpoint: str
    mode: str
    sf_norm_mean: float
    pg_norm_mean_cc: float
    pg_norm_mean_face: float
    res_norm_mean_cc: float
    res_norm_mean_face: float
    sf_to_pg_cc: float
    sf_to_pg_face: float
    res_norm_mean_face_liquid: float
    res_norm_mean_face_gas: float
    sf_norm_mean_liquid: float
    sf_norm_mean_gas: float
    pg_norm_mean_face_liquid: float
    pg_norm_mean_face_gas: float


def _we_map(phi: np.ndarray, we1: float, we2: float, mode: str) -> np.ndarray:
    c = 0.5 * (phi + 1.0)
    if mode == "harmonic":
        return 1.0 / (((1.0 - c) / we2) + (c / we1))
    if mode == "arithmetic":
        return (1.0 - c) * we2 + c * we1
    if mode == "constant_liquid":
        return np.full_like(phi, float(we2))
    raise ValueError(f"Unknown mode: {mode}")


def _surface_tension_force(
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
    sf = (3.0 * np.sqrt(2.0) * float(epsilon) / 4.0) * np.array(kappa2) * np.array(normg2) * np.array(grad)
    sf = sf / np.maximum(we2map, 1e-12)
    return sf


def _pressure_bcs_from_config(cfg: Dict) -> Dict[str, str]:
    pbc = cfg.get("boundary_conditions", {}).get("pressure", {})
    out = {}
    for b in ("left", "right", "bottom", "top"):
        raw = str(pbc.get(b, "neumann")).lower()
        out[b] = "dirichlet" if raw in ("open", "dirichlet") else "neumann"
    return out


def _solve_pdyn_from_sf(sf: np.ndarray, rho: np.ndarray, dx: float, dy: float, bcs: Dict[str, str]) -> np.ndarray:
    inv_rho = 1.0 / np.maximum(rho, 1e-6)
    nx, ny = rho.shape
    inv_u = np.zeros((nx + 1, ny), dtype=np.float64)
    inv_v = np.zeros((nx, ny + 1), dtype=np.float64)
    inv_u[1:nx, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
    inv_u[0, :] = inv_rho[0, :]
    inv_u[nx, :] = inv_rho[-1, :]
    inv_v[:, 1:ny] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    inv_v[:, 0] = inv_rho[:, 0]
    inv_v[:, ny] = inv_rho[:, -1]

    A, all_neu, gauge_ij = _build_variable_coefficient_pressure_matrix(inv_u, inv_v, dx, dy, bcs)
    rhs = np.array(jax_divergence(jnp.array(sf), dx, dy, jnp.zeros_like(jnp.array(rho))), dtype=np.float64)
    if all_neu and gauge_ij is not None:
        gi, gj = gauge_ij
        rhs[gi, gj] = 0.0
    return np.array(_solve_variable_coefficient_pressure(A, rhs, tol=1e-10, maxiter=3000), dtype=np.float64)


def _accelerations(sf: np.ndarray, p: np.ndarray, rho: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_rho = 1.0 / np.maximum(rho, 1e-12)
    grad_cc = np.array(jax_gradient(jnp.array(p), dx, dy, jnp.zeros_like(jnp.array(rho))))
    a_pg_cc = -grad_cc * inv_rho[..., None]

    # Face-consistent pressure acceleration then mapped to cell centers.
    nx, ny = rho.shape
    inv_u = np.zeros((nx + 1, ny), dtype=np.float64)
    inv_v = np.zeros((nx, ny + 1), dtype=np.float64)
    inv_u[1:nx, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
    inv_u[0, :] = inv_rho[0, :]
    inv_u[nx, :] = inv_rho[-1, :]
    inv_v[:, 1:ny] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    inv_v[:, 0] = inv_rho[:, 0]
    inv_v[:, ny] = inv_rho[:, -1]
    dpdx_u, dpdy_v = grad_p_to_faces(jnp.array(p), dx, dy)
    ax_u = -inv_u * np.array(dpdx_u)
    ay_v = -inv_v * np.array(dpdy_v)
    a_pg_face = np.zeros((nx, ny, 2), dtype=np.float64)
    a_pg_face[..., 0] = 0.5 * (ax_u[1:, :] + ax_u[:-1, :])
    a_pg_face[..., 1] = 0.5 * (ay_v[:, 1:] + ay_v[:, :-1])

    a_sf = -sf * inv_rho[..., None]
    return a_sf, a_pg_cc, a_pg_face


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


def _evaluate_checkpoint(ckpt: str, cfg: Dict, mode: str) -> CaseResult:
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

    sf = _surface_tension_force(phi, dx, dy, eps, we1, we2, mode)
    p_dyn = _solve_pdyn_from_sf(sf, rho, dx, dy, _pressure_bcs_from_config(cfg))
    a_sf, a_pg_cc, a_pg_face = _accelerations(sf, p_dyn, rho, dx, dy)
    r_cc = a_sf + a_pg_cc
    r_face = a_sf + a_pg_face
    m = _cl_mask(phi)
    m_liq = m & (phi < 0.0)
    m_gas = m & (phi >= 0.0)

    sf_norm = _mean_norm(a_sf, m)
    pg_cc_norm = _mean_norm(a_pg_cc, m)
    pg_face_norm = _mean_norm(a_pg_face, m)

    return CaseResult(
        checkpoint=os.path.basename(ckpt),
        mode=mode,
        sf_norm_mean=sf_norm,
        pg_norm_mean_cc=pg_cc_norm,
        pg_norm_mean_face=pg_face_norm,
        res_norm_mean_cc=_mean_norm(r_cc, m),
        res_norm_mean_face=_mean_norm(r_face, m),
        sf_to_pg_cc=float(sf_norm / max(pg_cc_norm, 1e-12)),
        sf_to_pg_face=float(sf_norm / max(pg_face_norm, 1e-12)),
        res_norm_mean_face_liquid=_mean_norm(r_face, m_liq),
        res_norm_mean_face_gas=_mean_norm(r_face, m_gas),
        sf_norm_mean_liquid=_mean_norm(a_sf, m_liq),
        sf_norm_mean_gas=_mean_norm(a_sf, m_gas),
        pg_norm_mean_face_liquid=_mean_norm(a_pg_face, m_liq),
        pg_norm_mean_face_gas=_mean_norm(a_pg_face, m_gas),
    )


def _write_csv(path: str, rows: Iterable[CaseResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "mode",
                "sf_norm_mean",
                "pg_norm_mean_cc",
                "pg_norm_mean_face",
                "res_norm_mean_cc",
                "res_norm_mean_face",
                "sf_to_pg_cc",
                "sf_to_pg_face",
                "res_norm_mean_face_liquid",
                "res_norm_mean_face_gas",
                "sf_norm_mean_liquid",
                "sf_norm_mean_gas",
                "pg_norm_mean_face_liquid",
                "pg_norm_mean_face_gas",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.checkpoint,
                    r.mode,
                    r.sf_norm_mean,
                    r.pg_norm_mean_cc,
                    r.pg_norm_mean_face,
                    r.res_norm_mean_cc,
                    r.res_norm_mean_face,
                    r.sf_to_pg_cc,
                    r.sf_to_pg_face,
                    r.res_norm_mean_face_liquid,
                    r.res_norm_mean_face_gas,
                    r.sf_norm_mean_liquid,
                    r.sf_norm_mean_gas,
                    r.pg_norm_mean_face_liquid,
                    r.pg_norm_mean_face_gas,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config json")
    p.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint .npz files")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["harmonic", "arithmetic", "constant_liquid"],
        choices=["harmonic", "arithmetic", "constant_liquid"],
    )
    p.add_argument(
        "--out-csv",
        default="/Users/burovnikita/Desktop/Study/PhD/droplet/droplet_spreading_modeling/diagnostics/balanced_force_ab/results.csv",
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
            r = _evaluate_checkpoint(ck, cfg, mode)
            rows.append(r)
            print(
                f"{r.checkpoint:>20} | {mode:>15} | "
                f"sf/pg_face={r.sf_to_pg_face:.4f} | "
                f"res_face(liq/gas)=({r.res_norm_mean_face_liquid:.3e}/{r.res_norm_mean_face_gas:.3e})"
            )

    _write_csv(args.out_csv, rows)
    print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()

