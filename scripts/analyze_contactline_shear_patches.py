#!/usr/bin/env python3
"""
Analyze contact-line patches for large tangential shear on staggered grid.

Reports, near bottom/contact-line region:
  - large du = u[i+1,0]-u[i,0]
  - large bottom-band divergence
  - term decomposition in u predictor at j=0,1:
      dudt = -Au + nu*Lu + fx_u + ax_u
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.state import SimulationState
from numerics.staggered_mac import advect_u, laplacian_u, divergence as mac_divergence
from numerics.finite_differences import jax_gradient
from physics.properties import jax_calculate_density


def _contact_span(phi: np.ndarray, pad: int) -> Tuple[int, int]:
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return (0, phi.shape[0] - 1)
    return max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad)


def _topk_indices(arr: np.ndarray, k: int) -> List[int]:
    k = max(1, min(k, arr.size))
    return list(np.argsort(np.abs(arr))[::-1][:k])


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze contact-line shear patches")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--contact_pad", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = json.load(f)
    st = SimulationState.from_config(cfg, restart_from=args.checkpoint)

    u = np.array(st.u_face)
    v = np.array(st.v_face)
    phi = np.array(st.phi)
    P = jnp.array(st.P)
    dx, dy = float(st.dx), float(st.dy)
    nu = 1.0 / max(float(st.Re2), 1e-6)

    # Predictor term decomposition
    Au = np.array(advect_u(jnp.array(u), jnp.array(v), dx, dy))
    Lu = np.array(laplacian_u(jnp.array(u), dx, dy))
    sf = np.array(st.compute_surface_tension())
    fx = sf[..., 0]
    Nx, Ny = fx.shape
    fx_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
    fx_u[1:Nx, :] = 0.5 * (fx[1:, :] + fx[:-1, :])

    rho = np.array(jax_calculate_density(jnp.array(phi), st.rho1, st.rho2))
    gradP = np.array(jax_gradient(P, dx, dy, st.geometry.f_1_grid))
    ax = -gradP[..., 0] / np.maximum(rho, 1e-6)
    ax_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
    ax_u[1:Nx, :] = 0.5 * (ax[1:, :] + ax[:-1, :])

    dudt = -Au + nu * Lu + fx_u + ax_u

    div = np.array(mac_divergence(jnp.array(u), jnp.array(v), dx, dy))
    ad = np.abs(div)

    i_lo, i_hi = _contact_span(phi, pad=args.contact_pad)
    contact_range = range(i_lo, i_hi + 1)

    # Candidate sets: strongest du on bottom in contact span, strongest row1 divergence in contact span
    du_bottom = u[1:, 0] - u[:-1, 0]  # indexed by i=0..Nx-1
    du_contact = np.array([du_bottom[i] for i in contact_range], dtype=np.float64)
    top_du_local = _topk_indices(du_contact, args.topk)
    top_du_i = [i_lo + ii for ii in top_du_local]

    row1_contact = ad[i_lo:i_hi + 1, 1]
    top_row1_local = _topk_indices(row1_contact, args.topk)
    top_row1_i = [i_lo + ii for ii in top_row1_local]

    rows = []
    for src, inds in (("du_bottom", top_du_i), ("row1_div", top_row1_i)):
        for i in inds:
            if not (0 <= i <= Nx - 1):
                continue
            for j in (0, 1):
                if not (0 <= j < Ny):
                    continue
                u_w = float(u[i, j])
                u_e = float(u[i + 1, j])
                du = u_e - u_w
                v_s = float(v[i, j])
                v_n = float(v[i, j + 1])
                dv = v_n - v_s
                div_ij = float(div[i, j]) if j < div.shape[1] else np.nan
                rows.append(
                    {
                        "source": src,
                        "i": int(i),
                        "j": int(j),
                        "phi_0": float(phi[i, 0]),
                        "phi_1": float(phi[i, 1]),
                        "rho_0": float(rho[i, 0]),
                        "rho_1": float(rho[i, 1]),
                        "u_w": u_w,
                        "u_e": u_e,
                        "du": float(du),
                        "v_s": v_s,
                        "v_n": v_n,
                        "dv": float(dv),
                        "div": div_ij,
                        "minus_Au": float(-Au[i, j]),
                        "nu_Lu": float(nu * Lu[i, j]),
                        "fx_u": float(fx_u[i, j]),
                        "ax_u": float(ax_u[i, j]),
                        "dudt_sum": float(dudt[i, j]),
                    }
                )

    # Global summary
    hot_i, hot_j = np.unravel_index(np.argmax(ad), ad.shape)
    summary = {
        "checkpoint": args.checkpoint,
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "nu": nu,
        "contact_i_lo": int(i_lo),
        "contact_i_hi": int(i_hi),
        "max_div": float(np.max(ad)),
        "hot_i": int(hot_i),
        "hot_j": int(hot_j),
        "row1_max": float(np.max(ad[:, 1])),
        "contact_row1_max": float(np.max(ad[i_lo:i_hi + 1, 1])),
        "max_abs_du_bottom_contact": float(np.max(np.abs(du_contact))) if du_contact.size else 0.0,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, "patch_terms.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("Saved:", os.path.join(args.output_dir, "summary.json"))
    print("Saved:", csv_path)
    print(
        "Summary:",
        f"max_div={summary['max_div']:.6g}",
        f"row1_max={summary['row1_max']:.6g}",
        f"contact_row1_max={summary['contact_row1_max']:.6g}",
        f"max|du_bottom_contact|={summary['max_abs_du_bottom_contact']:.6g}",
        f"hot=({summary['hot_i']},{summary['hot_j']})",
    )


if __name__ == "__main__":
    main()

