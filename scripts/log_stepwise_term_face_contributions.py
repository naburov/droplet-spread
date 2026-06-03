#!/usr/bin/env python3
"""
Run restart for N steps and log:
  1) Predictor-term decomposition near hotspot/contact band.
  2) Face-by-face divergence contributions for hotspot cells.

Outputs:
  - per_step_global.csv
  - per_step_hotspot_faces.csv
  - per_step_u_terms.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.two_phase import TwoPhaseSimulation
from numerics.staggered_mac import advect_u, laplacian_u, divergence as mac_divergence
from numerics.finite_differences import jax_gradient
from physics.properties import jax_calculate_density


def _contact_span(phi: np.ndarray, pad: int = 2) -> Optional[Tuple[int, int]]:
    phi0 = phi[:, 0]
    phi1 = phi[:, 1]
    mask = ((phi0 * phi1) < 0.0) | (np.abs(phi0) < 0.5)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return max(0, int(idx.min()) - pad), min(phi.shape[0] - 1, int(idx.max()) + pad)


def _face_contrib_row(u: np.ndarray, v: np.ndarray, i: int, j: int, dx: float, dy: float) -> Dict[str, float]:
    u_w = float(u[i, j])
    u_e = float(u[i + 1, j])
    v_s = float(v[i, j])
    v_n = float(v[i, j + 1])

    c_u_e = u_e / dx
    c_u_w = -u_w / dx
    c_v_n = v_n / dy
    c_v_s = -v_s / dy
    div = c_u_e + c_u_w + c_v_n + c_v_s

    abs_map = {
        "u_e": abs(c_u_e),
        "u_w": abs(c_u_w),
        "v_n": abs(c_v_n),
        "v_s": abs(c_v_s),
    }
    dominant_face = max(abs_map, key=abs_map.get)

    return {
        "u_w": u_w,
        "u_e": u_e,
        "v_s": v_s,
        "v_n": v_n,
        "c_u_e": c_u_e,
        "c_u_w": c_u_w,
        "c_v_n": c_v_n,
        "c_v_s": c_v_s,
        "div_reconstructed": div,
        "dominant_face": dominant_face,
        "dominant_abs_contrib": abs_map[dominant_face],
    }


def _u_term_row(
    i_face: int,
    j: int,
    Au: np.ndarray,
    Lu: np.ndarray,
    fx_u: np.ndarray,
    ax_u: np.ndarray,
    nu: float,
) -> Dict[str, float]:
    minus_Au = float(-Au[i_face, j])
    nu_Lu = float(nu * Lu[i_face, j])
    st = float(fx_u[i_face, j])
    pg = float(ax_u[i_face, j])
    dudt = minus_Au + nu_Lu + st + pg

    abs_map = {
        "minus_Au": abs(minus_Au),
        "nu_Lu": abs(nu_Lu),
        "surface_tension": abs(st),
        "pressure_grad": abs(pg),
    }
    dominant_term = max(abs_map, key=abs_map.get)

    return {
        "minus_Au": minus_Au,
        "nu_Lu": nu_Lu,
        "surface_tension": st,
        "pressure_grad": pg,
        "dudt_sum": dudt,
        "dominant_term": dominant_term,
        "dominant_abs_term": abs_map[dominant_term],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Log stepwise term and face contributions")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--contact_pad", type=int, default=2)
    p.add_argument("--stop_max_div", type=float, default=5000.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    global_csv = os.path.join(args.output_dir, "per_step_global.csv")
    faces_csv = os.path.join(args.output_dir, "per_step_hotspot_faces.csv")
    terms_csv = os.path.join(args.output_dir, "per_step_u_terms.csv")

    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("restart", {})["restart_from"] = args.checkpoint

    sim = TwoPhaseSimulation(cfg, output_dir=args.output_dir)
    nu = 1.0 / max(float(sim.state.Re2), 1e-6)

    rows_global = []
    rows_faces = []
    rows_terms = []

    for n in range(int(args.steps)):
        sim.step()

        u = np.array(sim.state.u_face)
        v = np.array(sim.state.v_face)
        phi = np.array(sim.state.phi)
        P = jnp.array(sim.state.P)
        dx, dy = float(sim.state.dx), float(sim.state.dy)

        # Divergence and hotspots
        div = np.array(mac_divergence(jnp.array(u), jnp.array(v), dx, dy))
        ad = np.abs(div)
        hot_i, hot_j = np.unravel_index(np.argmax(ad), ad.shape)

        contact_span = _contact_span(phi, pad=args.contact_pad)
        contact_hot_i, contact_hot_j = int(hot_i), int(hot_j)
        contact_row1_max = np.nan
        if contact_span is not None:
            lo, hi = contact_span
            row = ad[lo : hi + 1, 1]
            k = int(np.argmax(row))
            contact_hot_i = lo + k
            contact_hot_j = 1
            contact_row1_max = float(np.max(row))

        rows_global.append(
            {
                "n": n,
                "step": int(sim.state.step),
                "time": float(sim.state.t),
                "dt": float(sim.state.dt),
                "max_div": float(np.max(ad)),
                "max_div_interior": float(np.max(ad[1:-1, 1:-1])),
                "row1_max": float(np.max(ad[:, 1])),
                "contact_row1_max": contact_row1_max,
                "hot_i": int(hot_i),
                "hot_j": int(hot_j),
                "contact_hot_i": int(contact_hot_i),
                "contact_hot_j": int(contact_hot_j),
            }
        )

        # Face contributions for both global hotspot and contact-row1 hotspot
        targets = [("global_hotspot", int(hot_i), int(hot_j))]
        if contact_span is not None:
            targets.append(("contact_row1_hotspot", int(contact_hot_i), int(contact_hot_j)))

        for tag, i, j in targets:
            if not (0 <= i < div.shape[0] and 0 <= j < div.shape[1]):
                continue
            face = _face_contrib_row(u, v, i, j, dx, dy)
            face_row = {
                "n": n,
                "step": int(sim.state.step),
                "target": tag,
                "i": i,
                "j": j,
                "div_from_field": float(div[i, j]),
                "abs_div_from_field": float(ad[i, j]),
                **face,
            }
            rows_faces.append(face_row)

        # U-equation term decomposition at west/east faces of target cell
        Au = np.array(advect_u(jnp.array(u), jnp.array(v), dx, dy))
        Lu = np.array(laplacian_u(jnp.array(u), dx, dy))
        sf = np.array(sim.state.compute_surface_tension())
        fx = sf[..., 0]
        Nx, Ny = fx.shape
        fx_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
        fx_u[1:Nx, :] = 0.5 * (fx[1:, :] + fx[:-1, :])

        rho = np.array(jax_calculate_density(jnp.array(phi), sim.state.rho1, sim.state.rho2))
        gradP = np.array(jax_gradient(P, dx, dy, sim.state.geometry.f_1_grid))
        ax = -gradP[..., 0] / np.maximum(rho, 1e-6)
        ax_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
        ax_u[1:Nx, :] = 0.5 * (ax[1:, :] + ax[:-1, :])

        for tag, i, j in targets:
            for side, i_face in (("west_u_face", i), ("east_u_face", i + 1)):
                if not (0 <= i_face < Au.shape[0] and 0 <= j < Au.shape[1]):
                    continue
                trow = _u_term_row(i_face, j, Au, Lu, fx_u, ax_u, nu)
                rows_terms.append(
                    {
                        "n": n,
                        "step": int(sim.state.step),
                        "target": tag,
                        "cell_i": i,
                        "cell_j": j,
                        "u_face_side": side,
                        "u_face_i": int(i_face),
                        "u_face_j": int(j),
                        "rho_cell": float(rho[min(i, rho.shape[0] - 1), min(j, rho.shape[1] - 1)]),
                        "phi_cell": float(phi[min(i, phi.shape[0] - 1), min(j, phi.shape[1] - 1)]),
                        **trow,
                    }
                )

        print(
            f"n={n:04d} step={int(sim.state.step)} "
            f"dt={float(sim.state.dt):.3e} max_div={float(np.max(ad)):.3e} "
            f"row1={float(np.max(ad[:, 1])):.3e} hot=({int(hot_i)},{int(hot_j)})"
        )

        if float(np.max(ad)) >= float(args.stop_max_div):
            print(f"Stopping early: max_div reached {float(np.max(ad)):.3e}")
            break

        # Manual stepping outside run(): keep step counter progressing.
        sim.state.step += 1

    with open(global_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_global[0].keys()) if rows_global else [])
        w.writeheader()
        w.writerows(rows_global)

    with open(faces_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_faces[0].keys()) if rows_faces else [])
        w.writeheader()
        w.writerows(rows_faces)

    with open(terms_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_terms[0].keys()) if rows_terms else [])
        w.writeheader()
        w.writerows(rows_terms)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "steps_requested": int(args.steps),
        "steps_done": len(rows_global),
        "global_csv": global_csv,
        "faces_csv": faces_csv,
        "terms_csv": terms_csv,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", global_csv)
    print("Saved:", faces_csv)
    print("Saved:", terms_csv)
    print("Saved:", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()

