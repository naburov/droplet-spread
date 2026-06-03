#!/usr/bin/env python3
"""
Checkpoint-level PPE decomposition focused on checkerboard amplification.

Given one simulation checkpoint step (and corresponding ppe_diagnostics pair),
this script reconstructs the variable-density PPE correction path:

    rhs = (1/dt) * div(u*)
    div((1/rho) grad p') = rhs
    u_corr = -dt * (1/rho_face) * grad(p')

and compares observed PPE increment (U_after - U_before) against the reconstructed
increment, including checkerboard/parity metrics in interface and bottom rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from numerics.staggered_mac import divergence as mac_divergence, grad_p_to_faces
from numerics.staggered_utils import to_collocated, to_staggered
from physics.properties import jax_calculate_density
from solvers.ppe import (
    _apply_pressure_compatibility_from_dirichlet_velocity,
    _build_variable_coefficient_ppe_matrix,
    _solve_variable_coefficient_ppe,
)
from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config
from boundary_conditions.velocity_bc import VelocityBoundaryConditions


def _checkerboard_weight(shape: Tuple[int, int]) -> np.ndarray:
    nx, ny = shape
    wi = np.where((np.arange(nx) % 2) == 0, 1.0, -1.0)[:, None]
    wj = np.where((np.arange(ny) % 2) == 0, 1.0, -1.0)[None, :]
    return wi * wj


def _cb_metric(field: np.ndarray, mask: np.ndarray | None = None) -> float:
    f = np.asarray(field, dtype=np.float64)
    if mask is None:
        m = np.ones_like(f, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0
    w = _checkerboard_weight(f.shape)
    den = float(np.mean(np.abs(f[m])))
    if den <= 1e-30:
        return 0.0
    num = float(np.abs(np.mean((f * w)[m])))
    return num / den


def _xpar_bottom_metric(field: np.ndarray, rows=(0, 1, 2, 3)) -> float:
    f = np.asarray(field, dtype=np.float64)
    vals = []
    for j in rows:
        if j >= f.shape[1]:
            continue
        row = f[:, j]
        alt = np.where((np.arange(row.size) % 2) == 0, 1.0, -1.0)
        den = float(np.mean(np.abs(row)))
        if den <= 1e-30:
            vals.append(0.0)
        else:
            vals.append(float(np.abs(np.mean(alt * row)) / den))
    return float(np.mean(vals)) if vals else 0.0


def _plot(ax, arr: np.ndarray, title: str, signed: bool = True, q: float = 99.0) -> None:
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if np.any(finite):
        if signed:
            vmax = float(np.percentile(np.abs(a[finite]), q))
            vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0.0) else vmax
            im = ax.imshow(a.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            vmax = float(np.percentile(a[finite], q))
            vmax = 1.0 if (not np.isfinite(vmax) or vmax <= 0.0) else vmax
            im = ax.imshow(a.T, origin="lower", cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(np.zeros_like(a).T, origin="lower", cmap="magma", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    den = float(np.sqrt(np.mean(x * x)) * np.sqrt(np.mean(y * y)))
    if den <= 1e-30:
        return 0.0
    return float(np.mean(x * y) / den)


class _CompatVelocityBC:
    """Minimal adapter to reuse PPE compatibility helper in diagnostics."""

    def __init__(self, config: Dict):
        self.config = config

    def get_inlet_profile(self, ny: int, dy: float):
        return None


def _load_inputs(exp_dir: Path, step: int):
    cfg = json.loads((exp_dir / "simulation_parameters.json").read_text())
    cp = exp_dir / "checkpoints" / f"checkpoint_{step:06d}.npz"
    ppe = exp_dir / "ppe_diagnostics" / f"ppe_data_step{step:06d}_after_ppe.npz"
    if not cp.exists():
        raise FileNotFoundError(f"checkpoint not found: {cp}")
    if not ppe.exists():
        raise FileNotFoundError(f"ppe diagnostics pair not found: {ppe}")
    zc = np.load(cp)
    zp = np.load(ppe)
    phi = np.array(zc["phi"], dtype=np.float64)
    U_before = np.array(zp["U_before"], dtype=np.float64)
    U_after = np.array(zp["U_after"], dtype=np.float64)
    return cfg, phi, U_before, U_after


def _build_inv_rho_faces(phi: np.ndarray, rho1: float, rho2: float):
    rho = np.array(jax_calculate_density(jnp.array(phi), rho1, rho2), dtype=np.float64)
    rho = np.maximum(rho, 1e-6)
    inv_rho = 1.0 / rho
    nx, ny = rho.shape
    inv_u = np.zeros((nx + 1, ny), dtype=np.float64)
    inv_v = np.zeros((nx, ny + 1), dtype=np.float64)
    inv_u[1:nx, :] = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])
    inv_u[0, :] = inv_rho[0, :]
    inv_u[nx, :] = inv_rho[-1, :]
    inv_v[:, 1:ny] = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    inv_v[:, 0] = inv_rho[:, 0]
    inv_v[:, ny] = inv_rho[:, -1]
    return rho, inv_u, inv_v


def _ppe_bcs_from_cfg(cfg: Dict) -> Dict[str, str]:
    ppe_cfg = cfg.get("solver_params", {}).get("ppe", {})
    explicit = ppe_cfg.get("boundary_conditions")
    if explicit is not None:
        return explicit
    return derive_ppe_bcs_from_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--interface-band", type=float, default=0.75)
    ap.add_argument("--out-dir", default="diagnostics/ppe_checkpoint_decomposition")
    args = ap.parse_args()

    exp = Path(args.experiment_dir)
    cfg, phi, U_before, U_after = _load_inputs(exp, args.step)
    nx, ny = phi.shape
    dt = float(cfg["time_params"]["dt"])
    dx = float(cfg["grid_params"]["Lx"]) / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])
    rho1 = float(cfg["physical_params"]["rho1"])
    rho2 = float(cfg["physical_params"]["rho2"])

    rho, inv_rho_u, inv_rho_v = _build_inv_rho_faces(phi, rho1, rho2)
    ppe_bcs = _ppe_bcs_from_cfg(cfg)
    A, all_neumann, gauge_ij = _build_variable_coefficient_ppe_matrix(inv_rho_u, inv_rho_v, dx, dy, ppe_bcs)

    u_face_star, v_face_star = to_staggered(jnp.array(U_before))
    rhs = (1.0 / dt) * np.array(mac_divergence(u_face_star, v_face_star, dx, dy), dtype=np.float64)
    rhs_eff = rhs.copy()
    if all_neumann:
        rhs_eff -= np.mean(rhs_eff)
    if ppe_bcs.get("left") == "dirichlet":
        rhs_eff[0, :] = 0.0
    if ppe_bcs.get("right") == "dirichlet":
        rhs_eff[-1, :] = 0.0
    if ppe_bcs.get("bottom") == "dirichlet":
        rhs_eff[:, 0] = 0.0
    if ppe_bcs.get("top") == "dirichlet":
        rhs_eff[:, -1] = 0.0
    if all_neumann and gauge_ij is not None:
        gi, gj = gauge_ij
        rhs_eff[gi, gj] = 0.0

    p_corr_raw = _solve_variable_coefficient_ppe(A, rhs_eff, tol=1e-8, maxiter=2000)
    p_corr = _apply_pressure_compatibility_from_dirichlet_velocity(
        p_corr_raw,
        np.array(u_face_star),
        np.array(v_face_star),
        _CompatVelocityBC(cfg),
        dx,
        dy,
        dt,
        inv_rho_u_face=inv_rho_u,
        inv_rho_v_face=inv_rho_v,
        ppe_bcs=ppe_bcs,
    )

    dpdx_face, dpdy_face = grad_p_to_faces(jnp.array(p_corr), dx, dy)
    du_face = -dt * inv_rho_u * np.array(dpdx_face, dtype=np.float64)
    dv_face = -dt * inv_rho_v * np.array(dpdy_face, dtype=np.float64)
    dU_pred = np.array(to_collocated(jnp.array(du_face), jnp.array(dv_face)), dtype=np.float64)
    u_face_prebc = np.array(u_face_star) + du_face
    v_face_prebc = np.array(v_face_star) + dv_face
    vel_bc = VelocityBoundaryConditions(cfg)
    u_face_postbc, v_face_postbc = vel_bc.apply_to_faces(
        jnp.array(u_face_prebc),
        jnp.array(v_face_prebc),
        dx,
        dy,
        phi=jnp.array(phi),
        geometry=None,
    )
    u_face_postbc = np.array(u_face_postbc, dtype=np.float64)
    v_face_postbc = np.array(v_face_postbc, dtype=np.float64)

    dU_obs = U_after - U_before
    dU_res = dU_obs - dU_pred

    interface_mask = np.abs(phi) < float(args.interface_band)
    jmid0 = max(0, ny // 2 - 2)
    jmid1 = min(ny, ny // 2 + 3)
    js_bottom = slice(0, min(4, ny))
    js_middle = slice(jmid0, jmid1)

    def _corr_region(a: np.ndarray, b: np.ndarray, js: slice) -> float:
        return _corr(a[:, js], b[:, js])

    summary = {
        "experiment": exp.name,
        "step": int(args.step),
        "grid": {"Nx": int(nx), "Ny": int(ny), "dx": dx, "dy": dy, "dt": dt},
        "ppe_bcs": ppe_bcs,
        "rhs_l2": float(np.sqrt(np.mean(rhs_eff**2))),
        "rhs_cb_interface": _cb_metric(rhs_eff, interface_mask),
        "p_corr_cb_interface": _cb_metric(p_corr, interface_mask),
        "du_obs_cb_interface": _cb_metric(dU_obs[..., 0], interface_mask),
        "du_pred_cb_interface": _cb_metric(dU_pred[..., 0], interface_mask),
        "du_res_cb_interface": _cb_metric(dU_res[..., 0], interface_mask),
        "dv_obs_cb_interface": _cb_metric(dU_obs[..., 1], interface_mask),
        "dv_pred_cb_interface": _cb_metric(dU_pred[..., 1], interface_mask),
        "dv_res_cb_interface": _cb_metric(dU_res[..., 1], interface_mask),
        "dv_obs_xpar_bottom_j0_3": _xpar_bottom_metric(dU_obs[..., 1]),
        "dv_pred_xpar_bottom_j0_3": _xpar_bottom_metric(dU_pred[..., 1]),
        "dv_res_xpar_bottom_j0_3": _xpar_bottom_metric(dU_res[..., 1]),
        "du_obs_l2": float(np.sqrt(np.mean(dU_obs[..., 0] ** 2))),
        "du_pred_l2": float(np.sqrt(np.mean(dU_pred[..., 0] ** 2))),
        "du_res_l2": float(np.sqrt(np.mean(dU_res[..., 0] ** 2))),
        "dv_obs_l2": float(np.sqrt(np.mean(dU_obs[..., 1] ** 2))),
        "dv_pred_l2": float(np.sqrt(np.mean(dU_pred[..., 1] ** 2))),
        "dv_res_l2": float(np.sqrt(np.mean(dU_res[..., 1] ** 2))),
        "v_cc_bottom_before_mean": float(np.mean(U_before[:, 0, 1])),
        "v_cc_bottom_after_mean": float(np.mean(U_after[:, 0, 1])),
        "v_face_bottom_star_mean": float(np.mean(np.array(v_face_star)[:, 0])),
        "v_face_bottom_after_prebc_mean": float(np.mean(v_face_prebc[:, 0])),
        "v_face_bottom_after_postbc_mean": float(np.mean(v_face_postbc[:, 0])),
        "u_face_bottom_star_mean": float(np.mean(np.array(u_face_star)[:, 0])),
        "u_face_bottom_after_prebc_mean": float(np.mean(u_face_prebc[:, 0])),
        "u_face_bottom_after_postbc_mean": float(np.mean(u_face_postbc[:, 0])),
        "u_face_bottom_corr_star_vs_prebc": _corr(np.array(u_face_star)[:, 0], u_face_prebc[:, 0]),
        "u_face_bottom_corr_star_vs_postbc": _corr(np.array(u_face_star)[:, 0], u_face_postbc[:, 0]),
        "v_face_bottom_corr_star_vs_prebc": _corr(np.array(v_face_star)[:, 0], v_face_prebc[:, 0]),
        "v_face_bottom_corr_star_vs_postbc": _corr(np.array(v_face_star)[:, 0], v_face_postbc[:, 0]),
        "corr_before_after_u_bottom_j0_3": _corr_region(U_before[..., 0], U_after[..., 0], js_bottom),
        "corr_before_after_v_bottom_j0_3": _corr_region(U_before[..., 1], U_after[..., 1], js_bottom),
        "corr_before_after_u_middle_jmidpm2": _corr_region(U_before[..., 0], U_after[..., 0], js_middle),
        "corr_before_after_v_middle_jmidpm2": _corr_region(U_before[..., 1], U_after[..., 1], js_middle),
        "corr_before_dUpred_u_bottom_j0_3": _corr_region(U_before[..., 0], dU_pred[..., 0], js_bottom),
        "corr_before_dUpred_v_bottom_j0_3": _corr_region(U_before[..., 1], dU_pred[..., 1], js_bottom),
        "corr_before_dUpred_u_middle_jmidpm2": _corr_region(U_before[..., 0], dU_pred[..., 0], js_middle),
        "corr_before_dUpred_v_middle_jmidpm2": _corr_region(U_before[..., 1], dU_pred[..., 1], js_middle),
    }

    out = Path(args.out_dir) / exp.name / f"step_{args.step:06d}"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    _plot(ax[0, 0], rhs, "rhs = div(u*)/dt", signed=True)
    _plot(ax[0, 1], p_corr, "p_corr (compat applied)", signed=True)
    _plot(ax[0, 2], du_face[1:, :] - du_face[:-1, :], "du_face jump-x", signed=True)
    _plot(ax[0, 3], dv_face[:, 1:] - dv_face[:, :-1], "dv_face jump-y", signed=True)
    _plot(ax[1, 0], dU_obs[..., 0], "du_obs (after-before)", signed=True)
    _plot(ax[1, 1], dU_pred[..., 0], "du_pred from p_corr", signed=True)
    _plot(ax[1, 2], dU_res[..., 0], "du_res = obs-pred", signed=True)
    _plot(ax[1, 3], np.abs(dU_res[..., 0]), "|du_res|", signed=False)
    _plot(ax[2, 0], dU_obs[..., 1], "dv_obs (after-before)", signed=True)
    _plot(ax[2, 1], dU_pred[..., 1], "dv_pred from p_corr", signed=True)
    _plot(ax[2, 2], dU_res[..., 1], "dv_res = obs-pred", signed=True)
    _plot(ax[2, 3], np.abs(dU_res[..., 1]), "|dv_res|", signed=False)
    fig.suptitle(f"PPE checkpoint decomposition: {exp.name} step {args.step}")
    fig.savefig(out / "ppe_decomposition_maps.png", dpi=180)
    plt.close(fig)

    # Bottom profiles: collocated row vs wall faces.
    x_cc = np.arange(nx)
    x_uf = np.arange(nx + 1)
    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax2[0, 0].plot(x_cc, U_before[:, 0, 0], "b-", label="u_cc j=0 before")
    ax2[0, 0].plot(x_cc, U_after[:, 0, 0], "r-", label="u_cc j=0 after")
    ax2[0, 0].set_title("Collocated u at bottom row")
    ax2[0, 0].legend()
    ax2[0, 0].grid(alpha=0.3)

    ax2[0, 1].plot(x_uf, np.array(u_face_star)[:, 0], "b-", label="u_face wall before")
    ax2[0, 1].plot(x_uf, u_face_prebc[:, 0], "r-", label="u_face after (pre-BC)")
    ax2[0, 1].plot(x_uf, u_face_postbc[:, 0], "k--", label="u_face after (post-BC)")
    ax2[0, 1].set_title("Face u at bottom wall")
    ax2[0, 1].legend()
    ax2[0, 1].grid(alpha=0.3)

    ax2[1, 0].plot(x_cc, U_before[:, 0, 1], "b-", label="v_cc j=0 before")
    ax2[1, 0].plot(x_cc, U_after[:, 0, 1], "r-", label="v_cc j=0 after")
    ax2[1, 0].set_title("Collocated v at bottom row")
    ax2[1, 0].legend()
    ax2[1, 0].grid(alpha=0.3)

    ax2[1, 1].plot(x_cc, np.array(v_face_star)[:, 0], "b-", label="v_face wall before")
    ax2[1, 1].plot(x_cc, v_face_prebc[:, 0], "r-", label="v_face after (pre-BC)")
    ax2[1, 1].plot(x_cc, v_face_postbc[:, 0], "k--", label="v_face after (post-BC)")
    ax2[1, 1].set_title("Face v at bottom wall")
    ax2[1, 1].legend()
    ax2[1, 1].grid(alpha=0.3)
    fig2.suptitle("Bottom-wall diagnostics: collocated vs face variables")
    fig2.savefig(out / "ppe_bottom_profiles_cc_vs_face.png", dpi=180)
    plt.close(fig2)

    np.savez_compressed(
        out / "ppe_decomposition_fields.npz",
        phi=phi,
        rho=rho,
        U_before=U_before,
        U_after=U_after,
        u_face_star=np.array(u_face_star),
        v_face_star=np.array(v_face_star),
        u_face_prebc=u_face_prebc,
        v_face_prebc=v_face_prebc,
        u_face_postbc=u_face_postbc,
        v_face_postbc=v_face_postbc,
        rhs=rhs,
        rhs_eff=rhs_eff,
        p_corr_raw=p_corr_raw,
        p_corr=p_corr,
        du_face=du_face,
        dv_face=dv_face,
        dU_obs=dU_obs,
        dU_pred=dU_pred,
        dU_res=dU_res,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

