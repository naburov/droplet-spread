#!/usr/bin/env python3
"""
Decompose early grid-pattern/vortex artifacts around the interface.

For each selected step (from ppe_diagnostics):
  1) Velocity component heatmaps (u,v) before/after PPE and delta.
  2) Pattern proxies (2nd differences) for u,v and |U|.
  3) Momentum-term maps (pressure, viscous, convective, gravity, capillary).
  4) Phase-term maps (rho, Re, mu_CH, phase convective/diffusive RHS).
  5) Capillary/PPE coupling maps (rhs raw/smoothed, lhs, mismatch).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from numerics.finite_differences import (
    jax_divergence,
    jax_gradient,
    jax_laplacian,
    jax_norm,
)
from physics.phase_field import jax_willmore_chemical_potential
from boundary_conditions.chemical_potential_bc import jax_apply_chemical_potential_zero_flux_bc
from physics.pressure import compute_hydrostatic_pressure
from physics.properties import (
    jax_calculate_density,
    jax_calculate_reynolds_number,
    jax_df_2,
)
from physics.surface_tension import (
    jax_apply_surface_tension_boundary_conditions,
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


def _bbox(mask: np.ndarray, pad: int = 6) -> Tuple[slice, slice]:
    ii, jj = np.where(mask)
    nx, ny = mask.shape
    if ii.size == 0:
        return slice(0, nx), slice(0, min(ny, max(16, ny // 4)))
    i0 = max(0, int(ii.min()) - pad)
    i1 = min(nx, int(ii.max()) + pad + 1)
    j0 = max(0, int(jj.min()) - pad)
    j1 = min(ny, int(jj.max()) + pad + 1)
    return slice(i0, i1), slice(j0, j1)


def _plot(ax, f: np.ndarray, title: str, signed: bool = True, q: float = 99.0, crop: Tuple[slice, slice] | None = None):
    g = f if crop is None else f[crop[0], crop[1]]
    finite = np.isfinite(g)
    if np.any(finite):
        if signed:
            v = float(np.percentile(np.abs(g[finite]), q))
            if not np.isfinite(v) or v <= 0:
                v = 1.0
            im = ax.imshow(g.T, origin="lower", cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
        else:
            v = float(np.percentile(g[finite], q))
            if not np.isfinite(v) or v <= 0:
                v = 1.0
            im = ax.imshow(g.T, origin="lower", cmap="magma", vmin=0.0, vmax=v, aspect="auto")
    else:
        im = ax.imshow(np.zeros_like(g).T, origin="lower", cmap="magma", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _pattern2(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d2x = np.zeros_like(field)
    d2y = np.zeros_like(field)
    d2x[1:-1, :] = np.abs(field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :])
    d2y[:, 1:-1] = np.abs(field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2])
    return d2x, d2y


def _collect_steps(exp_dir: Path, selected: List[int]) -> List[int]:
    ppe_dir = exp_dir / "ppe_diagnostics"
    steps = []
    for p in sorted(ppe_dir.glob("ppe_data_step*_after_ppe.npz")):
        m = re.search(r"step(\d+)_after_ppe", p.name)
        if m:
            steps.append(int(m.group(1)))
    if selected:
        steps = [s for s in steps if s in selected]
    return sorted(set(steps))


def _load_ppe_pair(exp_dir: Path, step: int) -> Tuple[np.ndarray, np.ndarray]:
    p = exp_dir / "ppe_diagnostics" / f"ppe_data_step{step:06d}_after_ppe.npz"
    z = np.load(p)
    return np.array(z["U_before"], dtype=np.float64), np.array(z["U_after"], dtype=np.float64)


def _term_fields(cfg: Dict, phi: np.ndarray, P: np.ndarray, U_after: np.ndarray) -> Dict[str, np.ndarray]:
    nx, ny = phi.shape
    lx = float(cfg["grid_params"]["Lx"])
    ly = float(cfg["grid_params"]["Ly"])
    dx = lx / float(cfg["grid_params"]["Nx"])
    dy = ly / float(cfg["grid_params"]["Ny"])
    f1 = _f1_grid_from_config(cfg, nx, ny, dx, lx)
    f2 = np.zeros_like(f1)

    pp = cfg["physical_params"]
    st = pp.get("surface_tension", {})
    rho1 = float(pp["rho1"])
    rho2 = float(pp["rho2"])
    re1 = float(pp["Re1"])
    re2 = float(pp["Re2"])
    pe = float(pp["Pe"])
    eps = float(pp["epsilon"])
    g = float(pp["g"])
    fr = float(pp["Fr"])
    atm = float(pp["atm_pressure"])
    use_comp = bool(st.get("use_composition_field", True))
    cscale = float(st.get("composition_force_scale", 1.0))
    we_mode = str(st.get("weber_interpolation", "constant_liquid"))
    srad = int(st.get("smoothing_radius", 1))
    smooth_curv = bool(st.get("smooth_curvature", True))
    apply_over = bool(st.get("apply_boundary_overwrite", True))
    rhs_sr = int(st.get("capillary_rhs_smoothing_radius", 1))
    contact_angle = float(pp.get("contact_angle", 90.0))
    lambda_w = float(pp.get("lambda_willmore", 0.0))
    eps_w = float(pp.get("epsilon_willmore", 0.0))

    phi_j = jnp.array(phi)
    U_j = jnp.array(U_after)
    f1_j = jnp.array(f1)
    f2_j = jnp.array(f2)

    rho = np.array(jax_calculate_density(phi_j, rho1, rho2), dtype=np.float64)
    Re = np.array(jax_calculate_reynolds_number(phi_j, re1, re2), dtype=np.float64)
    inv_rho = 1.0 / np.maximum(rho, 1e-12)
    inv_rho2 = np.stack([inv_rho, inv_rho], axis=-1)

    # Pressure split
    p_h = np.array(compute_hydrostatic_pressure(jnp.array(rho), g, dy, fr, atm), dtype=np.float64)
    p_dyn = P - p_h
    grad_p = np.array(jax_gradient(jnp.array(P), dx, dy, f1_j), dtype=np.float64)
    grad_p_dyn = np.array(jax_gradient(jnp.array(p_dyn), dx, dy, f1_j), dtype=np.float64)
    grad_p_h = np.array(jax_gradient(jnp.array(p_h), dx, dy, f1_j), dtype=np.float64)
    a_pg = -grad_p * inv_rho2
    a_pg_dyn = -grad_p_dyn * inv_rho2
    a_pg_h = -grad_p_h * inv_rho2

    # Velocity terms (as in fluid dynamics)
    gradU = np.array(jax_gradient(U_j, dx, dy, f1_j), dtype=np.float64)  # (..., comp, dir)
    lap_u = np.array(jax_laplacian(U_j[..., 0], dx, dy, f1_j, f2_j), dtype=np.float64)
    lap_v = np.array(jax_laplacian(U_j[..., 1], dx, dy, f1_j, f2_j), dtype=np.float64)
    conv_x = U_after[..., 0] * gradU[..., 0, 0] + U_after[..., 1] * gradU[..., 0, 1]
    conv_y = U_after[..., 0] * gradU[..., 1, 0] + U_after[..., 1] * gradU[..., 1, 1]
    visc_x = lap_u / np.maximum(Re, 1e-12) * inv_rho
    visc_y = lap_v / np.maximum(Re, 1e-12) * inv_rho
    conv_acc_x = -conv_x
    conv_acc_y = -conv_y
    grav_y = np.full_like(phi, g / max(fr, 1e-12))

    # Capillary force and acceleration
    sf = np.array(
        jax_surface_tension_force(
            phi_j,
            float(pp["epsilon"]),
            float(pp["We1"]),
            float(pp["We2"]),
            dx,
            dy,
            f1_j,
            smooth_curvature=smooth_curv,
            smoothing_radius=srad,
            use_composition_field=use_comp,
            composition_force_scale=cscale,
            weber_interpolation=we_mode,
        ),
        dtype=np.float64,
    )
    if apply_over:
        sf = np.array(
            jax_apply_surface_tension_boundary_conditions(
                jnp.array(sf), phi_j, contact_angle=contact_angle, f_1_surface=f1_j[:, 0]
            ),
            dtype=np.float64,
        )
    a_sf = -sf * inv_rho2
    rhs_raw = np.array(jax_divergence(jnp.array(sf), dx, dy, f1_j), dtype=np.float64)
    rhs_sm = _smooth_3x3_edge(rhs_raw, rhs_sr)

    # Phase/chemical potential terms
    lap_phi = np.array(jax_laplacian(phi_j, dx, dy, f1_j, f2_j), dtype=np.float64)
    mu_ch = np.array(jax_df_2(phi_j) - eps * eps * jnp.array(lap_phi), dtype=np.float64)
    mu_ch_bc = np.array(
        jax_apply_chemical_potential_zero_flux_bc(jnp.array(mu_ch), dx, dy),
        dtype=np.float64,
    )
    if lambda_w > 0.0 and eps_w > 0.0:
        mu_w = np.array(jax_willmore_chemical_potential(phi_j, dx, dy, f1_j, f2_j, eps_w), dtype=np.float64)
        mu_w_bc = np.array(
            jax_apply_chemical_potential_zero_flux_bc(jnp.array(mu_w), dx, dy),
            dtype=np.float64,
        )
    else:
        mu_w = np.zeros_like(mu_ch)
        mu_w_bc = np.zeros_like(mu_ch)
    # Mirror runtime: BC is applied to each chemical potential contribution.
    mu_total = mu_ch_bc + lambda_w * mu_w_bc
    mu_centered = mu_total - np.mean(mu_total)
    diff_phi = (1.0 / max(pe, 1e-12)) * np.array(jax_laplacian(jnp.array(mu_centered), dx, dy, f1_j, f2_j), dtype=np.float64)
    grad_phi = np.array(jax_gradient(phi_j, dx, dy, f1_j), dtype=np.float64)
    conv_phi = U_after[..., 0] * grad_phi[..., 0] + U_after[..., 1] * grad_phi[..., 1]

    return {
        "phi": phi,
        "rho": rho,
        "Re": Re,
        "u_after": U_after[..., 0],
        "v_after": U_after[..., 1],
        "a_pg_x": a_pg[..., 0],
        "a_pg_y": a_pg[..., 1],
        "a_pg_dyn_x": a_pg_dyn[..., 0],
        "a_pg_dyn_y": a_pg_dyn[..., 1],
        "a_pg_h_x": a_pg_h[..., 0],
        "a_pg_h_y": a_pg_h[..., 1],
        "a_sf_x": a_sf[..., 0],
        "a_sf_y": a_sf[..., 1],
        "visc_x": visc_x,
        "visc_y": visc_y,
        "conv_acc_x": conv_acc_x,
        "conv_acc_y": conv_acc_y,
        "grav_y": grav_y,
        "rhs_cap_raw": rhs_raw,
        "rhs_cap_smoothed": rhs_sm,
        "mu_ch": mu_ch,
        "mu_ch_bc": mu_ch_bc,
        "mu_w": mu_w,
        "mu_w_bc": mu_w_bc,
        "mu_total": mu_total,
        "diff_phi": diff_phi,
        "conv_phi": conv_phi,
    }


def _make_figures(
    out_dir: Path,
    U_before: np.ndarray,
    U_after: np.ndarray,
    terms: Dict[str, np.ndarray],
    interface_band: float,
) -> None:
    phi = terms["phi"]
    m_if = np.abs(phi) < interface_band
    si, sj = _bbox(m_if, pad=6)
    crop = (si, sj)

    u0, v0 = U_before[..., 0], U_before[..., 1]
    u1, v1 = U_after[..., 0], U_after[..., 1]
    du, dv = u1 - u0, v1 - v0
    s0 = np.sqrt(u0 * u0 + v0 * v0)
    s1 = np.sqrt(u1 * u1 + v1 * v1)
    ds = s1 - s0

    d2x_u, d2y_u = _pattern2(u1)
    d2x_v, d2y_v = _pattern2(v1)
    d2x_s, d2y_s = _pattern2(s1)

    fig1, ax1 = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    _plot(ax1[0, 0], u0, "u before PPE", signed=True, crop=crop)
    _plot(ax1[0, 1], v0, "v before PPE", signed=True, crop=crop)
    _plot(ax1[0, 2], u1, "u after PPE", signed=True, crop=crop)
    _plot(ax1[0, 3], v1, "v after PPE", signed=True, crop=crop)
    _plot(ax1[1, 0], du, "delta u (after-before)", signed=True, crop=crop)
    _plot(ax1[1, 1], dv, "delta v (after-before)", signed=True, crop=crop)
    _plot(ax1[1, 2], s0, "|U| before", signed=False, crop=crop)
    _plot(ax1[1, 3], s1, "|U| after", signed=False, crop=crop)
    _plot(ax1[2, 0], d2x_u, "|d2x u_after|", signed=False, crop=crop)
    _plot(ax1[2, 1], d2y_u, "|d2y u_after|", signed=False, crop=crop)
    _plot(ax1[2, 2], d2x_v, "|d2x v_after|", signed=False, crop=crop)
    _plot(ax1[2, 3], d2y_v, "|d2y v_after|", signed=False, crop=crop)
    fig1.suptitle("Velocity components before/after PPE")
    fig1.savefig(out_dir / "01_velocity_before_after_ppe.png", dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _plot(ax2[0, 0], terms["a_pg_dyn_x"], "a_pg_dyn_x", signed=True, crop=crop)
    _plot(ax2[0, 1], terms["a_pg_dyn_y"], "a_pg_dyn_y", signed=True, crop=crop)
    _plot(ax2[0, 2], terms["a_sf_x"], "a_sf_x", signed=True, crop=crop)
    _plot(ax2[0, 3], terms["a_sf_y"], "a_sf_y", signed=True, crop=crop)
    _plot(ax2[1, 0], terms["visc_x"], "visc_x", signed=True, crop=crop)
    _plot(ax2[1, 1], terms["visc_y"], "visc_y", signed=True, crop=crop)
    _plot(ax2[1, 2], terms["conv_acc_x"], "convective_x", signed=True, crop=crop)
    _plot(ax2[1, 3], terms["conv_acc_y"], "convective_y", signed=True, crop=crop)
    fig2.suptitle("Momentum-term decomposition (interface crop)")
    fig2.savefig(out_dir / "02_momentum_terms.png", dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _plot(ax3[0, 0], terms["rho"], "rho", signed=False, crop=crop)
    _plot(ax3[0, 1], terms["Re"], "Re", signed=False, crop=crop)
    _plot(ax3[0, 2], terms["mu_ch_bc"], "mu_CH (after BC)", signed=True, crop=crop)
    _plot(ax3[0, 3], terms["mu_total"], "mu_total", signed=True, crop=crop)
    _plot(ax3[1, 0], terms["diff_phi"], "phase diffusive", signed=True, crop=crop)
    _plot(ax3[1, 1], terms["conv_phi"], "phase convective", signed=True, crop=crop)
    _plot(ax3[1, 2], d2x_s, "|d2x |U||", signed=False, crop=crop)
    _plot(ax3[1, 3], d2y_s, "|d2y |U||", signed=False, crop=crop)
    fig3.suptitle("Phase terms + speed-pattern proxies")
    fig3.savefig(out_dir / "03_phase_terms_and_pattern.png", dpi=180)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    _plot(ax4[0], terms["rhs_cap_raw"], "capillary rhs raw", signed=True, crop=crop)
    _plot(ax4[1], terms["rhs_cap_smoothed"], "capillary rhs smoothed", signed=True, crop=crop)
    _plot(ax4[2], ds, "delta |U| (after-before)", signed=True, crop=crop)
    fig4.suptitle("Capillary forcing maps and PPE response")
    fig4.savefig(out_dir / "04_capillary_rhs_and_delta_speed.png", dpi=180)
    plt.close(fig4)

    np.savez_compressed(
        out_dir / "decomposition_fields.npz",
        U_before=U_before,
        U_after=U_after,
        **terms,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--steps", nargs="*", type=int, default=[])
    ap.add_argument("--interface-band", type=float, default=0.75)
    ap.add_argument("--out-dir", default="diagnostics/vortex_gridpattern_decomposition")
    args = ap.parse_args()

    exp = Path(args.experiment_dir)
    cfg = json.loads((exp / "simulation_parameters.json").read_text())
    steps = _collect_steps(exp, args.steps)
    if not steps:
        print("no PPE diagnostics steps found")
        return

    # Early + latest by default if user did not specify.
    if not args.steps:
        picks = sorted(set(steps[:3] + steps[-1:]))
    else:
        picks = steps

    out_root = Path(args.out_dir) / exp.name
    out_root.mkdir(parents=True, exist_ok=True)

    for step in picks:
        cp = exp / "checkpoints" / f"checkpoint_{step:06d}.npz"
        if not cp.exists():
            print(f"skip step {step}: checkpoint missing")
            continue
        d = np.load(cp)
        phi = np.array(d["phi"], dtype=np.float64)
        P = np.array(d["P"], dtype=np.float64)
        U_before, U_after = _load_ppe_pair(exp, step)
        terms = _term_fields(cfg, phi, P, U_after)
        out = out_root / f"step_{step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        _make_figures(out, U_before, U_after, terms, args.interface_band)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()

