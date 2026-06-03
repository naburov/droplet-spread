#!/usr/bin/env python3
"""
Run exactly one MAC predictor step from a checkpoint and trace u_x.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from numerics.staggered_utils import to_collocated, to_staggered
from simulation.geometry import Geometry
from solvers.staggered_velocity import staggered_predictor_step


def _pick_checkpoint(experiment_dir: Path, step: int | None) -> tuple[int, Path]:
    cdir = experiment_dir / "checkpoints"
    cands = sorted(cdir.glob("checkpoint_*.npz"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {cdir}")
    if step is None:
        p = cands[-1]
        return int(p.stem.split("_")[1]), p
    p = cdir / f"checkpoint_{step:06d}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return int(step), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out-dir", default="diagnostics/one_predictor_step_trace")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    exp = Path(args.experiment_dir)
    step, cp = _pick_checkpoint(exp, args.step)
    z = np.load(cp)

    U0 = np.array(z["U"], dtype=np.float64)
    P = np.array(z["P"], dtype=np.float64)
    phi = np.array(z["phi"], dtype=np.float64)
    nx, ny = phi.shape

    dx = float(cfg["grid_params"]["Lx"]) / float(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / float(cfg["grid_params"]["Ny"])
    dt = float(cfg.get("time_params", {}).get("dt_initial", cfg.get("time_params", {}).get("dt", 1e-3)))

    pp = cfg["physical_params"]
    re2 = float(pp["Re2"])
    fr = float(pp["Fr"])
    g = float(pp.get("g", -1.0))
    include_gravity = bool(pp.get("include_gravity", True))

    geom = Geometry.flat(nx, ny)

    if "u_face" in z.files and "v_face" in z.files:
        u0_face = jnp.array(np.array(z["u_face"], dtype=np.float64))
        v0_face = jnp.array(np.array(z["v_face"], dtype=np.float64))
    else:
        u0_face, v0_face = to_staggered(jnp.array(U0))

    sf_dummy = jnp.zeros((nx, ny, 2), dtype=jnp.float64)
    u1_face, v1_face = staggered_predictor_step(
        u0_face,
        v0_face,
        sf_dummy,
        dt,
        dx,
        dy,
        re2,
        fr,
        g,
        include_gravity=include_gravity,
        include_advection=True,
        P=jnp.array(P),
        phi=jnp.array(phi),
        rho1=float(pp["rho1"]),
        rho2=float(pp["rho2"]),
        geometry=geom,
    )

    vel_bc = VelocityBoundaryConditions(cfg)
    u1_face, v1_face = vel_bc.apply_to_faces(u1_face, v1_face, dx, dy, psi=None, geometry=geom, phi=jnp.array(phi))
    U1 = np.array(to_collocated(u1_face, v1_face), dtype=np.float64)

    u0 = U0[..., 0]
    u1 = U1[..., 0]
    du = u1 - u0

    out = Path(args.out_dir) / exp.name / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    umax = float(max(np.max(np.abs(u0)), np.max(np.abs(u1)), 1e-12))
    dumax = float(max(np.max(np.abs(du)), 1e-12))
    im0 = ax[0].imshow(u0.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-umax, vmax=umax)
    ax[0].set_title("u_x before predictor")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(u1.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-umax, vmax=umax)
    ax[1].set_title("u_x after predictor")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(du.T, origin="lower", aspect="auto", cmap="bwr", vmin=-dumax, vmax=dumax)
    ax[2].set_title("delta u_x (after - before)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    for a in ax:
        a.set_xlabel("i")
        a.set_ylabel("j")
    fig.savefig(out / "u_x_predictor_maps.png", dpi=180)
    plt.close(fig)

    x = np.arange(nx)
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for k, j in enumerate([0, 1, 2]):
        jj = min(j, ny - 1)
        ax2[k].plot(x, u0[:, jj], "b-", linewidth=1.2, label="before")
        ax2[k].plot(x, u1[:, jj], "r-", linewidth=1.2, label="after")
        ax2[k].plot(x, du[:, jj], "k--", linewidth=1.1, label="delta")
        ax2[k].set_title(f"u_x row j={jj}")
        ax2[k].set_xlabel("i")
        ax2[k].grid(alpha=0.25)
        ax2[k].legend(fontsize=8)
    fig2.savefig(out / "u_x_bottom_profiles.png", dpi=180)
    plt.close(fig2)

    summary = {
        "config": str(Path(args.config)),
        "experiment": exp.name,
        "step": int(step),
        "dt_used": dt,
        "ux_before_abs_mean": float(np.mean(np.abs(u0))),
        "ux_after_abs_mean": float(np.mean(np.abs(u1))),
        "dux_abs_mean": float(np.mean(np.abs(du))),
        "dux_abs_max": float(np.max(np.abs(du))),
        "row0_dux_abs_mean": float(np.mean(np.abs(du[:, 0]))),
        "row1_dux_abs_mean": float(np.mean(np.abs(du[:, min(1, ny - 1)]))),
        "row2_dux_abs_mean": float(np.mean(np.abs(du[:, min(2, ny - 1)]))),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "u_x_predictor_fields.npz", u_before=u0, u_after=u1, du=du)

    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

