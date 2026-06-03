"""
Two-phase (phase field + incompressible flow) simulation on a staggered MAC grid.

Goal:
  - Reuse the clean staggered MAC predictor + projection we already debugged.
  - Evolve a phase field φ at cell centers using the existing JAX phase-field update.
  - Couple via advection (and later surface tension if desired).

This is intentionally standalone and much simpler than the full SimulationState stack.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp

from numerics.staggered_mac import (
    zeros_mac,
    divergence,
    grad_p_to_faces,
    laplacian_u,
    laplacian_v,
    advect_u,
    advect_v,
    apply_velocity_bcs,
    make_ppe_solver_pyamg,
    solve_pressure_poisson_pyamg,
)
from physics.phase_field import jax_update_phase
from physics.surface_tension import jax_surface_tension_force
from visualization.staggered_flow_logging import (
    StaggeredFlowLoggerConfig,
    save_frame,
    save_telemetry,
)


@dataclass
class TwoPhaseStaggeredConfig:
    # grid / domain
    Nx: int = 96
    Ny: int = 48
    Lx: float = 4.0
    Ly: float = 1.0

    # time / material
    dt: float = 5e-4
    nu: float = 1e-2  # kinematic viscosity (single value for now)

    # phase-field parameters (Cahn–Hilliard style)
    Pe: float = 10.0
    epsilon: float = 0.02
    contact_angle: float = 90.0  # not used yet in BCs here
    lambda_willmore: float = 0.0
    epsilon_willmore: float = 0.0

    # surface tension (dimensionless via Weber numbers used in existing code)
    include_surface_tension: bool = True
    We1: float = 1.0
    We2: float = 1.0
    smooth_curvature: bool = True
    smoothing_radius: int = 1

    steps: int = 400

    # inlet
    u_inlet: float = 1.0
    inlet_profile: str = "linear_to_half"  # "uniform" | "linear_to_half" | "poiseuille"

    # wall BCs
    top_bc: str = "open"  # "no_slip" | "free_slip" | "open"
    bottom_bc: str = "no_slip"  # "no_slip" | "free_slip"
    outflow_right: bool = True

    # phase-field initialization
    droplet_radius: float = 0.2
    droplet_center_x: float = 1.0
    droplet_center_y: float = 0.3

    # physics toggles
    include_advection: bool = True

    # logging
    log_dir: Optional[str] = None
    save_plots: bool = True
    save_every: int = 10


def faces_to_cell_center(u_face, v_face):
    """
    Convert MAC face velocities to cell-centered (u,v).
      u_face: (Nx+1, Ny)
      v_face: (Nx, Ny+1)
    Returns:
      uc, vc: (Nx, Ny) as JAX arrays.
    """
    uc = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    vc = 0.5 * (v_face[:, 1:] + v_face[:, :-1])
    return uc, vc


class TwoPhaseStaggeredSimulation:
    def __init__(self, cfg: TwoPhaseStaggeredConfig):
        self.cfg = cfg
        self.dx = cfg.Lx / cfg.Nx
        self.dy = cfg.Ly / cfg.Ny

        # MAC velocities + pressure (JAX)
        self.u, self.v, self.p = zeros_mac(cfg.Nx, cfg.Ny)
        self.ppe_solver = make_ppe_solver_pyamg(cfg.Nx, cfg.Ny, self.dx, self.dy)

        # Phase field φ on cell centers (JAX)
        self.phi = self._init_droplet()

        # logging
        self._logger_cfg = None
        if cfg.save_plots:
            out_dir = cfg.log_dir
            if out_dir is None:
                out_dir = f"experiment_two_phase_staggered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._logger_cfg = StaggeredFlowLoggerConfig(
                out_dir=out_dir, save_every=max(int(cfg.save_every), 1)
            )

    def _init_droplet(self):
        """Simple circular droplet in a uniform ambient phase."""
        Nx, Ny = self.cfg.Nx, self.cfg.Ny
        dx, dy = self.dx, self.dy
        x = (jnp.arange(Nx) + 0.5) * dx
        y = (jnp.arange(Ny) + 0.5) * dy
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        r = jnp.sqrt((X - self.cfg.droplet_center_x) ** 2 + (Y - self.cfg.droplet_center_y) ** 2)
        phi = jnp.where(r <= self.cfg.droplet_radius, -1.0, 1.0)
        return phi

    def inlet_profile(self):
        """Return u(y) at inlet u-face i=0, length Ny (JAX array)."""
        Ny = self.cfg.Ny
        y = (jnp.arange(Ny) + 0.5) * self.dy
        yhat = y / max(self.cfg.Ly, 1e-30)

        if self.cfg.inlet_profile == "poiseuille":
            prof = 4.0 * self.cfg.u_inlet * yhat * (1.0 - yhat)
            if self.cfg.bottom_bc == "no_slip":
                prof = prof.at[0].set(0.0)
            if self.cfg.top_bc == "no_slip":
                prof = prof.at[-1].set(0.0)
            return prof

        if self.cfg.inlet_profile == "uniform":
            prof = jnp.full(Ny, self.cfg.u_inlet)
            if self.cfg.bottom_bc == "no_slip":
                prof = prof.at[0].set(0.0)
            if self.cfg.top_bc == "no_slip":
                prof = prof.at[-1].set(0.0)
            return prof

        # default: linear rise to u_inlet at y=Ly/2, then constant
        y_half = 0.5 * self.cfg.Ly
        prof = jnp.where(
            y <= y_half, self.cfg.u_inlet * (y / y_half), self.cfg.u_inlet
        )
        if self.cfg.bottom_bc == "no_slip":
            prof = prof.at[0].set(0.0)
        if self.cfg.top_bc == "no_slip":
            prof = prof.at[-1].set(0.0)
        return prof

    def apply_velocity_bcs(self):
        self.u, self.v = apply_velocity_bcs(
            self.u,
            self.v,
            self.inlet_profile(),
            top_bc=self.cfg.top_bc,
            bottom_bc=self.cfg.bottom_bc,
            outflow_right=self.cfg.outflow_right,
        )

    def phase_step(self):
        """Update phase field φ using current (u,v) on staggered grid."""
        uc, vc = faces_to_cell_center(self.u, self.v)
        U = jnp.stack([uc, vc], axis=-1)
        self.phi = jax_update_phase(
            self.phi,
            U,
            self.cfg.dt,
            self.dx,
            self.dy,
            self.cfg.Pe,
            self.cfg.epsilon,
            self.cfg.contact_angle,
            lambda_willmore=self.cfg.lambda_willmore,
            epsilon_willmore=self.cfg.epsilon_willmore,
        )

    def velocity_predictor(self):
        """Compute intermediate (u*, v*) without pressure."""
        dt = self.cfg.dt
        nu = self.cfg.nu

        if self.cfg.include_advection:
            Au = advect_u(self.u, self.v, self.dx, self.dy)
            Av = advect_v(self.u, self.v, self.dx, self.dy)
        else:
            Au = 0.0 * self.u
            Av = 0.0 * self.v

        Lu = laplacian_u(self.u, self.dx, self.dy)
        Lv = laplacian_v(self.v, self.dx, self.dy)

        u_star = self.u + dt * (-Au + nu * Lu)
        v_star = self.v + dt * (-Av + nu * Lv)

        # Pressure-capillary route: no explicit capillary forcing in predictor.
        _ = jax_surface_tension_force

        return u_star, v_star

    def project(self, u_star, v_star):
        """Projection: solve PPE (pyamg) and correct velocities (JAX)."""
        dt = self.cfg.dt

        div_star = divergence(u_star, v_star, self.dx, self.dy)
        rhs = (1.0 / dt) * div_star

        rhs_np = np.asarray(rhs)
        p0_np = np.asarray(self.p) if self.p is not None else None
        p_new_np, iters, res = solve_pressure_poisson_pyamg(
            rhs_np, self.dx, self.dy, self.ppe_solver, p0=p0_np
        )
        p_new = jnp.asarray(p_new_np)

        dpdx, dpdy = grad_p_to_faces(p_new, self.dx, self.dy)
        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy

        return u_new, v_new, p_new, iters, res

    def step(self, n: int):
        # 1) Enforce BCs on current velocities
        self.apply_velocity_bcs()

        # 2) Phase-field update (uses current velocity field)
        self.phase_step()

        # 3) Velocity predictor
        u_star, v_star = self.velocity_predictor()

        # 4) Enforce BCs on u* (important for PPE RHS compatibility)
        self.u, self.v = u_star, v_star
        self.apply_velocity_bcs()
        u_star, v_star = self.u, self.v

        # 5) Projection
        u_new, v_new, p_new, iters, res = self.project(u_star, v_star)
        self.u, self.v, self.p = u_new, v_new, p_new

        # 6) Enforce BCs again after correction
        self.apply_velocity_bcs()

        # Release solver workspace (PyAMG solution/rhs refs) and intermediates
        if hasattr(self.ppe_solver, "clear_workspace"):
            self.ppe_solver.clear_workspace()

        # diagnostics
        div = divergence(self.u, self.v, self.dx, self.dy)
        info = {
            "step": n,
            "ppe_iters": iters,
            "ppe_res": float(res),
            "div_max": float(jnp.max(jnp.abs(div))),
            "div_mean": float(jnp.mean(jnp.abs(div))),
            "u_min": float(jnp.min(self.u)),
            "u_max": float(jnp.max(self.u)),
            "v_min": float(jnp.min(self.v)),
            "v_max": float(jnp.max(self.v)),
            "p_min": float(jnp.min(self.p)),
            "p_max": float(jnp.max(self.p)),
        }

        # optional logging
        if self._logger_cfg is not None:
            save_frame(
                self._logger_cfg,
                step=n,
                u_face=self.u,
                v_face=self.v,
                p_cell=self.p,
                phi_cell=self.phi,
                Lx=self.cfg.Lx,
                Ly=self.cfg.Ly,
                dx=self.dx,
                dy=self.dy,
                title_extra=f"div_max={info['div_max']:.2e}",
            )

        return info

    def run(self, *, log_every: int = 10):
        hist = []
        for n in range(self.cfg.steps):
            info = self.step(n)
            hist.append(info)
            if n > 0 and n % 50 == 0:
                gc.collect()
            if (n % log_every) == 0 or n == self.cfg.steps - 1:
                print(
                    f"[step {n:04d}] div_max={info['div_max']:.3e} div_mean={info['div_mean']:.3e} "
                    f"ppe(iters={info['ppe_iters']}, res={info['ppe_res']:.2e}) "
                    f"u[{info['u_min']:.3f},{info['u_max']:.3f}] p[{info['p_min']:.3f},{info['p_max']:.3f}]"
                )

        if self._logger_cfg is not None:
            save_telemetry(
                self._logger_cfg, history=hist, dx=self.dx, dy=self.dy, dt=self.cfg.dt
            )
        return hist

