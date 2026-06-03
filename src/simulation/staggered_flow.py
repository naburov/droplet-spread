"""
Standalone staggered-grid (MAC) incompressible flow simulation.

Goal: a minimal, self-contained predictor + projection loop to debug
checkerboard / PPE stability without the droplet/phase-field stack.
Uses JAX/JIT for MAC ops and pyamg for the pressure Poisson solve.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from datetime import datetime

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

from visualization.staggered_flow_logging import (
    StaggeredFlowLoggerConfig,
    save_frame,
    save_telemetry,
)


@dataclass
class StaggeredFlowConfig:
    Nx: int = 64
    Ny: int = 64
    Lx: float = 1.0
    Ly: float = 1.0
    dt: float = 1e-4
    nu: float = 1e-2  # kinematic viscosity

    steps: int = 200
    ppe_max_iter: int = 8000
    ppe_tol: float = 1e-6
    ppe_omega: float = 1.7

    # inlet
    u_inlet: float = 1.0
    inlet_profile: str = "linear_to_half"  # "uniform" | "linear_to_half" | "poiseuille"

    # wall BCs
    top_bc: str = "no_slip"  # "no_slip" | "free_slip"
    bottom_bc: str = "no_slip"  # "no_slip" | "free_slip"
    outflow_right: bool = True

    # physics toggles
    include_advection: bool = True

    # logging
    log_dir: str | None = None
    save_plots: bool = True
    save_every: int = 1


class StaggeredFlowSimulation:
    def __init__(self, cfg: StaggeredFlowConfig):
        self.cfg = cfg
        self.dx = cfg.Lx / cfg.Nx
        self.dy = cfg.Ly / cfg.Ny
        self.u, self.v, self.p = zeros_mac(cfg.Nx, cfg.Ny)  # JAX arrays
        self.ppe_solver = make_ppe_solver_pyamg(cfg.Nx, cfg.Ny, self.dx, self.dy)
        self._logger_cfg = None
        if cfg.save_plots:
            out_dir = cfg.log_dir
            if out_dir is None:
                out_dir = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._logger_cfg = StaggeredFlowLoggerConfig(
                out_dir=out_dir, save_every=max(int(cfg.save_every), 1)
            )

    def inlet_profile(self):
        """Return u(y) at inlet u-face i=0, length Ny (JAX array)."""
        Ny = self.cfg.Ny
        y = (jnp.arange(Ny) + 0.5) * self.dy
        yhat = y / max(self.cfg.Ly, 1e-30)

        if self.cfg.inlet_profile == "poiseuille":
            # plane Poiseuille (parabolic) with peak = u_inlet at centerline
            prof = 4.0 * self.cfg.u_inlet * yhat * (1.0 - yhat)
            # enforce no-slip only if the corresponding wall is no-slip
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
        y_half = 0.5 * self.cfg.Ly
        prof = jnp.where(
            y <= y_half, self.cfg.u_inlet * (y / y_half), self.cfg.u_inlet
        )
        if self.cfg.bottom_bc == "no_slip":
            prof = prof.at[0].set(0.0)
        if self.cfg.top_bc == "no_slip":
            prof = prof.at[-1].set(0.0)
        return prof

    def apply_bcs(self):
        self.u, self.v = apply_velocity_bcs(
            self.u,
            self.v,
            self.inlet_profile(),
            top_bc=self.cfg.top_bc,
            bottom_bc=self.cfg.bottom_bc,
            outflow_right=self.cfg.outflow_right,
        )

    def predictor(self):
        """Compute intermediate (u*, v*) without pressure."""
        dt = self.cfg.dt
        nu = self.cfg.nu

        # advection term on faces
        if self.cfg.include_advection:
            Au = advect_u(self.u, self.v, self.dx, self.dy)
            Av = advect_v(self.u, self.v, self.dx, self.dy)
        else:
            Au = 0.0 * self.u
            Av = 0.0 * self.v

        # diffusion on faces
        Lu = laplacian_u(self.u, self.dx, self.dy)
        Lv = laplacian_v(self.v, self.dx, self.dy)

        u_star = self.u + dt * (-Au + nu * Lu)
        v_star = self.v + dt * (-Av + nu * Lv)
        return u_star, v_star

    def project(self, u_star, v_star):
        """Projection: solve PPE (pyamg) and correct velocities (JAX)."""
        dt = self.cfg.dt

        div_star = divergence(u_star, v_star, self.dx, self.dy)
        rhs = (1.0 / dt) * div_star  # ∇²p = (1/dt) ∇·u*

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

    def step(self):
        # BCs on current fields
        self.apply_bcs()

        # predictor
        u_star, v_star = self.predictor()

        # enforce BCs on u* (important)
        self.u, self.v = u_star, v_star
        self.apply_bcs()
        u_star, v_star = self.u, self.v

        # projection
        u_new, v_new, p_new, iters, res = self.project(u_star, v_star)
        self.u, self.v, self.p = u_new, v_new, p_new

        # enforce BCs again after correction
        self.apply_bcs()

        # Release solver workspace (PyAMG solution/rhs refs)
        if hasattr(self.ppe_solver, "clear_workspace"):
            self.ppe_solver.clear_workspace()

        # diagnostics (JAX arrays -> Python floats for logging)
        div = divergence(self.u, self.v, self.dx, self.dy)
        return {
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

    def run(self, *, log_every: int = 10):
        hist = []
        for n in range(self.cfg.steps):
            info = self.step()
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
                save_frame(
                    self._logger_cfg,
                    step=n,
                    u_face=self.u,
                    v_face=self.v,
                    p_cell=self.p,
                    Lx=self.cfg.Lx,
                    Ly=self.cfg.Ly,
                    dx=self.dx,
                    dy=self.dy,
                    title_extra=f"div_max={info['div_max']:.2e}",
                )
        if self._logger_cfg is not None:
            save_telemetry(
                self._logger_cfg, history=hist, dx=self.dx, dy=self.dy, dt=self.cfg.dt
            )
        return hist

