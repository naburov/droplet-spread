"""Advection boundary conditions for phase field.

Integration with geometry and gravity:
- Grid is terrain-following (x, eta): surface is always at eta=0 (bottom row j=0).
- Bottom impermeable: no flux of phase through the surface (correct for tilted/sloped walls).
- Left/right: x-boundaries. For tilted sliding (origin bottom_left), gravity drives flow
  toward the origin (negative x). Use left="open" for outflow at the low side; use
  left="impermeable" to block phase at the wall (closed box).
- Open: outflow radiation (∂tφ + c ∂nφ = 0) so phase can advect out; impermeable: no flux
  (boundary phi set from interior so normal gradient is zero or velocity-based copy).
"""

import jax.numpy as jnp
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


class AdvectionBoundaryConditions(BaseBoundaryCondition):
    """Advection BC for phase field with impermeable and open boundaries."""

    BC_ALIASES = {
        **BaseBoundaryCondition.BC_ALIASES,
        "impermeable": BCType.SPECIAL,
        "open": BCType.SPECIAL,
    }

    def __init__(self, config=None):
        super().__init__(config, "advection")
        # Default BC types
        if not self.config.get("boundary_conditions", {}).get("advection"):
            self.bc_raw = {"top": "open", "bottom": "impermeable", "left": "open", "right": "open"}
            self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}

        bc_cfg = self.config.get("boundary_conditions", {}).get("advection", {})
        self.cout = bc_cfg.get("cout", 1.0)
        self.velocity_threshold = bc_cfg.get("velocity_threshold", 1e-10)

    def apply(self, phi, U_or_dx, dt_or_dy=None, dx=None, dy=None, **kwargs):
        """Apply advection BCs. phi shape: (Nx, Ny).

        Supports two calling conventions:
        - apply(phi, dx, dy, U=..., dt=...)  # new style
        - apply(phi, U, dt, dx, dy, ...)     # legacy style
        """
        if dt_or_dy is None or isinstance(dt_or_dy, float) and dx is None:
            dx_val, dy_val = U_or_dx, dt_or_dy
            U = kwargs.get("U")
            dt = kwargs.get("dt", 1e-4)
        else:
            U = U_or_dx
            dt = dt_or_dy
            dx_val, dy_val = dx, dy

        skip_top = kwargs.get("skip_top", False)
        skip_bottom = kwargs.get("skip_bottom", False)
        skip_left = kwargs.get("skip_left", False)
        skip_right = kwargs.get("skip_right", False)

        if self.bc_raw["top"] == "impermeable" and not skip_top:
            phi = self._impermeable_top(phi, U)
        if self.bc_raw["bottom"] == "impermeable" and not skip_bottom:
            phi = self._impermeable_bottom(phi, U)
        if self.bc_raw["left"] == "impermeable" and not skip_left:
            phi = self._impermeable_left(phi, U)
        if self.bc_raw["right"] == "impermeable" and not skip_right:
            phi = self._impermeable_right(phi, U)

        for b, raw in self.bc_raw.items():
            if raw == "open":
                if (b == "top" and skip_top) or (b == "bottom" and skip_bottom) or (b == "left" and skip_left) or (b == "right" and skip_right):
                    continue
                phi = self._open_radiation(phi, b, dt, dx_val, dy_val)

        return phi

    def _impermeable_top(self, phi, U):
        """No flux through the top boundary; mirror the adjacent interior row."""
        if U is None:
            return phi.at[:, -1].set(phi[:, -2])
        v_top = U[:, -1, 1]
        mask = jnp.abs(v_top) > self.velocity_threshold
        phi_top_new = jnp.where(mask, phi[:, -2], phi[:, -1])
        return phi.at[:, -1].set(phi_top_new)

    def _impermeable_bottom(self, phi, U):
        """No flux through surface (eta=0). If velocity points into boundary, copy from interior."""
        if U is None:
            return phi.at[:, 0].set(phi[:, 1])
        v_bottom = U[:, 0, 1]
        mask = jnp.abs(v_bottom) > self.velocity_threshold
        phi_bottom_new = jnp.where(mask, phi[:, 1], phi[:, 0])
        return phi.at[:, 0].set(phi_bottom_new)

    def _impermeable_left(self, phi, U):
        """No flux through left (x=0). Copy from interior so normal gradient ≈ 0 (no advective flux)."""
        return phi.at[0, :].set(phi[1, :])

    def _impermeable_right(self, phi, U):
        """No flux through right (x=Lx). Copy from interior so normal gradient ≈ 0."""
        return phi.at[-1, :].set(phi[-2, :])

    def _open_radiation(self, phi, b, dt, dx, dy):
        """Open BC with radiation: ∂tφ + cout ∂nφ = 0 (outflow extrapolation)."""
        if b == "top":
            dphi = (phi[:, -1] - phi[:, -2]) / dy
            return phi.at[:, -1].set(phi[:, -2] - (self.cout * dt / dy) * dphi)
        elif b == "left":
            dphi = (phi[0, :] - phi[1, :]) / dx
            return phi.at[0, :].set(phi[1, :] - (self.cout * dt / dx) * dphi)
        elif b == "right":
            dphi = (phi[-1, :] - phi[-2, :]) / dx
            return phi.at[-1, :].set(phi[-2, :] - (self.cout * dt / dx) * dphi)
        return phi
