"""
Velocity operator backends: collocated (current) and staggered (MAC).

This module provides a thin abstraction so that the high-level
two-phase simulation can switch between:
  - collocated velocity layout (existing code path)
  - staggered MAC layout (to be implemented incrementally)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp

from physics.fluid_dynamics import FluidDynamicsSolver, jax_check_continuity


@dataclass
class CollocatedVelocityBackend:
    """Wrapper around existing FluidDynamicsSolver for collocated layout."""

    solver: FluidDynamicsSolver

    def predictor(
        self,
        U,
        p,
        surface_tension,
        dt: float,
        dx: float,
        dy: float,
        phi,
        geometry,
        include_gravity: bool = False,
        psi=None,
    ):
        """Return predicted velocity U* on collocated layout. geometry: from state."""
        U_star = self.solver.update_velocity(
            U, p, surface_tension, dt, dx, dy, phi, geometry,
            include_gravity=include_gravity,
            use_jax=True,
            psi=psi,
        )
        return U_star

    def continuity_diagnostics(self, U, dx: float, dy: float, geometry) -> Dict[str, Any]:
        """Return divergence field and simple diagnostics. geometry: from state."""
        div_field, div_max, div_mean = self.solver.check_continuity(U, dx, dy, geometry)
        return {
            "div_field": div_field,
            "div_max": float(div_max),
            "div_mean": float(div_mean),
        }


def make_velocity_backend(
    layout: str,
    solver: FluidDynamicsSolver,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    config: Dict[str, Any],
):
    """
    Factory for velocity backends.

    layout:
      - "collocated": current default, uses FluidDynamicsSolver directly
      - "staggered": TODO – will use numerics.staggered_mac operators
    """
    layout = layout.lower()
    if layout == "collocated":
        return CollocatedVelocityBackend(solver)

    if layout == "staggered":
        # Placeholder for future MAC integration
        raise NotImplementedError(
            "Staggered velocity backend not wired into TwoPhaseSimulation yet."
        )

    raise ValueError(f"Unknown velocity layout: {layout}")

