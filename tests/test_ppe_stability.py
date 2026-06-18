#!/usr/bin/env python3
"""
Test PPE stability with different velocity fields.

Updated to current src/ APIs:
- check_continuity was replaced by physics.fluid_dynamics.jax_check_continuity,
  which requires the terrain gradient grid f_1_grid (zeros for a flat surface).
- solvers.projection_methods.ppe (= solvers.ppe.ppe_solve) now requires a
  geometry argument and returns (U_corrected, info_dict) instead of just U.
- The solver-less PPE path (correction_solver=None) and the manager-less
  velocity BC fallback inside ppe_solve are bit-rotted in src (removed helper /
  numerically unstable fallback), so both tests now exercise the supported
  configuration: an explicit SparseSolverWrapper correction solver plus a
  velocity BC manager (free-slip box), which is the closest current code path
  to what the original test checked.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from solvers.projection_methods import ppe
from physics.fluid_dynamics import jax_check_continuity
from simulation.geometry import Geometry
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from solvers.sparse_solver import SparseSolverWrapper


def _make_velocity_bc_manager():
    """Free-slip (no-penetration) box, the configuration PPE converges under."""
    return VelocityBoundaryConditions({
        "boundary_conditions": {
            "velocity": {
                "top": "slip_symmetry",
                "bottom": "slip_symmetry",
                "left": "slip_symmetry",
                "right": "slip_symmetry",
            }
        }
    })


def _make_correction_solver(Nx, Ny, dx, dy):
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    solver.set_top_boundary_condition("neumann")
    solver.set_bottom_boundary_condition("neumann")
    solver.set_left_boundary_condition("neumann")
    solver.set_right_boundary_condition("neumann")
    solver.create_sparse_matrix()
    return solver


def test_ppe_stability():
    """PPE must stay stable (no NaN/blow-up) and not increase divergence for simple fields."""
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01
    geometry = Geometry.flat(Nx, Ny)

    rng = np.random.default_rng(0)
    test_cases = [
        ("Zero field", jnp.zeros((Nx, Ny, 2))),
        ("Small constant", jnp.ones((Nx, Ny, 2)) * 0.01),
        ("Gravity-like", jnp.stack([
            jnp.zeros((Nx, Ny)),
            jnp.ones((Nx, Ny)) * -0.1
        ], axis=-1)),
        ("Random small", jnp.array(rng.normal(0, 0.01, (Nx, Ny, 2)))),
    ]

    for name, U in test_cases:
        # Divergence before PPE
        _, max_div_before, mean_div_before = jax_check_continuity(U, dx, dy, geometry.f_1_grid)

        U_corrected, info = ppe(
            U, dx, dy, dt, geometry,
            correction_solver=_make_correction_solver(Nx, Ny, dx, dy),
            velocity_bc_manager=_make_velocity_bc_manager(),
            max_iterations=50,
        )

        # PPE must never produce non-finite velocities or report blow-up.
        assert np.all(np.isfinite(np.asarray(U_corrected))), f"{name}: PPE produced non-finite velocities"
        assert not info.get('diverged', False), f"{name}: PPE reported divergence blow-up"

        _, max_div_after, mean_div_after = jax_check_continuity(U_corrected, dx, dy, geometry.f_1_grid)

        if not info['applied']:
            # Already (numerically) divergence-free: velocity must pass through unchanged.
            np.testing.assert_array_equal(np.asarray(U_corrected), np.asarray(U))
        else:
            # PPE applied: mean divergence must not grow.
            assert float(mean_div_after) <= float(mean_div_before) + 1e-12, \
                f"{name}: PPE increased mean divergence " \
                f"({float(mean_div_before):.3e} -> {float(mean_div_after):.3e})"

        # Constant fields are exactly divergence-free, so PPE should be a no-op for them.
        if name in ("Zero field", "Small constant", "Gravity-like"):
            assert float(max_div_before) < 1e-12, f"{name}: constant field should be divergence-free"
            assert not info['applied'], f"{name}: PPE should not trigger on a divergence-free field"


def test_ppe_with_solver():
    """PPE with an explicit sparse correction solver reduces divergence of a divergent field."""
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01
    geometry = Geometry.flat(Nx, Ny)

    # Genuinely divergent velocity field: u = sin(pi x), v = 0.
    x = jnp.arange(Nx, dtype=jnp.float64) * dx
    U = jnp.stack([
        jnp.broadcast_to(0.1 * jnp.sin(jnp.pi * x)[:, None], (Nx, Ny)),
        jnp.zeros((Nx, Ny)),
    ], axis=-1)

    _, max_div_before, mean_div_before = jax_check_continuity(U, dx, dy, geometry.f_1_grid)
    assert float(max_div_before) > 0.05, "test field should start with significant divergence"

    U_corrected, info = ppe(
        U, dx, dy, dt, geometry,
        correction_solver=_make_correction_solver(Nx, Ny, dx, dy),
        velocity_bc_manager=_make_velocity_bc_manager(),
        max_iterations=50,
    )

    assert info['applied'], "PPE should engage on a divergent field"
    assert not info.get('diverged', False), "PPE with sparse solver must not blow up"
    assert np.all(np.isfinite(np.asarray(U_corrected)))

    _, max_div_after, mean_div_after = jax_check_continuity(U_corrected, dx, dy, geometry.f_1_grid)
    assert float(mean_div_after) < float(mean_div_before), \
        f"PPE with solver should reduce mean divergence " \
        f"({float(mean_div_before):.3e} -> {float(mean_div_after):.3e})"


if __name__ == "__main__":
    test_ppe_stability()
    test_ppe_with_solver()
    print("PPE stability tests passed.")
