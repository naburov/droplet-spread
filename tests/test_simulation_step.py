#!/usr/bin/env python3
"""
Test a single simulation step to see what's happening.

Updated to current src/ APIs:
- configs/config_water_droplet.json no longer exists; the test uses
  configs/config_template.json (the current reference config, which includes
  solver_params).
- The predictor step uses jax_update_velocity directly with
  geometry.f_1_grid / f_2_grid (zeros for a flat surface). The
  FluidDynamicsSolver.update_velocity wrapper cannot currently be invoked: it
  forwards mu_convention (a string) as a non-static argument to the jitted
  kernel, which TypeErrors under the installed JAX. The jitted kernel is the
  same physics the wrapper would run.
- check_continuity was replaced by jax_check_continuity(U, dx, dy, f_1_grid).
- ppe requires a geometry argument and returns (U_corrected, info_dict); a
  velocity BC manager is passed explicitly because the manager-less fallback
  inside ppe_solve references a removed helper.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from physics.fluid_dynamics import jax_update_velocity, jax_check_continuity
from solvers.projection_methods import ppe
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from config.config_loader import load_config
from solvers.sparse_solver import SparseSolverWrapper
from simulation.geometry import Geometry

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "config_template.json")


def test_single_step():
    """A single predictor + BC + (optional) PPE step stays finite and bounded."""
    # Load config
    config = load_config(CONFIG_PATH)
    assert "solver_params" in config, "template config must provide solver_params"

    # Parameters
    Nx, Ny = 16, 16
    dx = dy = 0.1
    dt = 0.01
    geometry = Geometry.flat(Nx, Ny)

    # Physical parameters
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    Fr = config["physical_params"]["Fr"]
    g = config["physical_params"]["g"]
    include_gravity = config["physical_params"]["include_gravity"]

    # Initialize fields
    U = jnp.zeros((Nx, Ny, 2))
    P = jnp.zeros((Nx, Ny))
    phi = jnp.ones((Nx, Ny))  # Single phase
    surface_tension = jnp.zeros((Nx, Ny, 2))

    correction_solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    correction_solver.set_top_boundary_condition("neumann")
    correction_solver.set_bottom_boundary_condition("neumann")
    correction_solver.set_left_boundary_condition("neumann")
    correction_solver.set_right_boundary_condition("neumann")
    correction_solver.create_sparse_matrix()

    # Step 1: Predictor step
    U_pred = jax_update_velocity(
        U, P, surface_tension, dt, dx, dy,
        rho1, rho2, Re1, Re2, Fr, g, phi,
        geometry.f_1_grid, geometry.f_2_grid,
        include_gravity=include_gravity
    )
    assert np.all(np.isfinite(np.asarray(U_pred))), "predictor produced non-finite velocities"
    if include_gravity:
        # Fluid at rest with zero pressure: predictor change is the gravity body force.
        expected_dv = (1 / Fr**2) * g * dt
        np.testing.assert_allclose(np.asarray(U_pred[..., 1]), expected_dv, rtol=1e-12)

    # Step 2: Apply velocity boundary conditions
    velocity_bc_manager = VelocityBoundaryConditions(config)
    U_bc = velocity_bc_manager.apply_boundary_conditions(U_pred, dx, dy, use_jax=True)
    assert np.all(np.isfinite(np.asarray(U_bc)))

    # Step 3: Check continuity
    divergence, max_div, mean_div = jax_check_continuity(U_bc, dx, dy, geometry.f_1_grid)
    assert np.all(np.isfinite(np.asarray(divergence)))
    assert float(max_div) < 1000.0, "divergence exploded after one predictor step"

    # Step 4: Apply PPE if needed
    ppe_params = config["solver_params"]["ppe"]
    max_div_threshold = ppe_params["max_div_threshold"]
    mean_div_threshold = ppe_params["mean_div_threshold"]
    div_threshold = ppe_params.get("div_threshold", 0.05)

    if max_div > max_div_threshold or mean_div > mean_div_threshold:
        U_corrected, info = ppe(
            U_bc, dx, dy, dt, geometry, correction_solver,
            velocity_bc_manager=velocity_bc_manager,
            div_threshold=div_threshold, max_div_threshold=max_div_threshold,
            mean_div_threshold=mean_div_threshold, max_iterations=50,
        )
        assert np.all(np.isfinite(np.asarray(U_corrected))), "PPE produced non-finite velocities"
        assert not info.get('diverged', False), "PPE must not blow up"

        _, max_div_final, mean_div_final = jax_check_continuity(
            U_corrected, dx, dy, geometry.f_1_grid
        )
        if info['applied']:
            assert float(mean_div_final) <= float(mean_div) + 1e-12, (
                f"PPE increased mean divergence "
                f"({float(mean_div):.3e} -> {float(mean_div_final):.3e})"
            )


if __name__ == "__main__":
    test_single_step()
    print("Single simulation step test passed.")
