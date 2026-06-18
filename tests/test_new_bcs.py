#!/usr/bin/env python3
"""
Test script for the new boundary condition setup.

Tests the updated boundary conditions:
1. Pressure: Top Dirichlet p=0, all others Neumann dp/dn=0
2. Velocity: Left/Right slip/symmetry u=0, du/dn=0; Top open du/dn=0; Bottom no-slip u=0

Updated to current src/ APIs:
- BC application is JAX-only (field.at[...]), so inputs are jnp arrays.
- PressureBoundaryConditions.apply now requires dx, dy.
- The printed diagnostics from the original script are now assertions.
- The "write test_config.json and run main manually" step was replaced by
  constructing all BC managers from the same config dict and asserting they
  pick up the configured boundary types.
- Matplotlib visualization was dropped.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from boundary_conditions.pressure_bc import PressureBoundaryConditions


def create_test_velocity_field(Nx=64, Ny=64):
    """Create a test velocity field with some flow."""
    U = np.zeros((Nx, Ny, 2))

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Horizontal flow (left to right)
    U[:, :, 0] = 0.1 * np.sin(np.pi * Y) * np.cos(np.pi * X)

    # Vertical flow (upward)
    U[:, :, 1] = 0.05 * np.sin(np.pi * X) * np.cos(np.pi * Y)

    return jnp.asarray(U)


def create_test_pressure_field(Nx=64, Ny=64):
    """Create a test pressure field."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    P = 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    return jnp.asarray(P)


def test_velocity_boundary_conditions():
    """no_slip / slip_symmetry / do_nothing velocity BCs hold on their boundaries."""
    U = create_test_velocity_field()
    dx = dy = 1.0 / 63  # Grid spacing

    config = {
        "boundary_conditions": {
            "velocity": {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "slip_symmetry",
                "right": "slip_symmetry"
            }
        }
    }

    velocity_bc = VelocityBoundaryConditions(config)
    U_with_bc = np.asarray(velocity_bc.apply_boundary_conditions(U, dx, dy))

    # Bottom: no-slip (u = 0, v = 0)
    np.testing.assert_allclose(U_with_bc[:, 0, 0], 0.0, atol=1e-12,
                               err_msg="bottom no-slip: u must vanish")
    np.testing.assert_allclose(U_with_bc[:, 0, 1], 0.0, atol=1e-12,
                               err_msg="bottom no-slip: v must vanish")

    # Left: slip/symmetry (u = 0, dv/dx = 0)
    np.testing.assert_allclose(U_with_bc[0, :, 0], 0.0, atol=1e-12,
                               err_msg="left slip/symmetry: normal velocity must vanish")
    np.testing.assert_allclose(U_with_bc[0, :, 1], U_with_bc[1, :, 1], atol=1e-12,
                               err_msg="left slip/symmetry: tangential gradient must vanish")

    # Right: slip/symmetry (u = 0, dv/dx = 0)
    np.testing.assert_allclose(U_with_bc[-1, :, 0], 0.0, atol=1e-12,
                               err_msg="right slip/symmetry: normal velocity must vanish")
    np.testing.assert_allclose(U_with_bc[-1, :, 1], U_with_bc[-2, :, 1], atol=1e-12,
                               err_msg="right slip/symmetry: tangential gradient must vanish")

    # Top: do-nothing (dU/dy = 0)
    np.testing.assert_allclose(U_with_bc[1:-1, -1, :], U_with_bc[1:-1, -2, :], atol=1e-12,
                               err_msg="top do-nothing: normal gradient must vanish")

    # Interior must be untouched.
    np.testing.assert_array_equal(U_with_bc[1:-1, 1:-1, :], np.asarray(U)[1:-1, 1:-1, :])


def test_pressure_boundary_conditions():
    """Top open (Dirichlet p=0) and Neumann elsewhere hold after application."""
    P = create_test_pressure_field()
    dx = dy = 1.0 / 63

    config = {
        "boundary_conditions": {
            "pressure": {
                "top": "open",
                "bottom": "neumann",
                "left": "neumann",
                "right": "neumann",
                "open_pressure": 0.0
            }
        }
    }

    pressure_bc = PressureBoundaryConditions(config)
    # apply now requires dx, dy
    P_with_bc = np.asarray(pressure_bc.apply_boundary_conditions(P, dx, dy))

    # Top: Dirichlet (p = 0)
    np.testing.assert_allclose(P_with_bc[:, -1], 0.0, atol=1e-12,
                               err_msg="top open BC: p must equal open_pressure (0)")

    # Bottom: Neumann (dp/dy = 0)
    np.testing.assert_allclose(P_with_bc[:, 0], P_with_bc[:, 1], atol=1e-12,
                               err_msg="bottom Neumann: dp/dy must vanish")

    # Left: Neumann (dp/dx = 0)
    np.testing.assert_allclose(P_with_bc[0, :], P_with_bc[1, :], atol=1e-12,
                               err_msg="left Neumann: dp/dx must vanish")

    # Right: Neumann (dp/dx = 0)
    np.testing.assert_allclose(P_with_bc[-1, :], P_with_bc[-2, :], atol=1e-12,
                               err_msg="right Neumann: dp/dx must vanish")

    # Interior must be untouched.
    np.testing.assert_array_equal(P_with_bc[1:-1, 1:-1], np.asarray(P)[1:-1, 1:-1])


def test_simulation_with_new_bcs():
    """The full BC config block is accepted by all BC managers."""
    test_config = {
        "physical_params": {
            "rho2": 1000.0,
            "Re2": 100.0,
            "We2": 10.0,
            "rho1": 1.0,
            "Re1": 10.0,
            "We1": 0.1,
            "Pe": 1.0,
            "epsilon": 0.05,
            "contact_angle": 120,
            "include_gravity": True,
            "Fr": 1.0,
            "g": -1.0,
            "atm_pressure": 0.0
        },
        "grid_params": {
            "Lx": 1.0,
            "Ly": 1.0,
            "Nx": 64,
            "Ny": 64
        },
        "time_params": {
            "dt": 0.001,
            "t_max": 0.01,
            "checkpoint_interval": 5,
            "dt_initial": 0.001,
            "cfl_number": 0.05
        },
        "initial_conditions": {
            "droplet_radius": 0.2
        },
        "boundary_conditions": {
            "pressure": {
                "top": "open",
                "bottom": "neumann",
                "left": "neumann",
                "right": "neumann",
                "open_pressure": 0.0
            },
            "velocity": {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "slip_symmetry",
                "right": "slip_symmetry"
            },
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                # "robin" was removed; "simple" is the current method.
                "contact_angle_method": "simple"
            },
            "chemical_potential": {
                "top": "zero_flux",
                "bottom": "zero_flux",
                "left": "zero_flux",
                "right": "zero_flux"
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "impermeable",
                "right": "impermeable",
                "cout": 1.0,
                "velocity_threshold": 1e-10
            }
        }
    }

    from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions
    from boundary_conditions.advection_bc import AdvectionBoundaryConditions

    velocity_bc = VelocityBoundaryConditions(test_config)
    pressure_bc = PressureBoundaryConditions(test_config)
    phase_field_bc = PhaseFieldBoundaryConditions(test_config)
    advection_bc = AdvectionBoundaryConditions(test_config)

    assert velocity_bc.bc_raw == {
        "top": "do_nothing", "bottom": "no_slip",
        "left": "slip_symmetry", "right": "slip_symmetry",
    }
    assert pressure_bc.bc_raw == {
        "top": "open", "bottom": "neumann", "left": "neumann", "right": "neumann",
    }
    assert pressure_bc.open_pressure == 0.0
    assert phase_field_bc.bc_raw["bottom"] == "contact_angle"
    assert phase_field_bc.contact_angle_bc.contact_angle == 120
    assert phase_field_bc.contact_angle_bc.method == "simple"
    assert advection_bc.bc_raw == {
        "top": "open", "bottom": "impermeable",
        "left": "impermeable", "right": "impermeable",
    }
    assert advection_bc.cout == 1.0


if __name__ == "__main__":
    test_velocity_boundary_conditions()
    test_pressure_boundary_conditions()
    test_simulation_with_new_bcs()
    print("All new BC tests completed successfully!")
