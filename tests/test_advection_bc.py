#!/usr/bin/env python3
"""
Tests for advection boundary conditions.

Checks the advection BCs provided by AdvectionBoundaryConditions:
- Bottom: Impermeable (no phase flux through the surface)
- Top/Left/Right: Open with radiation (dphi/dt + cout * dphi/dn = 0)

Updated to current src/ APIs:
- The free function apply_advection_boundary_conditions was removed; the class
  AdvectionBoundaryConditions is the only entry point now.
- BC application is JAX-only (phi.at[...]), so inputs are jnp arrays.
- Matplotlib visualization was dropped; the checks are now numerical assertions
  on the boundary rows/columns the plots used to show.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from boundary_conditions.advection_bc import AdvectionBoundaryConditions
from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions


def create_test_phase_field(Nx=64, Ny=64, radius=0.2, cx=0.5, cy=0.3):
    """Create a test phase field with a circular droplet."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Distance from center
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Create phase field: +1 inside droplet, -1 outside
    phi = np.tanh((radius - r) / 0.05)

    return jnp.asarray(phi), x, y


def create_test_velocity_field(Nx=64, Ny=64):
    """Create a test velocity field."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    U = np.zeros((Nx, Ny, 2))

    # Horizontal velocity (left to right)
    U[:, :, 0] = 0.1 * np.sin(np.pi * Y)  # Zero at boundaries

    # Vertical velocity (upward)
    U[:, :, 1] = 0.05 * np.sin(np.pi * X)  # Zero at boundaries

    return jnp.asarray(U)


def _make_config(top="open", bottom="impermeable", left="open", right="open", cout=1.0):
    return {
        "boundary_conditions": {
            "advection": {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "cout": cout,
                "velocity_threshold": 1e-10,
            }
        }
    }


def test_advection_boundary_conditions():
    """Open (radiation) and impermeable BCs modify only the boundary cells, with the expected values."""
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001

    advection_bc = AdvectionBoundaryConditions(_make_config(cout=1.0))
    phi_with_bc = advection_bc.apply_boundary_conditions(phi, U, dt, dx, dy)

    phi_np = np.asarray(phi)
    phi_bc_np = np.asarray(phi_with_bc)

    # Interior must be untouched.
    np.testing.assert_array_equal(phi_bc_np[1:-1, 1:-1], phi_np[1:-1, 1:-1])

    # Bottom impermeable: where |v_bottom| > threshold, copy from interior; else keep.
    v_bottom = np.asarray(U)[:, 0, 1]
    mask = np.abs(v_bottom) > 1e-10
    expected_bottom = np.where(mask, phi_np[:, 1], phi_np[:, 0])
    np.testing.assert_allclose(phi_bc_np[1:-1, 0], expected_bottom[1:-1], rtol=0, atol=1e-12)

    # Top open radiation: phi_top = phi[:, -2] - (cout*dt/dy) * (phi[:, -1] - phi[:, -2]) / dy
    expected_top = phi_np[:, -2] - (1.0 * dt / dy) * (phi_np[:, -1] - phi_np[:, -2]) / dy
    np.testing.assert_allclose(phi_bc_np[1:-1, -1], expected_top[1:-1], rtol=0, atol=1e-12)

    # Different cout values must change the open-boundary update proportionally.
    # Use a droplet near the top so the open top boundary sees an actual interface gradient.
    phi_top, _, _ = create_test_phase_field(cy=0.9)
    phi_top_np = np.asarray(phi_top)
    results = {}
    for cout in (0.5, 1.0, 2.0):
        bc = AdvectionBoundaryConditions(_make_config(cout=cout))
        phi_cout = np.asarray(bc.apply_boundary_conditions(phi_top, U, dt, dx, dy))
        results[cout] = phi_cout
        expected = phi_top_np[:, -2] - (cout * dt / dy) * (phi_top_np[:, -1] - phi_top_np[:, -2]) / dy
        np.testing.assert_allclose(phi_cout[1:-1, -1], expected[1:-1], rtol=0, atol=1e-12)

    assert not np.allclose(results[0.5][:, -1], results[2.0][:, -1]), \
        "cout must affect the open (radiation) boundary update"


def test_impermeable_vs_open():
    """Impermeable bottom copies from interior; open bottom leaves the surface row alone."""
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001

    advection_impermeable = AdvectionBoundaryConditions(_make_config(bottom="impermeable"))
    advection_open = AdvectionBoundaryConditions(_make_config(bottom="open"))

    phi_impermeable = np.asarray(advection_impermeable.apply_boundary_conditions(phi, U, dt, dx, dy))
    phi_open = np.asarray(advection_open.apply_boundary_conditions(phi, U, dt, dx, dy))

    phi_np = np.asarray(phi)
    v_bottom = np.asarray(U)[:, 0, 1]
    mask = np.abs(v_bottom) > 1e-10

    # Impermeable: bottom row copied from interior wherever vertical velocity is significant.
    expected_bottom = np.where(mask, phi_np[:, 1], phi_np[:, 0])
    np.testing.assert_allclose(phi_impermeable[1:-1, 0], expected_bottom[1:-1], atol=1e-12)

    # Open bottom: no radiation update is implemented for the surface row (by design,
    # the bottom is the substrate), so the bottom row keeps its advected values.
    np.testing.assert_allclose(phi_open[1:-1, 0], phi_np[1:-1, 0], atol=1e-12)

    # Both configurations treat the open top identically.
    np.testing.assert_allclose(phi_impermeable[1:-1, -1], phi_open[1:-1, -1], atol=1e-12)


def test_integration_with_phase_field_bc():
    """Advection BCs compose with the phase-field (contact angle) BC manager."""
    phi, x, y = create_test_phase_field()
    U = create_test_velocity_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.001

    # Note: contact_angle_method "robin" no longer exists; "simple" is the current
    # equivalent boundary-row update method.
    config = {
        "physical_params": {
            "contact_angle": 60,
            "epsilon": 0.02
        },
        "boundary_conditions": {
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "simple"
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "open",
                "right": "open",
                "cout": 1.0
            }
        }
    }

    advection_bc = AdvectionBoundaryConditions(config)
    phase_field_bc = PhaseFieldBoundaryConditions(config)

    phi_advection = advection_bc.apply_boundary_conditions(phi, U, dt, dx, dy)
    phi_combined = phase_field_bc.apply_boundary_conditions(phi_advection, dx, dy)

    phi_np = np.asarray(phi)
    combined_np = np.asarray(phi_combined)

    # Interior preserved through both BC applications.
    np.testing.assert_array_equal(combined_np[1:-1, 1:-1], phi_np[1:-1, 1:-1])

    # Phase-field Neumann boundaries hold after the combined application.
    np.testing.assert_allclose(combined_np[:, -1], combined_np[:, -2], atol=1e-12)  # top
    np.testing.assert_allclose(combined_np[0, :], combined_np[1, :], atol=1e-12)    # left
    np.testing.assert_allclose(combined_np[-1, :], combined_np[-2, :], atol=1e-12)  # right

    # Everything stays finite and within the physical phase-field bounds (with small slack
    # for the boundary extrapolation).
    assert np.all(np.isfinite(combined_np))
    assert np.max(np.abs(combined_np)) < 1.5


if __name__ == "__main__":
    test_advection_boundary_conditions()
    test_impermeable_vs_open()
    test_integration_with_phase_field_bc()
    print("All advection BC tests completed successfully!")
