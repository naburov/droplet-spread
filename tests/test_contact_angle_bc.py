#!/usr/bin/env python3
"""
Tests for configurable contact angle boundary conditions.

Updated to current src/ APIs:
- The "robin" and "young_laplace" contact angle methods were removed from src.
  Current methods are "simple", "geometry_aware" and "ghost_cell"; the tests
  were re-targeted to those (see individual docstrings).
- ContactAngleBoundaryCondition.apply is JAX-only (uses phi.at[...]), so phi is
  converted with jnp.asarray.
- PhaseFieldBoundaryConditions no longer exposes get_boundary_info /
  update_contact_angle; tests read bc_raw and set the contact angle on the
  nested contact_angle_bc directly.
- Matplotlib visualization was dropped; the checks are numerical assertions on
  the boundary rows the plots used to show.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition
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


def create_wall_droplet_phase_field(Nx=64, Ny=64, radius=0.25):
    """Droplet sitting on the bottom wall so the contact line is active."""
    return create_test_phase_field(Nx=Nx, Ny=Ny, radius=radius, cx=0.5, cy=0.0)


def test_contact_angle_methods():
    """All current contact angle methods produce valid boundary updates.

    Originally checked methods ["simple", "robin", "young_laplace"]; "robin"
    and "young_laplace" no longer exist in src, so the test now covers the
    current set ["simple", "geometry_aware", "ghost_cell"].
    """
    phi, x, y = create_wall_droplet_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    methods = ["simple", "geometry_aware", "ghost_cell"]
    contact_angles = [30, 60, 90, 120]

    bottom_rows = {}
    for method in methods:
        for angle in contact_angles:
            bc = ContactAngleBoundaryCondition(
                contact_angle=angle,
                method=method,
                epsilon=0.02,
            )
            phi_bc = bc.apply(phi, dx, dy)

            phi_bc_np = np.asarray(phi_bc)
            assert phi_bc_np.shape == phi.shape
            assert np.all(np.isfinite(phi_bc_np)), f"{method} theta={angle}: non-finite phi"

            # The BC only modifies boundary rows/columns; interior must be untouched.
            np.testing.assert_array_equal(
                phi_bc_np[1:-1, 1:-1], np.asarray(phi)[1:-1, 1:-1],
                err_msg=f"{method} theta={angle}: interior modified",
            )

            if method == "ghost_cell":
                # ghost_cell returns phi unchanged by design (ghost handling
                # happens inside the phase solver).
                np.testing.assert_array_equal(phi_bc_np, np.asarray(phi))

            bottom_rows[(method, angle)] = phi_bc_np[:, 0]

    # The contact angle value must actually enter the wall update for the
    # explicit methods: hydrophilic (30 deg) and hydrophobic (120 deg) wall rows differ.
    for method in ("simple", "geometry_aware"):
        assert not np.allclose(bottom_rows[(method, 30)], bottom_rows[(method, 120)]), \
            f"{method}: contact angle does not affect the wall row"


def test_configurable_bcs():
    """Phase-field BC manager configured from a config dict applies contact angle + periodic sides."""
    config = {
        "physical_params": {
            "contact_angle": 75,
            "epsilon": 0.02
        },
        "boundary_conditions": {
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "periodic",
                "right": "periodic",
                # "robin" was removed; "simple" is the current boundary-row update method.
                "contact_angle_method": "simple"
            }
        }
    }

    bc_manager = PhaseFieldBoundaryConditions(config)

    # Configuration must be reflected in the manager (get_boundary_info was removed;
    # bc_raw is the current source of truth).
    assert bc_manager.bc_raw == {
        "top": "neumann", "bottom": "contact_angle", "left": "periodic", "right": "periodic",
    }
    assert bc_manager.contact_angle_bc.contact_angle == 75
    assert bc_manager.contact_angle_bc.method == "simple"

    phi, x, y = create_wall_droplet_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    angles = [30, 60, 90, 120]
    bottom_rows = {}
    for angle in angles:
        # update_contact_angle was removed; set the angle on the nested BC directly.
        bc_manager.contact_angle_bc.contact_angle = angle

        phi_bc = bc_manager.apply_boundary_conditions(phi, dx, dy)
        phi_bc_np = np.asarray(phi_bc)

        assert np.all(np.isfinite(phi_bc_np))

        # Periodic left/right after full application.
        np.testing.assert_allclose(phi_bc_np[0, :], phi_bc_np[-2, :], atol=1e-12)
        np.testing.assert_allclose(phi_bc_np[-1, :], phi_bc_np[1, :], atol=1e-12)

        # Neumann top.
        np.testing.assert_allclose(phi_bc_np[:, -1], phi_bc_np[:, -2], atol=1e-12)

        bottom_rows[angle] = phi_bc_np[:, 0]

    # Different configured angles must yield different wall rows.
    assert not np.allclose(bottom_rows[30], bottom_rows[120]), \
        "configured contact angle does not affect the wall row"


def test_simple_vs_geometry_aware():
    """Compare the two explicit contact angle methods.

    Originally compared "robin" vs "simple"; "robin" was removed, so this now
    compares "simple" (contact-line-masked wall update) against
    "geometry_aware" (full-wall update with surface-normal handling), which are
    the two remaining explicit methods.
    """
    phi, x, y = create_wall_droplet_phase_field()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    simple_bc = ContactAngleBoundaryCondition(contact_angle=60, method="simple", epsilon=0.02)
    geom_bc = ContactAngleBoundaryCondition(contact_angle=60, method="geometry_aware", epsilon=0.02)

    phi_simple = np.asarray(simple_bc.apply(phi, dx, dy))
    phi_geom = np.asarray(geom_bc.apply(phi, dx, dy))
    phi_np = np.asarray(phi)

    # Both methods modify the wall row of a wall-touching droplet.
    assert not np.allclose(phi_simple[:, 0], phi_np[:, 0]), "simple method did not act on the wall row"
    assert not np.allclose(phi_geom[:, 0], phi_np[:, 0]), "geometry_aware method did not act on the wall row"

    # They are different formulations and must not coincide identically.
    assert not np.allclose(phi_simple[:, 0], phi_geom[:, 0]), \
        "simple and geometry_aware methods unexpectedly produced identical wall rows"

    # Both must leave the interior untouched and stay finite.
    np.testing.assert_array_equal(phi_simple[1:-1, 1:-1], phi_np[1:-1, 1:-1])
    np.testing.assert_array_equal(phi_geom[1:-1, 1:-1], phi_np[1:-1, 1:-1])
    assert np.all(np.isfinite(phi_simple))
    assert np.all(np.isfinite(phi_geom))


if __name__ == "__main__":
    test_contact_angle_methods()
    test_configurable_bcs()
    test_simple_vs_geometry_aware()
    print("All contact angle BC tests completed successfully!")
