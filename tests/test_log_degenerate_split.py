"""Bounds and conservation tests for the log-entropy degenerate convex split.

The Flory-Huggins entropy diverges at phi = +-1, so the implicit convex split
must keep phi strictly inside (-1, 1) without any clip-based enforcement in
the physics (properties interpolation evaluates phi as-is).
"""

import os
import sys

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.geometry import Geometry
from physics.phase_field import (
    PhaseFieldSolverGhostCell,
    jax_degenerate_mobility,
    jax_free_energy_derivative,
)


def _make_config(nx, ny):
    return {
        "physical_params": {
            "rho1": 0.001225,
            "rho2": 1.0,
            "epsilon": 0.04,
            "contact_angle": 120,
        },
        "grid_params": {"Lx": 1.0, "Ly": 1.0, "Nx": nx, "Ny": ny},
        "boundary_conditions": {
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "ghost_cell",
                "contact_angle_ghost_law": "wall_energy",
                "contact_angle_full_wall": True,
            },
            "chemical_potential": {
                "top": "zero_flux",
                "bottom": "zero_flux",
                "left": "zero_flux",
                "right": "zero_flux",
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "impermeable",
                "right": "impermeable",
            },
        },
        "solver_params": {
            "phase_field_solver": "ghost_cell",
            "phase_update_mode": "semi_implicit_ch",
            "semi_implicit_contact_split": "implicit_wall_energy",
            "use_degenerate_mobility": True,
            "degenerate_mobility_power": 2.0,
            "degenerate_mobility_imex": True,
            "degenerate_mobility_imex_mref": 0.0,
            "phase_potential": "flory_huggins",
            "phase_log_theta": 0.25,
            "phase_log_theta_c": 1.0,
            "phase_log_delta": 1e-6,
            "phase_convex_split": "log_entropy",
            "phase_convex_split_maxiter": 8,
            "phase_convex_split_tol": 1e-7,
            "phase_convex_split_damping": 0.8,
        },
    }


def _droplet_phi(nx, ny, radius=0.25, eps=0.04):
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + Y**2)
    return jnp.asarray(np.tanh((r - radius) / (np.sqrt(2.0) * eps)))


def test_log_degenerate_split_keeps_phi_bounded():
    nx = ny = 48
    config = _make_config(nx, ny)
    solver = PhaseFieldSolverGhostCell(
        Pe=20.0, epsilon=0.04, contact_angle=120, config=config
    )
    geometry = Geometry.flat(nx, ny)
    phi = _droplet_phi(nx, ny)
    dx = dy = 1.0 / nx
    dt = 5e-4

    # Shear-like velocity to exercise advection + wetting together.
    U = jnp.zeros((nx, ny, 2))
    U = U.at[:, :, 0].set(jnp.linspace(0.0, 1.0, ny)[None, :])

    mass0 = float(jnp.sum(phi))
    delta = config["solver_params"]["phase_log_delta"]
    for _ in range(20):
        phi = solver.update(phi, U, dt, dx, dy, geometry, use_jax=True)
        assert bool(jnp.all(jnp.isfinite(phi)))

    # The convex split keeps phi within [-1+delta, 1-delta]; the subsequent
    # global phase-sum preservation may shift it by O(1e-9). Allow that slack
    # but require phi to stay strictly inside the physical interval.
    phi_max = float(jnp.max(phi))
    phi_min = float(jnp.min(phi))
    assert phi_max <= 1.0 - delta + 1e-7, f"phi max {phi_max} escaped bounds"
    assert phi_min >= -1.0 + delta - 1e-7, f"phi min {phi_min} escaped bounds"
    assert phi_max < 1.0 and phi_min > -1.0

    # Phase sum preserved by the conservative split (advection BCs allow small flux).
    mass1 = float(jnp.sum(phi))
    assert abs(mass1 - mass0) / (nx * ny) < 5e-3


def test_degenerate_mobility_vanishes_beyond_pure_phases():
    phi = jnp.asarray([-1.2, -1.0, 0.0, 1.0, 1.2])
    m = np.asarray(jax_degenerate_mobility(phi, mobility_power=2.0))
    assert m[0] == 0.0 and m[1] == 0.0 and m[3] == 0.0 and m[4] == 0.0
    assert m[2] == pytest.approx(1.0)
    assert np.all(m >= 0.0)


def test_log_free_energy_derivative_restoring_near_bounds():
    phi = jnp.asarray([-1.0 + 2e-6, -0.5, 0.0, 0.5, 1.0 - 2e-6])
    mu = np.asarray(
        jax_free_energy_derivative(phi, potential_code=1, log_theta=0.25,
                                   log_theta_c=1.0, log_delta=1e-6)
    )
    assert np.all(np.isfinite(mu))
    # Odd symmetry and a restoring (positive) chemical potential approaching
    # phi = +1: the entropy term dominates the concave -theta_c*phi part.
    assert mu[2] == pytest.approx(0.0, abs=1e-12)
    assert mu[-1] > 0.0 and mu[0] < 0.0
    np.testing.assert_allclose(mu, -mu[::-1], atol=1e-12)


def test_clip_config_keys_are_rejected():
    config = _make_config(32, 32)
    config["solver_params"]["phase_potential_clip"] = 1.2
    with pytest.raises(ValueError, match="removed"):
        PhaseFieldSolverGhostCell(Pe=20.0, epsilon=0.04, contact_angle=120, config=config)
