"""Tests for the BDF2 + Newton Cahn-Hilliard stepper (phase_update_mode=bdf2_ch)."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from physics.phase_field import PhaseFieldSolverGhostCell
from simulation.geometry import Geometry

BINODAL = 0.9993286


def make_config(**solver_overrides):
    solver_params = {
        "phase_field_solver": "ghost_cell",
        "phase_update_mode": "bdf2_ch",
        "phase_potential": "flory_huggins",
        "phase_log_theta": 0.25,
        "phase_log_theta_c": 1.0,
        "phase_log_delta": 1e-6,
        "use_degenerate_mobility": True,
        "degenerate_mobility_power": 2.0,
        "degenerate_mobility_blend": 0.0,
    }
    solver_params.update(solver_overrides)
    return {
        "solver_params": solver_params,
        "boundary_conditions": {"phase_field": {}},
    }


def droplet(n, eps=0.06, radius=0.25):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    return jnp.asarray(BINODAL * np.tanh((r - radius) / (np.sqrt(2.0) * eps)))


def make_solver(eps=0.06, **overrides):
    return PhaseFieldSolverGhostCell(
        Pe=20.0, epsilon=eps, contact_angle=90, config=make_config(**overrides)
    )


def run_steps(solver, phi, n_steps, dt, n):
    geometry = Geometry.flat(n, n)
    U = jnp.zeros((n, n, 2))
    h = 1.0 / n
    for _ in range(n_steps):
        phi = solver.update(phi, U, dt, h, h, geometry, use_jax=True)
    return phi


def test_steps_remain_finite_and_bounded():
    n = 32
    phi = run_steps(make_solver(), droplet(n), 10, 1e-5, n)
    a = np.asarray(phi)
    assert np.all(np.isfinite(a))
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_mass_conservation_without_advection():
    n = 32
    phi0 = droplet(n)
    phi = run_steps(make_solver(), phi0, 25, 1e-5, n)
    # conservative flux form with zero boundary fluxes: mean phi preserved
    np.testing.assert_allclose(
        float(jnp.mean(phi)), float(jnp.mean(phi0)), rtol=0, atol=1e-8
    )


def test_second_order_convergence_in_dt():
    n = 32
    phi0 = droplet(n)
    t_final = 8e-5

    def integrate(dt):
        solver = make_solver()
        return np.asarray(run_steps(solver, phi0, int(round(t_final / dt)), dt, n))

    ref = integrate(5e-7)
    err_coarse = np.max(np.abs(integrate(8e-6) - ref))
    err_fine = np.max(np.abs(integrate(4e-6) - ref))
    order = np.log2(err_coarse / err_fine)
    # first step is backward Euler, so the observed order sits slightly
    # below the asymptotic 2; anything clearly above 1.5 confirms BDF2
    assert order > 1.5, f"observed order {order:.2f} (errors {err_coarse:.2e}/{err_fine:.2e})"


def test_newton_nonconvergence_raises():
    n = 32
    solver = make_solver(phase_bdf2_newton_maxiter=1, phase_bdf2_newton_tol=1e-14)
    with pytest.raises(RuntimeError, match="did not converge"):
        run_steps(solver, droplet(n), 1, 5e-3, n)
