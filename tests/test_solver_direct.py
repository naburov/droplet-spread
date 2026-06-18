#!/usr/bin/env python3
"""
Test the sparse solver directly to see if it's working.

Updated to current src/ APIs:
- numerics.finite_differences.jax_divergence now requires the terrain gradient
  grid f_1_grid (zeros for a flat surface).
- The original try/except-with-print blocks were converted into hard
  assertions so solver failures actually fail the test.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from solvers.sparse_solver import SparseSolverWrapper


def _make_neumann_solver(Nx, Ny, dx, dy):
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    solver.set_top_boundary_condition("neumann")
    solver.set_bottom_boundary_condition("neumann")
    solver.set_left_boundary_condition("neumann")
    solver.set_right_boundary_condition("neumann")
    solver.create_sparse_matrix()
    return solver


def test_solver_direct():
    """The sparse solver solves a simple Poisson problem and returns a finite field."""
    Nx, Ny = 8, 8
    dx = dy = 0.1

    solver = _make_neumann_solver(Nx, Ny, dx, dy)

    # Simple RHS (mean-zero so the all-Neumann problem is solvable)
    rhs = np.ones((Nx, Ny))
    rhs = rhs - np.mean(rhs)

    solver.set_rhs(rhs)
    solver.solve()
    solution = solver.get_solution()

    solution = np.asarray(solution)
    assert solution.shape == (Nx, Ny)
    assert np.all(np.isfinite(solution)), "solver returned non-finite solution"


def test_solver_with_divergence():
    """Solver handles the (near-zero) divergence RHS of a gravity-like constant field.

    The point of the original debug script: a uniform gravity-induced velocity
    field is divergence-free, so the PPE RHS it generates is ~zero and the
    solver must return a benign (finite, ~constant) correction rather than
    blowing up.
    """
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01

    # Create gravity-like velocity field
    U = jnp.stack([
        jnp.zeros((Nx, Ny)),
        jnp.ones((Nx, Ny)) * -0.1
    ], axis=-1)

    # Calculate divergence (flat surface: f_1_grid = zeros)
    from numerics.finite_differences import jax_divergence
    f_1_grid = jnp.zeros((Nx, Ny))
    div = jax_divergence(U, dx, dy, f_1_grid)

    # A constant velocity field must be (numerically) divergence-free.
    np.testing.assert_allclose(np.asarray(div), 0.0, atol=1e-12,
                               err_msg="uniform gravity-like field must be divergence-free")

    solver = _make_neumann_solver(Nx, Ny, dx, dy)

    # RHS for PPE
    rhs = np.asarray(div) / dt
    rhs = rhs - np.mean(rhs)  # Subtract mean

    solver.set_rhs(rhs)
    solver.solve()
    solution = np.asarray(solver.get_solution())

    assert np.all(np.isfinite(solution)), "solver returned non-finite solution"
    # Zero RHS with Neumann BCs: the pressure correction is constant (zero up to gauge),
    # i.e. it must not introduce any spurious velocity correction.
    assert float(np.max(solution) - np.min(solution)) < 1e-8, \
        "zero-divergence RHS must yield a constant (gauge) pressure correction"


if __name__ == "__main__":
    test_solver_direct()
    test_solver_with_divergence()
    print("Direct solver tests passed.")
