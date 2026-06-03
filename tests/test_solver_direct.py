#!/usr/bin/env python3
"""
Test the solver directly to see if it's working.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from solvers.sparse_solver import SparseSolverWrapper
from config.config_loader import load_config


def test_solver_direct():
    """Test the solver directly."""
    print("=" * 60)
    print("DIRECT SOLVER TEST")
    print("=" * 60)
    
    Nx, Ny = 8, 8
    dx = dy = 0.1
    
    # Create solver
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    solver.set_top_boundary_condition("neumann")
    solver.set_bottom_boundary_condition("neumann")
    solver.set_left_boundary_condition("neumann")
    solver.set_right_boundary_condition("neumann")
    solver.create_sparse_matrix()
    
    print(f"Solver created successfully")
    print(f"Solver attributes: {dir(solver)}")
    
    # Test with simple RHS
    rhs = np.ones((Nx, Ny))
    print(f"RHS range: {rhs.min():.6f} / {rhs.max():.6f}")
    
    try:
        solver.set_rhs(rhs)
        solver.solve()
        solution = solver.get_solution()
        print(f"✓ Solver succeeded")
        print(f"Solution range: {solution.min():.6f} / {solution.max():.6f}")
        print(f"Solution shape: {solution.shape}")
    except Exception as e:
        print(f"✗ Solver failed: {e}")


def test_solver_with_divergence():
    """Test solver with actual divergence from gravity case."""
    print(f"\n" + "=" * 60)
    print("SOLVER WITH GRAVITY DIVERGENCE TEST")
    print("=" * 60)
    
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01
    
    # Create gravity-like velocity field
    U = jnp.stack([
        jnp.zeros((Nx, Ny)),
        jnp.ones((Nx, Ny)) * -0.1
    ], axis=-1)
    
    # Calculate divergence
    from numerics.finite_differences import jax_divergence
    div = jax_divergence(U, dx, dy)
    print(f"Velocity field: {U.min():.6f} / {U.max():.6f}")
    print(f"Divergence: {div.min():.6f} / {div.max():.6f}")
    
    # Create solver
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
    solver.set_top_boundary_condition("neumann")
    solver.set_bottom_boundary_condition("neumann")
    solver.set_left_boundary_condition("neumann")
    solver.set_right_boundary_condition("neumann")
    solver.create_sparse_matrix()
    
    # RHS for PPE
    rhs = div / dt
    rhs = rhs - np.mean(rhs)  # Subtract mean
    print(f"RHS range: {rhs.min():.6f} / {rhs.max():.6f}")
    
    try:
        solver.set_rhs(rhs)
        solver.solve()
        solution = solver.get_solution()
        print(f"✓ Solver succeeded")
        print(f"Solution range: {solution.min():.6f} / {solution.max():.6f}")
    except Exception as e:
        print(f"✗ Solver failed: {e}")


def main():
    """Run solver tests."""
    test_solver_direct()
    test_solver_with_divergence()
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If the solver works directly, then the issue is in how")
    print("the PPE function is calling the solver or handling the results.")


if __name__ == "__main__":
    main()
