#!/usr/bin/env python3
"""
Test PPE stability with different velocity fields.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from solvers.projection_methods import ppe
from physics.fluid_dynamics import check_continuity


def test_ppe_stability():
    """Test PPE stability with different velocity fields."""
    print("=" * 60)
    print("PPE STABILITY TEST")
    print("=" * 60)
    
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01
    
    # Test cases
    test_cases = [
        ("Zero field", jnp.zeros((Nx, Ny, 2))),
        ("Small constant", jnp.ones((Nx, Ny, 2)) * 0.01),
        ("Gravity-like", jnp.stack([
            jnp.zeros((Nx, Ny)),
            jnp.ones((Nx, Ny)) * -0.1
        ], axis=-1)),
        ("Random small", jnp.array(np.random.normal(0, 0.01, (Nx, Ny, 2)))),
    ]
    
    for name, U in test_cases:
        print(f"\nTesting: {name}")
        print(f"  Input velocity range: {U.min():.6f} / {U.max():.6f}")
        
        # Check divergence before PPE
        div_before, max_div_before, mean_div_before = check_continuity(U, dx, dy)
        print(f"  Divergence before: Max = {max_div_before:.6f}, Mean = {mean_div_before:.6f}")
        
        # Try PPE
        try:
            U_corrected = ppe(U, dx, dy, dt)
            div_after, max_div_after, mean_div_after = check_continuity(U_corrected, dx, dy)
            print(f"  ✓ PPE succeeded")
            print(f"  Output velocity range: {U_corrected.min():.6f} / {U_corrected.max():.6f}")
            print(f"  Divergence after: Max = {max_div_after:.6f}, Mean = {mean_div_after:.6f}")
            
            # Check if velocity changed significantly
            velocity_change = jnp.linalg.norm(U_corrected - U)
            print(f"  Velocity change magnitude: {velocity_change:.6f}")
            
        except Exception as e:
            print(f"  ✗ PPE failed: {e}")


def test_ppe_with_solver():
    """Test PPE with a proper solver."""
    print(f"\n" + "=" * 60)
    print("PPE WITH SOLVER TEST")
    print("=" * 60)
    
    Nx, Ny = 8, 8
    dx = dy = 0.1
    dt = 0.01
    
    # Create a simple velocity field
    U = jnp.stack([
        jnp.zeros((Nx, Ny)),
        jnp.ones((Nx, Ny)) * -0.1
    ], axis=-1)
    
    print(f"Input velocity: {U.min():.6f} / {U.max():.6f}")
    
    # Check divergence
    div, max_div, mean_div = check_continuity(U, dx, dy)
    print(f"Divergence: Max = {max_div:.6f}, Mean = {mean_div:.6f}")
    
    # Try PPE with solver
    try:
        from solvers.sparse_solver import SparseSolverWrapper
        solver = SparseSolverWrapper(Nx, Ny, dx, dy, "pyamg")
        solver.set_top_boundary_condition("neumann")
        solver.set_bottom_boundary_condition("neumann")
        solver.set_left_boundary_condition("neumann")
        solver.set_right_boundary_condition("neumann")
        solver.create_sparse_matrix()
        
        U_corrected = ppe(U, dx, dy, dt, correction_solver=solver)
        
        div_after, max_div_after, mean_div_after = check_continuity(U_corrected, dx, dy)
        print(f"✓ PPE with solver succeeded")
        print(f"Output velocity: {U_corrected.min():.6f} / {U_corrected.max():.6f}")
        print(f"Divergence after: Max = {max_div_after:.6f}, Mean = {mean_div_after:.6f}")
        
    except Exception as e:
        print(f"✗ PPE with solver failed: {e}")


def main():
    """Run PPE stability tests."""
    test_ppe_stability()
    test_ppe_with_solver()
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The issue is likely in the PPE solver implementation.")
    print("The gravity term itself is not creating divergence,")
    print("but the PPE solver is becoming numerically unstable.")


if __name__ == "__main__":
    main()
