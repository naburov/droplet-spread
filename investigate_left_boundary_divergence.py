"""
Investigate why divergence is huge at the left boundary.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jax.numpy as jnp
from src.config.config_loader import load_config
from src.simulation.state import SimulationState
from src.simulation.two_phase import TwoPhaseSimulation
from src.numerics.finite_differences import jax_divergence, jax_dx, jax_dy


def investigate_left_boundary_divergence(config_path):
    """Investigate divergence computation at left boundary."""
    
    config = load_config(config_path)
    state = SimulationState.from_config(config)
    
    print("=" * 80)
    print("LEFT BOUNDARY DIVERGENCE INVESTIGATION")
    print("=" * 80)
    print(f"Grid: {state.Nx} x {state.Ny}, dx={state.dx:.6f}, dy={state.dy:.6f}")
    print()
    
    # Initialize simulation
    sim = TwoPhaseSimulation(config, output_dir=None)
    sim.state = state
    
    # Run predictor
    surface_tension = state.compute_surface_tension()
    solver_params = config.get('solver_params', {})
    use_rhie_chow = solver_params.get('use_rhie_chow', False)
    rhie_chow_alpha = solver_params.get('rhie_chow_alpha', 0.25)
    
    state.U = state.fluid_solver.update_velocity(
        state.U, state.P, surface_tension, state.dt,
        state.dx, state.dy, state.phi, state.geometry,
        include_gravity=state.include_gravity, use_jax=True,
        psi=sim._get_psi_for_physics(),
        use_rhie_chow=use_rhie_chow,
        rhie_chow_alpha=rhie_chow_alpha
    )
    
    # Apply velocity BCs
    state.U = state.velocity_bc.apply(state.U, state.dx, state.dy)
    
    print("Velocity after predictor + BCs:")
    print(f"  U range: [{np.min(state.U):.6f}, {np.max(state.U):.6f}]")
    print()
    
    # Check velocity at left boundary
    print("=" * 80)
    print("LEFT BOUNDARY VELOCITY ANALYSIS")
    print("=" * 80)
    
    u_left = state.U[0, :, 0]  # u at x=0
    v_left = state.U[0, :, 1]  # v at x=0
    u_left_plus1 = state.U[1, :, 0]  # u at x=1
    
    print(f"\nu at x=0 (left boundary):")
    print(f"  Range: [{np.min(u_left):.6f}, {np.max(u_left):.6f}]")
    print(f"  Mean: {np.mean(u_left):.6f}")
    print(f"  First 10 values: {u_left[:10]}")
    
    print(f"\nu at x=1 (one cell in):")
    print(f"  Range: [{np.min(u_left_plus1):.6f}, {np.max(u_left_plus1):.6f}]")
    print(f"  Mean: {np.mean(u_left_plus1):.6f}")
    print(f"  First 10 values: {u_left_plus1[:10]}")
    
    print(f"\nv at x=0 (left boundary):")
    print(f"  Range: [{np.min(v_left):.6f}, {np.max(v_left):.6f}]")
    print(f"  Mean: {np.mean(v_left):.6f}")
    print(f"  First 10 values: {v_left[:10]}")
    
    # Compute divergence manually at left boundary
    print("\n" + "=" * 80)
    print("DIVERGENCE COMPUTATION AT LEFT BOUNDARY")
    print("=" * 80)
    
    # Standard divergence
    div = jax_divergence(state.U, state.dx, state.dy)
    div_left = div[0, :]
    
    print(f"\nDivergence at x=0 (from jax_divergence):")
    print(f"  Range: [{np.min(div_left):.6f}, {np.max(div_left):.6f}]")
    print(f"  Mean: {np.mean(div_left):.6f}")
    print(f"  First 10 values: {div_left[:10]}")
    
    # Manual computation
    print(f"\nManual divergence computation:")
    print(f"  Formula: div = du/dx + dv/dy")
    
    # du/dx at x=0: one-sided difference
    # At boundary, we use: du/dx[0] = (u[1] - u[0]) / dx
    du_dx_left = (u_left_plus1 - u_left) / state.dx
    
    print(f"\n  du/dx at x=0 (one-sided):")
    print(f"    Range: [{np.min(du_dx_left):.6f}, {np.max(du_dx_left):.6f}]")
    print(f"    Mean: {np.mean(du_dx_left):.6f}")
    print(f"    First 10 values: {du_dx_left[:10]}")
    
    # dv/dy at x=0: central difference in y
    dv_dy_left = jax_dy(v_left, h=state.dy)
    
    print(f"\n  dv/dy at x=0:")
    print(f"    Range: [{np.min(dv_dy_left):.6f}, {np.max(dv_dy_left):.6f}]")
    print(f"    Mean: {np.mean(dv_dy_left):.6f}")
    print(f"    First 10 values: {dv_dy_left[:10]}")
    
    # Combined
    div_manual = du_dx_left + dv_dy_left
    
    print(f"\n  Manual div = du/dx + dv/dy:")
    print(f"    Range: [{np.min(div_manual):.6f}, {np.max(div_manual):.6f}]")
    print(f"    Mean: {np.mean(div_manual):.6f}")
    print(f"    First 10 values: {div_manual[:10]}")
    
    # Compare
    print(f"\n  Comparison with jax_divergence:")
    diff = div_manual - div_left
    print(f"    Difference: range=[{np.min(diff):.6f}, {np.max(diff):.6f}], max_abs={np.max(np.abs(diff)):.6e}")
    
    # Check what jax_divergence does
    print(f"\n" + "=" * 80)
    print("HOW jax_divergence COMPUTES AT BOUNDARY")
    print("=" * 80)
    
    # Check if jax_divergence uses one-sided or central differences at boundary
    # jax_divergence is already imported at the top
    
    # Test with a simple field
    test_U = jnp.zeros((state.Nx, state.Ny, 2))
    test_U = test_U.at[0, :, 0].set(1.0)  # u=1 at left boundary
    test_U = test_U.at[1, :, 0].set(2.0)  # u=2 one cell in
    
    test_div = jax_divergence(test_U, state.dx, state.dy)
    test_div_left = test_div[0, :]
    
    print(f"\nTest: u[0]=1, u[1]=2, v=0 everywhere")
    print(f"  Expected du/dx[0] = (2-1)/dx = {1.0/state.dx:.6f}")
    print(f"  Actual div[0] = {test_div_left[0]:.6f}")
    print(f"  Match: {np.abs(test_div_left[0] - 1.0/state.dx) < 1e-6}")
    
    # Check velocity BC profile
    print(f"\n" + "=" * 80)
    print("VELOCITY BC PROFILE CHECK")
    print("=" * 80)
    
    vel_bc = config.get("boundary_conditions", {}).get("velocity", {})
    left_bc = vel_bc.get("left", {})
    print(f"Left velocity BC: {left_bc}")
    
    # Check if profile satisfies no-slip at bottom
    print(f"\nChecking no-slip at bottom (y=0):")
    print(f"  u[0, 0] = {u_left[0]:.6f} (should be 0)")
    print(f"  v[0, 0] = {v_left[0]:.6f} (should be 0)")
    
    # Check if there's a jump at y=1
    print(f"\nChecking jump at y=1 (first cell above bottom):")
    print(f"  u[0, 0] = {u_left[0]:.6f}")
    print(f"  u[0, 1] = {u_left[1]:.6f}")
    print(f"  Jump: {u_left[1] - u_left[0]:.6f}")
    
    # This jump might cause large divergence
    if abs(u_left[1] - u_left[0]) > 0.1:
        print(f"  ⚠ Large jump detected! This will cause large du/dy contribution to divergence")
    
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Investigate left boundary divergence')
    parser.add_argument('config', type=str, help='Config file path',
                       default='configs/config_upstream_flow_only.json', nargs='?')
    args = parser.parse_args()
    
    investigate_left_boundary_divergence(args.config)
