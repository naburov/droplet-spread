"""
Investigate what happens during the first PPE step, especially at the left boundary.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.config.config_loader import load_config
from src.simulation.state import SimulationState
from src.simulation.two_phase import TwoPhaseSimulation
from src.numerics.finite_differences import jax_divergence
from src.solvers.ppe_bc_derivation import derive_ppe_bcs_from_config


def investigate_first_ppe(config_path):
    """Investigate the first PPE step in detail."""
    
    config = load_config(config_path)
    state = SimulationState.from_config(config)
    
    print("=" * 80)
    print("FIRST PPE INVESTIGATION")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Grid: {state.Nx} x {state.Ny}, dx={state.dx:.6f}, dy={state.dy:.6f}")
    print()
    
    # Initialize simulation
    sim = TwoPhaseSimulation(config, output_dir=None)
    sim.state = state
    
    # Step 0: Initial state
    print("=" * 80)
    print("STEP 0: INITIAL STATE")
    print("=" * 80)
    
    print(f"\nInitial velocity:")
    print(f"  U range: [{np.min(state.U):.6f}, {np.max(state.U):.6f}]")
    print(f"  U mean: {np.mean(state.U):.6f}")
    
    print(f"\nInitial pressure:")
    print(f"  P range: [{np.min(state.P):.6f}, {np.max(state.P):.6f}]")
    print(f"  P mean: {np.mean(state.P):.6f}")
    
    # Check left boundary
    print(f"\nLeft boundary (x=0) pressure:")
    p_left = state.P[0, :]
    print(f"  Range: [{np.min(p_left):.6f}, {np.max(p_left):.6f}]")
    print(f"  Mean: {np.mean(p_left):.6f}")
    print(f"  First 10 values: {p_left[:10]}")
    
    # Check for checkerboard pattern in initial pressure
    print(f"\nChecking for checkerboard in initial pressure...")
    p_checkerboard = check_checkerboard_pattern(state.P)
    print(f"  Checkerboard amplitude: {p_checkerboard:.6e}")
    
    # Run predictor step
    print("\n" + "=" * 80)
    print("PREDICTOR STEP")
    print("=" * 80)
    
    surface_tension = state.compute_surface_tension()
    
    solver_params = config.get('solver_params', {})
    use_rhie_chow = solver_params.get('use_rhie_chow', False)
    rhie_chow_alpha = solver_params.get('rhie_chow_alpha', 0.25)
    
    U_before_predictor = state.U.copy()
    state.U = state.fluid_solver.update_velocity(
        state.U, state.P, surface_tension, state.dt,
        state.dx, state.dy, state.phi, state.geometry,
        include_gravity=state.include_gravity, use_jax=True,
        psi=sim._get_psi_for_physics(),
        use_rhie_chow=use_rhie_chow,
        rhie_chow_alpha=rhie_chow_alpha
    )
    U_after_predictor = state.U.copy()
    
    print(f"\nVelocity after predictor (before BCs):")
    print(f"  U range: [{np.min(state.U):.6f}, {np.max(state.U):.6f}]")
    print(f"  Max change: {np.max(np.abs(U_after_predictor - U_before_predictor)):.6e}")
    
    # Apply velocity BCs
    state.U = state.velocity_bc.apply(state.U, state.dx, state.dy)
    
    print(f"\nVelocity after BCs:")
    print(f"  U range: [{np.min(state.U):.6f}, {np.max(state.U):.6f}]")
    
    # Check divergence
    print("\n" + "=" * 80)
    print("DIVERGENCE CHECK (BEFORE PPE)")
    print("=" * 80)
    
    div = jax_divergence(state.U, state.dx, state.dy)
    max_div = np.max(np.abs(div))
    mean_div = np.mean(np.abs(div))
    
    print(f"  Max divergence: {max_div:.6e}")
    print(f"  Mean divergence: {mean_div:.6e}")
    
    # Check divergence at left boundary
    div_left = div[0, :]
    print(f"\n  Divergence at left boundary (x=0):")
    print(f"    Range: [{np.min(div_left):.6f}, {np.max(div_left):.6f}]")
    print(f"    Mean: {np.mean(div_left):.6f}")
    print(f"    First 10 values: {div_left[:10]}")
    
    # Check for checkerboard in divergence
    div_checkerboard = check_checkerboard_pattern(div)
    print(f"\n  Checkerboard amplitude in divergence: {div_checkerboard:.6e}")
    
    if max_div > 0.05 or mean_div > 0.05:
        print("\n  → PPE needed!")
        
        # Store state before PPE
        U_before_ppe = state.U.copy()
        P_before_ppe = state.P.copy()
        
        # Get PPE BCs
        ppe_bcs = derive_ppe_bcs_from_config(config)
        print(f"\n  PPE BCs: {ppe_bcs}")
        
        # Check what BCs should be at left boundary
        print(f"\n  Left boundary analysis:")
        vel_bc = config.get("boundary_conditions", {}).get("velocity", {})
        left_vel_type = vel_bc.get("left", {})
        if isinstance(left_vel_type, dict):
            left_vel_type = left_vel_type.get("type", "")
        else:
            left_vel_type = str(left_vel_type)
        
        print(f"    Velocity BC type: {left_vel_type}")
        print(f"    PPE BC type: {ppe_bcs.get('left', 'unknown')}")
        
        # Run PPE
        print("\n" + "=" * 80)
        print("RUNNING FIRST PPE")
        print("=" * 80)
        
        from src.solvers.ppe import ppe_solve
        
        ppe_settings = solver_params.get('ppe', {})
        max_iterations = ppe_settings.get('max_iterations', 1000)
        under_relaxation = ppe_settings.get('under_relaxation', 1.0)
        
        print(f"\nPPE settings:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Under-relaxation: {under_relaxation}")
        
        state.U, ppe_info = ppe_solve(
            state.U, state.dx, state.dy, state.dt,
            state.geometry,
            correction_solver=state.correction_solver,
            velocity_bc_manager=state.velocity_bc,
            ppe_bcs=ppe_bcs,
            psi=sim._get_psi_for_physics(),
            max_iterations=max_iterations,
            under_relaxation=under_relaxation,
        )
        
        U_after_ppe = state.U.copy()
        
        print(f"\nPPE results:")
        print(f"  Iterations: {ppe_info.get('iterations', 0)}")
        print(f"  Div before: max={ppe_info.get('div_before_max', 0):.6e}, mean={ppe_info.get('div_before_mean', 0):.6e}")
        print(f"  Div after: max={ppe_info.get('div_after_max', 0):.6e}, mean={ppe_info.get('div_after_mean', 0):.6e}")
        
        # Check velocity change
        print(f"\nVelocity change from PPE:")
        U_change = U_after_ppe - U_before_ppe
        print(f"  Max absolute change: {np.max(np.abs(U_change)):.6e}")
        print(f"  Mean absolute change: {np.mean(np.abs(U_change)):.6e}")
        
        # Check left boundary specifically
        print(f"\nLeft boundary (x=0) velocity change:")
        u_change_left = U_change[0, :, 0]
        v_change_left = U_change[0, :, 1]
        print(f"  u change: range=[{np.min(u_change_left):.6f}, {np.max(u_change_left):.6f}], max_abs={np.max(np.abs(u_change_left)):.6e}")
        print(f"  v change: range=[{np.min(v_change_left):.6f}, {np.max(v_change_left):.6f}], max_abs={np.max(np.abs(v_change_left)):.6e}")
        print(f"  First 10 u changes: {u_change_left[:10]}")
        
        # Check for checkerboard in velocity change
        print(f"\nChecking for checkerboard in velocity change...")
        u_change_checkerboard = check_checkerboard_pattern(U_change[..., 0])
        v_change_checkerboard = check_checkerboard_pattern(U_change[..., 1])
        print(f"  Checkerboard amplitude in u change: {u_change_checkerboard:.6e}")
        print(f"  Checkerboard amplitude in v change: {v_change_checkerboard:.6e}")
        
        # Check final velocity
        print(f"\nFinal velocity after PPE:")
        print(f"  U range: [{np.min(state.U):.6f}, {np.max(state.U):.6f}]")
        
        print(f"\nLeft boundary (x=0) final velocity:")
        u_left_final = state.U[0, :, 0]
        v_left_final = state.U[0, :, 1]
        print(f"  u: range=[{np.min(u_left_final):.6f}, {np.max(u_left_final):.6f}], mean={np.mean(u_left_final):.6f}")
        print(f"  v: range=[{np.min(v_left_final):.6f}, {np.max(v_left_final):.6f}], mean={np.mean(v_left_final):.6f}")
        print(f"  First 10 u values: {u_left_final[:10]}")
        
        # Check for checkerboard in final velocity
        print(f"\nChecking for checkerboard in final velocity...")
        u_final_checkerboard = check_checkerboard_pattern(state.U[..., 0])
        v_final_checkerboard = check_checkerboard_pattern(state.U[..., 1])
        print(f"  Checkerboard amplitude in u: {u_final_checkerboard:.6e}")
        print(f"  Checkerboard amplitude in v: {v_final_checkerboard:.6e}")
        
        # Check pressure correction if available
        print(f"\n" + "=" * 80)
        print("PRESSURE CORRECTION ANALYSIS")
        print("=" * 80)
        
        # Try to get pressure correction from the solver
        # This might require modifying the PPE solver to return it
        print("  (Pressure correction not directly available, would need to modify PPE solver)")
        
    else:
        print("\n  → PPE not needed (divergence is low)")
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


def check_checkerboard_pattern(field):
    """Check for checkerboard pattern in a 2D field.
    
    Returns the amplitude of the checkerboard mode.
    """
    Nx, Ny = field.shape[:2]
    
    # Checkerboard pattern: (-1)^(i+j)
    checkerboard = np.array([[(1 if (i+j) % 2 == 0 else -1) 
                              for j in range(Ny)] 
                             for i in range(Nx)])
    
    # Project field onto checkerboard mode
    field_flat = field.flatten()
    checkerboard_flat = checkerboard.flatten()
    
    # Normalize
    checkerboard_norm = np.linalg.norm(checkerboard_flat)
    if checkerboard_norm > 0:
        amplitude = np.abs(np.dot(field_flat, checkerboard_flat)) / checkerboard_norm
    else:
        amplitude = 0.0
    
    return amplitude


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Investigate first PPE step')
    parser.add_argument('config', type=str, help='Config file path',
                       default='configs/config_upstream_flow_only.json', nargs='?')
    args = parser.parse_args()
    
    investigate_first_ppe(args.config)
