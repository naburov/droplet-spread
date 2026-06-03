#!/usr/bin/env python3
"""
Comprehensive first-step visualization script.

Creates all visualizations used for debugging the first step of the simulation,
including:
- Phase field structure
- Surface tension distribution
- Velocity field and divergence
- Pressure distribution
- PPE analysis
- Boundary conditions
- Force balance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from config.config_loader import load_config
from simulation.state import SimulationState
from numerics.finite_differences import jax_divergence, jax_gradient, jax_norm
import jax.numpy as jnp


def create_first_step_visualizations(config_path, output_dir='debug_output'):
    """Create comprehensive first-step visualizations.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save visualizations
    """
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    state = SimulationState.from_config(config, restart_from=None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinates
    Nx, Ny = state.phi.shape
    x_coords = np.linspace(0, config['grid_params']['Lx'], Nx)
    y_coords = np.linspace(0, config['grid_params']['Ly'], Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Curvilinear (x, eta): surface at eta=0 (j=0)
    h_bottom_np = np.array(state.geometry.h_bottom) if state.geometry.has_geometry else np.zeros(Nx)
    surface_cell_indices_np = np.zeros(Nx, dtype=np.int32)  # j=0 for all x
    near_surface_mask = np.zeros((Nx, Ny), dtype=bool)
    near_surface_mask[:, 0] = True
    
    # Initial state
    phi_initial = np.array(state.phi)
    U_initial = np.array(state.U)
    P_initial = np.array(state.P)
    
    # Compute surface tension
    print("Computing surface tension...")
    surface_tension = state.compute_surface_tension()
    sf_np = np.array(surface_tension)
    sf_x = sf_np[:, :, 0]
    sf_y = sf_np[:, :, 1]
    sf_mag = np.sqrt(sf_x**2 + sf_y**2)
    
    # Compute divergence of surface tension
    sf_jax = jnp.array(surface_tension)
    div_sf = jax_divergence(sf_jax, state.dx, state.dy, state.geometry.f_1_grid)
    div_sf_np = np.array(div_sf)
    
    # Initial divergence
    divergence_initial, max_div_initial, mean_div_initial = state.fluid_solver.check_continuity(
        state.U, state.dx, state.dy, geometry=state.geometry
    )
    div_initial_np = np.array(divergence_initial)
    
    # Calculate CFL-limited dt (same logic as main sim); cap by config dt
    time_params = config.get("time_params", {})
    config_dt = time_params.get("dt", time_params.get("dt_initial", 1e-4))
    cfl_number = time_params.get("cfl_number", 0.5)
    U_mag_initial = np.sqrt(U_initial[:, :, 0]**2 + U_initial[:, :, 1]**2)
    u_max = float(U_mag_initial.max())
    if u_max < 1e-10:
        dt_cfl = np.inf  # velocity zero: no advective CFL limit
    else:
        dt_cfl = cfl_number * min(state.dx, state.dy) / u_max

    capillary_cfl = time_params.get("capillary_cfl_number", 0.02)
    epsilon = state.phase_solver.epsilon
    We1 = state.surface_tension_solver.We1
    We2 = state.surface_tension_solver.We2
    from physics.properties import jax_calculate_weber_number, jax_calculate_density
    We = jnp.array(jax_calculate_weber_number(state.phi, We1, We2))
    We_np = np.array(We)
    rho = state.compute_density()
    rho_np = np.array(rho)
    v_cap = np.sqrt(epsilon / (We_np * rho_np + 1e-10))
    v_cap_max = float(v_cap.max())
    dt_cap = capillary_cfl * min(state.dx, state.dy) / (v_cap_max + 1e-10) if v_cap_max > 1e-10 else np.inf
    dt = min(dt_cfl, dt_cap, config_dt)
    print(f"Time step: dt={dt:.6e} (velocity-CFL dt: {dt_cfl:.6e}, capillary-CFL dt: {dt_cap:.6e}, config dt: {config_dt:.6e})")
    
    # Run first step
    print("Running first step...")
    
    # Update phase field
    state.phi = state.phase_solver.update(
        state.phi, state.U, dt, state.dx, state.dy, state.geometry,
        use_jax=True, psi=state.psi
    )
    
    # Update velocity
    surface_tension = state.compute_surface_tension()
    state.U = state.fluid_solver.update_velocity(
        state.U, state.P, surface_tension, dt,
        state.dx, state.dy, state.phi, geometry=state.geometry,
        include_gravity=True, psi=state.psi
    )
    
    # Apply BC
    state.U = state.velocity_bc.apply_boundary_conditions(
        state.U, state.dx, state.dy, use_jax=True, psi=state.psi, geometry=state.geometry
    )
    
    # Check divergence after velocity update
    divergence_after_velocity, max_div_after_velocity, mean_div_after_velocity = state.fluid_solver.check_continuity(
        state.U, state.dx, state.dy, geometry=state.geometry
    )
    div_after_velocity_np = np.array(divergence_after_velocity)
    
    # PPE correction
    print("Running PPE correction...")
    from solvers.ppe_utils import correction_step, check_divergence
    ppe_bcs = config.get("solver_params", {}).get("ppe", {}).get("boundary_conditions", {})
    U_star = np.array(state.U)
    state.U, p_correction = correction_step(
        state.U, state.dx, state.dy, dt, state.geometry,
        correction_solver=state.correction_solver,
        ppe_bcs=ppe_bcs,
        U_star=U_star,
        velocity_bc_manager=state.velocity_bc
    )
    p_corr_np = np.array(p_correction)

    # Re-apply velocity BCs after PPE (inlet, no_slip, outflow) so streamlines are correct
    state.U = state.velocity_bc.apply_boundary_conditions(
        state.U, state.dx, state.dy, use_jax=True, psi=state.psi, geometry=state.geometry
    )
    
    # Check divergence after PPE
    div_after_ppe, max_div_after_ppe, mean_div_after_ppe, _ = check_divergence(
        state.U, state.dx, state.dy, state.geometry
    )
    div_after_ppe_np = np.array(div_after_ppe)
    
    # Update pressure
    state.P = state.P + p_correction
    
    # Final divergence
    divergence_final, max_div_final, mean_div_final = state.fluid_solver.check_continuity(
        state.U, state.dx, state.dy, geometry=state.geometry
    )
    div_final_np = np.array(divergence_final)
    
    # Final state
    phi_final = np.array(state.phi)
    U_final = np.array(state.U)
    P_final = np.array(state.P)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
    
    # 1. Phase field initial
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X, Y, phi_initial, levels=20, cmap='RdBu_r')
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=2, label='Surface')
    ax.set_title('Phase Field (Initial)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 2. Phase field final
    ax = fig.add_subplot(gs[0, 1])
    im = ax.contourf(X, Y, phi_final, levels=20, cmap='RdBu_r')
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=2)
    ax.set_title('Phase Field (After 1 Step)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 3. Surface tension magnitude
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(sf_mag, cmap='hot', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   norm=LogNorm(vmin=1e-6, vmax=sf_mag.max()))
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'w-', linewidth=1)
    ax.set_title('Surface Tension Magnitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 4. Divergence of surface tension
    ax = fig.add_subplot(gs[0, 3])
    vmax = np.abs(div_sf_np).max()
    im = ax.imshow(div_sf_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   vmin=-vmax, vmax=vmax)
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title('Div(∇·F_sf)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 5. Velocity magnitude initial
    ax = fig.add_subplot(gs[1, 0])
    U_mag_initial = np.sqrt(U_initial[:, :, 0]**2 + U_initial[:, :, 1]**2)
    im = ax.imshow(U_mag_initial, cmap='viridis', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'w-', linewidth=1)
    ax.set_title('Velocity Magnitude (Initial)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 6. Velocity magnitude final
    ax = fig.add_subplot(gs[1, 1])
    U_mag_final = np.sqrt(U_final[:, :, 0]**2 + U_final[:, :, 1]**2)
    im = ax.imshow(U_mag_final, cmap='viridis', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'w-', linewidth=1)
    ax.set_title('Velocity Magnitude (Final)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 7. Divergence initial
    ax = fig.add_subplot(gs[1, 2])
    vmax = np.abs(div_initial_np).max()
    im = ax.imshow(div_initial_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   norm=SymLogNorm(linthresh=1e-6, vmin=-vmax, vmax=vmax))
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title(f'Divergence (Initial)\nmax={max_div_initial:.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 8. Divergence after velocity update
    ax = fig.add_subplot(gs[1, 3])
    vmax = np.abs(div_after_velocity_np).max()
    im = ax.imshow(div_after_velocity_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   norm=SymLogNorm(linthresh=1e-6, vmin=-vmax, vmax=vmax))
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title(f'Divergence (After Velocity Update)\nmax={max_div_after_velocity:.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 9. Pressure correction
    ax = fig.add_subplot(gs[2, 0])
    vmax = np.abs(p_corr_np).max()
    im = ax.imshow(p_corr_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   vmin=-vmax, vmax=vmax)
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title('Pressure Correction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 10. Divergence after PPE
    ax = fig.add_subplot(gs[2, 1])
    vmax = np.abs(div_after_ppe_np).max()
    im = ax.imshow(div_after_ppe_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   norm=SymLogNorm(linthresh=1e-6, vmin=-vmax, vmax=vmax))
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title(f'Divergence (After PPE)\nmax={max_div_after_ppe:.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 11. Divergence final
    ax = fig.add_subplot(gs[2, 2])
    vmax = np.abs(div_final_np).max()
    im = ax.imshow(div_final_np, cmap='RdBu_r', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                   norm=SymLogNorm(linthresh=1e-6, vmin=-vmax, vmax=vmax))
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'k-', linewidth=1)
    ax.set_title(f'Divergence (Final)\nmax={max_div_final:.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 12. Pressure final
    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(P_final, cmap='viridis', origin='lower', aspect='auto',
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    if state.geometry.has_geometry:
        ax.plot(x_coords, h_bottom_np, 'w-', linewidth=1)
    ax.set_title('Pressure (Final)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    # 13. Divergence evolution
    ax = fig.add_subplot(gs[3, 0])
    steps = ['Initial', 'After Velocity', 'After PPE', 'Final']
    max_divs = [max_div_initial, max_div_after_velocity, max_div_after_ppe, max_div_final]
    mean_divs = [mean_div_initial, mean_div_after_velocity, mean_div_after_ppe, mean_div_final]
    x_pos = np.arange(len(steps))
    ax.semilogy(x_pos, np.abs(max_divs), 'ro-', linewidth=2, markersize=8, label='Max |∇·u|')
    ax.semilogy(x_pos, np.abs(mean_divs), 'bo-', linewidth=2, markersize=8, label='Mean |∇·u|')
    ax.axhline(y=1e-3, color='g', linestyle='--', alpha=0.7, label='Good (1e-3)')
    ax.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='Acceptable (1e-2)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.set_ylabel('Divergence')
    ax.set_title('Divergence Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 14. Surface tension at surface
    if state.geometry.has_geometry and len(surface_cell_indices_np) > 0:
        ax = fig.add_subplot(gs[3, 1])
        surface_x = []
        surface_sf = []
        for idx in surface_cell_indices_np:
            i, j = idx
            surface_x.append(x_coords[i])
            surface_sf.append(sf_mag[i, j])
        surface_x = np.array(surface_x)
        surface_sf = np.array(surface_sf)
        ax.plot(surface_x, surface_sf, 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Surface Tension Magnitude')
        ax.set_title('Surface Tension at Surface')
        ax.grid(True, alpha=0.3)
    
    # 15. Divergence at surface
    if state.geometry.has_geometry and len(surface_cell_indices_np) > 0:
        ax = fig.add_subplot(gs[3, 2])
        surface_x = []
        surface_div = []
        for idx in surface_cell_indices_np:
            i, j = idx
            surface_x.append(x_coords[i])
            surface_div.append(div_after_velocity_np[i, j])
        surface_x = np.array(surface_x)
        surface_div = np.array(surface_div)
        ax.plot(surface_x, surface_div, 'r-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Divergence')
        ax.set_title('Divergence at Surface (After Velocity)')
        ax.grid(True, alpha=0.3)
    
    # 16. Statistics summary
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    stats_text = f"""
    FIRST STEP STATISTICS
    
    Time Step: dt = {dt:.6e}
    
    DIVERGENCE:
    Initial:      max={max_div_initial:.2e}, mean={mean_div_initial:.2e}
    After Velocity: max={max_div_after_velocity:.2e}, mean={mean_div_after_velocity:.2e}
    After PPE:     max={max_div_after_ppe:.2e}, mean={mean_div_after_ppe:.2e}
    Final:         max={max_div_final:.2e}, mean={mean_div_final:.2e}
    
    SURFACE TENSION:
    Max magnitude: {sf_mag.max():.2e}
    Mean magnitude: {sf_mag.mean():.2e}
    Max div(F_sf): {np.abs(div_sf_np).max():.2e}
    
    VELOCITY:
    Initial max: {U_mag_initial.max():.2e}
    Final max:   {U_mag_final.max():.2e}
    
    PRESSURE:
    Initial: min={P_initial.min():.2e}, max={P_initial.max():.2e}
    Final:   min={P_final.min():.2e}, max={P_final.max():.2e}
    Correction: min={p_corr_np.min():.2e}, max={p_corr_np.max():.2e}
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 17-20. Cross-sections at different x positions
    x_sections = [Nx//4, Nx//2, 3*Nx//4]
    for idx, x_section in enumerate(x_sections):
        ax = fig.add_subplot(gs[4, idx])
        ax.plot(y_coords, phi_initial[x_section, :], 'b-', linewidth=2, label='φ initial', alpha=0.7)
        ax.plot(y_coords, phi_final[x_section, :], 'r-', linewidth=2, label='φ final', alpha=0.7)
        if state.geometry.has_geometry:
            h_val = h_bottom_np[x_section]
            ax.axhline(y=h_val, color='k', linestyle='--', linewidth=1, label='Surface')
        ax.set_xlabel('y')
        ax.set_ylabel('Phase Field')
        ax.set_title(f'Cross-section at x={x_coords[x_section]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 21. Velocity cross-section
    ax = fig.add_subplot(gs[4, 3])
    x_center = Nx // 2
    ax.plot(y_coords, U_initial[x_center, :, 0], 'b-', linewidth=2, label='u initial', alpha=0.7)
    ax.plot(y_coords, U_final[x_center, :, 0], 'r-', linewidth=2, label='u final', alpha=0.7)
    if state.geometry.has_geometry:
        h_val = h_bottom_np[x_center]
        ax.axhline(y=h_val, color='k', linestyle='--', linewidth=1, label='Surface')
    ax.set_xlabel('y')
    ax.set_ylabel('u velocity')
    ax.set_title(f'Velocity Cross-section at x={x_coords[x_center]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 22-24. Divergence cross-sections
    for idx, x_section in enumerate(x_sections):
        ax = fig.add_subplot(gs[5, idx])
        ax.plot(y_coords, div_initial_np[x_section, :], 'b-', linewidth=2, label='Initial', alpha=0.7)
        ax.plot(y_coords, div_after_velocity_np[x_section, :], 'g-', linewidth=2, label='After Velocity', alpha=0.7)
        ax.plot(y_coords, div_final_np[x_section, :], 'r-', linewidth=2, label='Final', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        if state.geometry.has_geometry:
            h_val = h_bottom_np[x_section]
            ax.axhline(y=h_val, color='k', linestyle=':', linewidth=1, label='Surface')
        ax.set_xlabel('y')
        ax.set_ylabel('Divergence')
        ax.set_title(f'Divergence at x={x_coords[x_section]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 25. Summary
    ax = fig.add_subplot(gs[5, 3])
    ax.axis('off')
    summary_text = f"""
    DIAGNOSTIC SUMMARY
    
    ✓ Phase field updated
    ✓ Velocity updated
    ✓ PPE correction applied
    ✓ Boundary conditions applied
    
    Divergence reduction:
    {max_div_initial:.2e} → {max_div_final:.2e}
    ({100*(1 - max_div_final/max_div_initial):.1f}% reduction)
    
    Status: {'✓ GOOD' if max_div_final < 1e-2 else '⚠ WARNING' if max_div_final < 1e-1 else '✗ POOR'}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if max_div_final < 1e-2 else 'lightyellow' if max_div_final < 1e-1 else 'lightcoral', alpha=0.8))
    
    plt.suptitle('First Step Comprehensive Diagnostics', fontsize=16, y=0.995)
    
    output_path = os.path.join(output_dir, 'first_step_comprehensive.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"First step visualizations saved to: {output_path}")
    
    plt.close()
    
    return {
        'max_div_initial': max_div_initial,
        'max_div_after_velocity': max_div_after_velocity,
        'max_div_after_ppe': max_div_after_ppe,
        'max_div_final': max_div_final,
        'dt': dt,
        'sf_mag_max': sf_mag.max(),
        'div_sf_max': np.abs(div_sf_np).max()
    }


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python first_step_visualization.py <config_path> [output_dir]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'debug_output'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} does not exist!")
        sys.exit(1)
    
    stats = create_first_step_visualizations(config_path, output_dir)
    
    print("\n=== FIRST STEP DIAGNOSTICS SUMMARY ===")
    print(f"Max divergence: {stats['max_div_initial']:.2e} → {stats['max_div_final']:.2e}")
    print(f"Time step: {stats['dt']:.6e}")
    print(f"Surface tension max: {stats['sf_mag_max']:.2e}")
    print(f"Div(F_sf) max: {stats['div_sf_max']:.2e}")


if __name__ == "__main__":
    main()

