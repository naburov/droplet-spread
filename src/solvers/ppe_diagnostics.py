"""
Diagnostic utilities for PPE debugging.
"""

import numpy as np
import os

try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz


def compute_stream_function(U, dx, dy):
    """Compute stream function from velocity field.
    
    Stream function ψ satisfies:
        u = ∂ψ/∂y
        v = -∂ψ/∂x
    
    We integrate from bottom (y=0) where ψ=0:
        ψ(x,y) = ∫[0 to y] u(x,η) dη
    
    Args:
        U: Velocity field (Nx, Ny, 2)
        dx, dy: Grid spacing
    
    Returns:
        psi: Stream function (Nx, Ny)
    """
    U_np = np.array(U) if not isinstance(U, np.ndarray) else U
    u = U_np[..., 0]  # x-component
    v = U_np[..., 1]  # y-component
    
    Nx, Ny = u.shape
    psi = np.zeros((Nx, Ny))
    
    # Integrate u along y-direction from bottom (j=0)
    y_coords = np.arange(Ny) * dy
    
    for i in range(Nx):
        # Integrate u[i, :] from y=0
        psi[i, :] = cumtrapz(u[i, :], y_coords, initial=0.0)
    
    return psi


def dump_ppe_diagnostics(U_before, U_after, dx, dy, step, iteration, output_dir):
    """Dump velocity and streamline diagnostics before and after PPE.
    
    Args:
        U_before: Velocity before PPE (Nx, Ny, 2)
        U_after: Velocity after PPE (Nx, Ny, 2)
        dx, dy: Grid spacing
        step: Simulation step number
        iteration: PPE iteration number (or -1 for final)
        output_dir: Output directory for diagnostics
    """
    import matplotlib.pyplot as plt
    
    # Create diagnostics directory
    diag_dir = os.path.join(output_dir, 'ppe_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    
    U_before_np = np.array(U_before) if not isinstance(U_before, np.ndarray) else U_before
    U_after_np = np.array(U_after) if not isinstance(U_after, np.ndarray) else U_after
    
    u_before = U_before_np[..., 0]
    v_before = U_before_np[..., 1]
    u_after = U_after_np[..., 0]
    v_after = U_after_np[..., 1]
    
    Nx, Ny = u_before.shape
    x_coords = np.arange(Nx) * dx
    y_coords = np.arange(Ny) * dy
    
    # Compute stream functions
    psi_before = compute_stream_function(U_before_np, dx, dy)
    psi_after = compute_stream_function(U_after_np, dx, dy)
    
    # Key diagnostic: u velocity just above bottom (y=dy, j=1)
    u_bottom_before = u_before[:, 1]  # j=1 is first fluid cell above bottom
    u_bottom_after = u_after[:, 1]
    
    # Save velocity profile at bottom
    if iteration >= 0:
        suffix = f"_step{step:06d}_iter{iteration:04d}"
    else:
        suffix = f"_step{step:06d}_after_ppe"  # After complete PPE cycle
    
    # 1. Plot u at y=dy (just above bottom)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    ax = axes[0]
    ax.plot(x_coords, u_bottom_before, 'b-', label='Before PPE', linewidth=2)
    ax.plot(x_coords, u_bottom_after, 'r-', label='After PPE', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('u at y=dy')
    if iteration >= 0:
        title_suffix = f" (step {step}, iter {iteration})"
    else:
        title_suffix = f" (step {step}, after PPE cycle)"
    ax.set_title(f'Velocity u just above bottom{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight negative (reverse flow) regions
    reverse_before = u_bottom_before < 0
    reverse_after = u_bottom_after < 0
    if np.any(reverse_before):
        ax.fill_between(x_coords, 0, u_bottom_before, where=reverse_before, 
                       alpha=0.3, color='blue', label='Reverse flow (before)')
    if np.any(reverse_after):
        ax.fill_between(x_coords, 0, u_bottom_after, where=reverse_after, 
                       alpha=0.3, color='red', label='Reverse flow (after)')
    
    # 2. Plot difference
    ax = axes[1]
    u_diff = u_bottom_after - u_bottom_before
    ax.plot(x_coords, u_diff, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('Δu = u_after - u_before')
    ax.set_title('Change in u due to PPE')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f'u_bottom_profile{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Save velocity field slices
    # Middle x-slice
    i_mid = Nx // 2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax = axes[0]
    ax.plot(y_coords, u_before[i_mid, :], 'b-', label='Before PPE', linewidth=2)
    ax.plot(y_coords, u_after[i_mid, :], 'r-', label='After PPE', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('y')
    ax.set_ylabel('u')
    ax.set_title(f'Velocity u at x={x_coords[i_mid]:.3f} (middle of domain)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(y_coords, v_before[i_mid, :], 'b-', label='Before PPE', linewidth=2)
    ax.plot(y_coords, v_after[i_mid, :], 'r-', label='After PPE', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('y')
    ax.set_ylabel('v')
    ax.set_title(f'Velocity v at x={x_coords[i_mid]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f'velocity_slice_mid{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Streamline plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Before PPE - streamlines
    ax = axes[0, 0]
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    contour = ax.contour(X, Y, psi_before, levels=20, colors='blue', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines Before PPE')
    ax.set_aspect('equal')
    
    # Before PPE - velocity vectors (subsampled)
    skip = max(1, min(Nx, Ny) // 20)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u_before[::skip, ::skip], v_before[::skip, ::skip],
              scale=1.0, alpha=0.5, color='blue')
    
    # After PPE - streamlines
    ax = axes[0, 1]
    contour = ax.contour(X, Y, psi_after, levels=20, colors='red', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines After PPE')
    ax.set_aspect('equal')
    
    # After PPE - velocity vectors
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u_after[::skip, ::skip], v_after[::skip, ::skip],
              scale=1.0, alpha=0.5, color='red')
    
    # Difference in stream function
    ax = axes[1, 0]
    psi_diff = psi_after - psi_before
    im = ax.contourf(X, Y, psi_diff, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Change in Stream Function (After - Before)')
    ax.set_aspect('equal')
    
    # Velocity magnitude difference
    ax = axes[1, 1]
    mag_before = np.sqrt(u_before**2 + v_before**2)
    mag_after = np.sqrt(u_after**2 + v_after**2)
    mag_diff = mag_after - mag_before
    im = ax.contourf(X, Y, mag_diff, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Change in Velocity Magnitude')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f'streamlines{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Save raw data as numpy arrays
    np.savez(
        os.path.join(diag_dir, f'ppe_data{suffix}.npz'),
        U_before=U_before_np,
        U_after=U_after_np,
        psi_before=psi_before,
        psi_after=psi_after,
        u_bottom_before=u_bottom_before,
        u_bottom_after=u_bottom_after,
        x_coords=x_coords,
        y_coords=y_coords,
        step=step,
        iteration=iteration
    )
    
    # Print summary statistics
    print(f"\n📊 PPE Diagnostics (step {step}, iter {iteration}):")
    print(f"   u_bottom_before: min={u_bottom_before.min():.6f}, max={u_bottom_before.max():.6f}, mean={u_bottom_before.mean():.6f}")
    print(f"   u_bottom_after:  min={u_bottom_after.min():.6f}, max={u_bottom_after.max():.6f}, mean={u_bottom_after.mean():.6f}")
    reverse_before_count = np.sum(u_bottom_before < 0)
    reverse_after_count = np.sum(u_bottom_after < 0)
    if reverse_before_count > 0:
        print(f"   ⚠️  Reverse flow (before): {reverse_before_count}/{Nx} points ({100*reverse_before_count/Nx:.1f}%)")
    if reverse_after_count > 0:
        print(f"   ⚠️  Reverse flow (after):  {reverse_after_count}/{Nx} points ({100*reverse_after_count/Nx:.1f}%)")
    print(f"   Saved to: {diag_dir}")
