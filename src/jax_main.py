import os
import argparse
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import utils  # Moved to legacy folder - using jax_utils instead
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time 
import sys
import jax.numpy as jnp
import jax
from jax import jit
from scipy.sparse import kron, identity, linalg
from jax_utils import (
    jax_calculate_density,
    jax_calculate_reynolds_number,
    jax_calculate_weber_number,
    jax_dx,
    jax_dy,
    jax_apply_contact_angle_boundary_conditions,
    jax_build_2d_laplacian_matrix_with_variable_steps,
    jax_laplacian,
    jax_gradient,
    jax_divergence,
    jax_df_2,
    jax_f_2,
    jax_surface_tension_force,
    solve_poisson_pyamg,
    jax_apply_surface_tension_boundary_conditions,
)
from chemical_potential_bc import apply_chemical_potential_contact_angle_bc
from plot_utils import (
    create_joint_plot, 
    save_checkpoint, 
    list_checkpoints, 
    load_checkpoint, 
    load_config
)
from sparse_solver import SparseSolverWrapper
# Global variables used by multiple functions
phi = None
Re1 = None
Re2 = None
Pe = None
epsilon = None
phase_penalty = None
contact_angle = None
Fr = None
g = None
atm_pressure = None


def solve_poisson(rhs, Nx, Ny, dx, dy):
    """Solve the Poisson equation ∇²φ = f with Neumann boundary conditions."""
    A = jax_build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy)
    
    # Reshape right-hand side to match matrix equation
    rhs_flat = np.transpose(rhs).flatten(order='C')  # Use C-style ordering (row-major)
    
    # Solve the linear system
    phi_flat = spsolve(A, rhs_flat)
    
    # Reshape solution to 2D - use proper ordering
    phi = phi_flat.reshape((Nx, Ny), order='C')
    phi = np.transpose(phi)
    
    return phi

@jit
def compute_viscous_term(U, dx, dy, Re):
    """Simplified viscous term for constant viscosity: (1/Re) * ∇²U"""
    return jnp.stack([jax_laplacian(U[..., 0], dx, dy) / Re, 
                      jax_laplacian(U[..., 1], dx, dy) / Re], axis=-1)

@jit
def check_continuity(U, dx, dy):
    """
    Check continuity equation condition (∇·U = 0)
    Returns the divergence field and maximum absolute divergence
    """
    # Calculate divergence: du/dx + dv/dy
    u_x = jax_dx(U[..., 0], h=dx)
    v_y = jax_dy(U[..., 1], h=dy)
    
    divergence = u_x + v_y
    max_div = jnp.max(jnp.abs(divergence))
    mean_div = jnp.mean(jnp.abs(divergence))
    
    return divergence, max_div, mean_div


def jax_update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2, include_gravity=False):
    """Update the velocity field U based on the phase field phi."""
    # Calculate the Reynolds number and density
    Re = jax_calculate_reynolds_number(phi, Re1, Re2)
    rho = jax_calculate_density(phi, rho1, rho2)
    rho_stacked = jnp.stack([rho, rho], axis=-1) + 1e-6

    # Calculate gradients and terms
    grad_U = jax_gradient(U, dx, dy)
    p_grad = jax_gradient(p, dx, dy)
    
    # Calculate viscous term with proper scaling
    viscous_term = compute_viscous_term(U, dx, dy, Re)
    
    # Calculate convective term (in conservative form)
    convective_term = jnp.stack(
        [
            U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1],
            U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]
        ],
        axis=-1
    )

    # Combine terms with proper density scaling
    rhs_U = (
        -p_grad / rho_stacked +  # Pressure term
        viscous_term / rho_stacked +  # Viscous term
        -surface_tension / rho_stacked +  # Surface tension
        -convective_term  # Convective term (already includes velocity)
    )

    # Add gravity if included
    if include_gravity:
        rhs_U = rhs_U + (1 / Fr) * jnp.stack([jnp.zeros_like(U[..., 0]), -jnp.ones_like(U[..., 1])], axis=-1)

    # Update velocity field using explicit Euler
    U = U + current_dt * rhs_U
    
    return U

update_velocity = jit(jax_update_velocity, static_argnames='include_gravity')

@jit
def apply_velocity_boundary_conditions(U, beta, dy):
    """Apply physically appropriate boundary conditions to velocity field.
    
    Args:
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
    
    Returns:
        np.ndarray: Velocity with boundary conditions applied.
    """
    # Make a copy to avoid modifying the original array
    U = U.at[:, 0, 1].set(0.0)
    U = U.at[:, 0, 0].set(U[:, 1, 0] - dy * 1/beta * U[:, 1, 0])

    # Top boundary (open atmosphere): Zero-gradient condition
    # U = U.at[:, -1, :].set(U[:, -2, :])
    U = U.at[:, -1, :].set(0.0)
    
    # Left and right boundaries: Zero-gradient condition
    U = U.at[0, :, :].set(U[1, :, :])
    U = U.at[-1, :, :].set(U[-2, :, :])
    # U = U.at[0, :, :].set(0.0)
    # U = U.at[-1, :, :].set(0.0)
    
    return U

def update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2, pressure_solver, include_gravity=True):
    """Update the pressure field P based on the velocity field U and phase field phi."""
    sf_grad = -jax_divergence(surface_tension, dx, dy)
    rho = jax_calculate_density(phi, rho1, rho2)

    # Apply gravity only if include_gravity is True
    if include_gravity:
        mass_sum = jnp.cumsum(rho[..., ::-1] * g * dy, axis=1) + atm_pressure
        mass_sum = mass_sum[:, ::-1]
    else:
        # No gravity - just use atmospheric pressure
        mass_sum = jnp.full_like(rho, atm_pressure)

    sf_grad = sf_grad.at[0, :].set(mass_sum[0, :])
    sf_grad = sf_grad.at[-1, :].set(mass_sum[-1, :])
    sf_grad = sf_grad.at[:, 0].set(mass_sum[:, 0])
    sf_grad = sf_grad.at[:, -1].set(mass_sum[:, -1])
    # sf_grad = sf_grad.at[:, 0].set(jnp.sum(rho * g * dy, axis=1) + atm_pressure)
    # sf_grad = sf_grad.at[:, -1].set(atm_pressure)
    pressure_solver.set_rhs(sf_grad)
    pressure_solver.solve()
    P = pressure_solver.get_solution()
    # print(f"P top: {P[10:15, -1]}, bottom: {P[-1, 10:15]}, left: {P[10:15, 0]}, right: {P[10:15, -1]}")

    # print(f"P top: {P[10:15, -1]}, bottom: {P[-1, 10:15]}, left: {P[10:15, 0]}, right: {P[10:15, -1]}")

    # P = solve_poisson(sf_grad, Nx, Ny, dx, dy) 
    # sf_grad = sf_grad.transpose()
    # pressure_solver.set_rhs(sf_grad)
    # pressure_solver.solve()
    # P = pressure_solver.get_solution()
    # P = P.transpose()
    return P

def penalization(phi, alpha):
    # Apply penalization force if phi exceeds [-1, 1]
    mask_pos = phi > 1.0
    mask_neg = phi < -1.0
    
    penalty = np.zeros_like(phi)
    penalty[mask_pos] = (phi[mask_pos] - 1.0)
    penalty[mask_neg] = (phi[mask_neg] + 1.0)
    
    return alpha * penalty

@jit
def update_phase(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon):
    """Update the phase field using the explicit Euler method with proper physical mass conservation.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Time step.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        contact_angle (float): Contact angle for boundary conditions.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
    """
    # Store initial physical mass for conservation (not just phi sum)
    initial_rho = jax_calculate_density(phi, rho1, rho2)
    initial_physical_mass = jnp.sum(initial_rho) * dx * dy
    
    # Step 1: Calculate the gradient of phi
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the Laplacian of phi
    lap_phi = jax_laplacian(phi, dx, dy)

    # Step 4: Calculate chemical potential (standard Allen-Cahn)
    chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi
    
    # Mass-conserving Lagrange multiplier: enforce ∫(∂φ/∂t) dV = 0
    lagrange_multiplier = jnp.mean(chemical_potential)
    
    # Add convective contribution to Lagrange multiplier for better mass conservation
    mean_convective = jnp.mean(convective_term)
    lagrange_multiplier += Pe * mean_convective
    
    source_term = -1/Pe * (chemical_potential - lagrange_multiplier)

    # Step 5: Right-hand side of the phase equation
    rhs_phi = -convective_term + source_term

    # Step 6: Update phase field
    phi_new = phi + current_dt * rhs_phi

    # Step 7: Apply boundary conditions and maintain phase field bounds
    phi_new = jax_apply_contact_angle_boundary_conditions(phi_new, dx, dy, contact_angle=contact_angle)
    
    # Ensure phi stays within physical bounds [-1, 1] for stability
    phi_new = jnp.clip(phi_new, -1.0, 1.0)
    
    # Step 8: Apply physical mass correction to ensure exact mass conservation
    current_rho = jax_calculate_density(phi_new, rho1, rho2)
    current_physical_mass = jnp.sum(current_rho) * dx * dy
    mass_error = initial_physical_mass - current_physical_mass
    
    # Calculate how density changes with phi
    # Derivative of density with respect to phi
    # For rho = 1/((1+phi_mapped)/(2*rho2) + (1-phi_mapped)/(2*rho1))
    # d(rho)/d(phi_mapped) = rho^2 * (1/(2*rho1) - 1/(2*rho2))
    # d(phi_mapped)/d(phi) = 1/2
    drho_dphi = current_rho**2 * (1/(2*rho1) - 1/(2*rho2)) * 0.5
    
    # Calculate correction to phi that will fix the mass error
    # mass_error ≈ ∑(drho_dphi * dphi) * dx * dy
    # So: dphi = mass_error / (∑(drho_dphi) * dx * dy)
    total_sensitivity = jnp.sum(drho_dphi) * dx * dy
    
    # Calculate phi correction (avoid division by zero)
    phi_correction = jnp.where(
        jnp.abs(total_sensitivity) > 1e-12,
        mass_error / total_sensitivity,
        0.0
    )
    
    # Apply correction only in interface region and only if mass error is significant
    interface_mask = (jnp.abs(phi_new) < 0.9)
    mass_error_significant = jnp.abs(mass_error) > 1e-10
    
    phi_correction_field = jnp.where(
        jnp.logical_and(interface_mask, mass_error_significant),
        phi_correction,
        0.0
    )
    
    phi_new = phi_new + phi_correction_field
    
    # Ensure bounds are maintained after mass correction
    phi_new = jnp.clip(phi_new, -1.0, 1.0)

    return phi_new

"""
=== CAHN-HILLIARD PHASE FIELD METHODS ===

The following functions implement the Cahn-Hilliard equation as an alternative to the Allen-Cahn 
equation used in the update_phase function above.

Key differences between Allen-Cahn and Cahn-Hilliard:

1. CONSERVATION PROPERTIES:
   - Allen-Cahn: Non-conservative (requires manual mass correction)
   - Cahn-Hilliard: Naturally mass-conservative (∫φ dV = constant)

2. EQUATION ORDER:
   - Allen-Cahn: 2nd order PDE: ∂φ/∂t = -1/Pe * (f'(φ) - ε²∇²φ) - u·∇φ
   - Cahn-Hilliard: 4th order PDE: ∂φ/∂t = ∇·(M∇μ) - u·∇φ, where μ = f'(φ) - ε²∇²φ

3. PHYSICAL INTERPRETATION:
   - Allen-Cahn: Models interface motion driven by curvature reduction
   - Cahn-Hilliard: Models phase separation via diffusion with conserved order parameter

4. NUMERICAL STABILITY:
   - Allen-Cahn: Generally more stable, faster convergence
   - Cahn-Hilliard: Can be less stable due to 4th order nature, requires smaller time steps

5. APPLICATIONS:
   - Allen-Cahn: Interface tracking, solidification, grain growth
   - Cahn-Hilliard: Phase separation, spinodal decomposition, conserved dynamics

For droplet spreading simulations, both approaches are valid:
- Use Allen-Cahn for faster computation with manual mass conservation
- Use Cahn-Hilliard for physically accurate mass conservation without corrections

The mobility parameter M controls the diffusion rate in Cahn-Hilliard:
- Constant mobility: M = constant (simple case)
- Degenerate mobility: M(φ) = (1-φ²) (physically motivated, zero in pure phases)
"""

@jit
def update_phase_cahn_hilliard(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon, mobility=1.0, chemical_potential_bc_strength=0.1):
    """Update the phase field using the Cahn-Hilliard equation with explicit Euler method.
    
    The Cahn-Hilliard equation is:
    ∂φ/∂t + ∇·(φu) = ∇·(M∇μ)
    
    where μ = f'(φ) - ε²∇²φ is the chemical potential and M is the mobility.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Time step.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        contact_angle (float): Contact angle for boundary conditions.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
        mobility (float): Mobility parameter M (default: 1.0).
    
    Returns:
        np.ndarray: Updated phase field.
    """
    # Step 1: Calculate the gradient of phi for convective term
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term ∇·(φu)
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Apply contact-angle Robin BC for chemical potential BEFORE computing chemical potential
    # This ensures the phase field satisfies ∂nφ = -(√2/η) cosθw (φ² - 1)
    # Using very conservative strength for stability
    phi = apply_chemical_potential_contact_angle_bc(phi, dx, dy, contact_angle, epsilon, strength=chemical_potential_bc_strength)
    
    # Step 3: Calculate the chemical potential μ = f'(φ) - ε²∇²φ
    lap_phi = jax_laplacian(phi, dx, dy)
    
    # Calculate the chemical potential
    chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi

    # Step 4: Calculate the divergence of the mobility flux ∇·(M∇μ)
    # For constant mobility M, this becomes M∇²μ
    # Scale mobility by grid spacing for numerical stability
    effective_mobility = mobility * min(dx, dy)**2
    diffusion_term = effective_mobility * jax_laplacian(chemical_potential, dx, dy)

    # Step 5: Right-hand side of the Cahn-Hilliard equation
    # ∂φ/∂t = -∇·(φu) + ∇·(M∇μ)
    # Scale the diffusion term to prevent numerical instability
    rhs_phi = -convective_term + diffusion_term / (Pe * 100.0)  # Additional scaling factor

    # Step 6: Apply CFL-like stability constraint for Cahn-Hilliard
    # For 4th order PDE, stability requires dt < C * dx^4 / (mobility * epsilon^2)
    max_dt_ch = 0.01 * min(dx, dy)**4 / (effective_mobility * epsilon**2 + 1e-12)
    stable_dt = min(current_dt, max_dt_ch)
    
    # Step 7: Update phase field using explicit Euler with stable time step
    phi_new = phi + stable_dt * rhs_phi

    # Step 8: Apply boundary conditions
    phi_new = jax_apply_contact_angle_boundary_conditions(phi_new, dx, dy, contact_angle=contact_angle)
    
    # Step 9: CRITICAL: Ensure phi stays within physical bounds [-1, 1] for stability
    # This is essential for Cahn-Hilliard to prevent blow-up
    phi_new = jnp.clip(phi_new, -1.0, 1.0)

    return phi_new

@jit
def update_phase_cahn_hilliard_improved(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon, 
                                       mobility_constant=1.0, degenerate_mobility=True, chemical_potential_bc_strength=0.1):
    """Update the phase field using the Cahn-Hilliard equation with improved numerical stability.
    
    The Cahn-Hilliard equation is:
    ∂φ/∂t + ∇·(φu) = ∇·(M(φ)∇μ)
    
    where μ = f'(φ) - ε²∇²φ is the chemical potential and M(φ) is the mobility.
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Time step.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        contact_angle (float): Contact angle for boundary conditions.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
        mobility_constant (float): Mobility scaling parameter (default: 1.0).
        degenerate_mobility (bool): Use degenerate mobility M(φ) = (1-φ²) (default: True).
    
    Returns:
        np.ndarray: Updated phase field.
    """
    # Step 1: Calculate the gradient of phi for convective term
    grad_phi = jax_gradient(phi, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 2: Calculate the convective term ∇·(φu)
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Apply contact-angle Robin BC for chemical potential BEFORE computing chemical potential
    phi = apply_chemical_potential_contact_angle_bc(phi, dx, dy, contact_angle, epsilon)
    
    # Step 3: Calculate the chemical potential μ = f'(φ) - ε²∇²φ
    lap_phi = jax_laplacian(phi, dx, dy)
    chemical_potential = jax_df_2(phi) - epsilon**2 * lap_phi

    # Step 4: Calculate mobility M(φ)
    if degenerate_mobility:
        # Degenerate mobility: M(φ) = (1 - φ²) 
        # This ensures mobility goes to zero in pure phases (φ = ±1)
        mobility = mobility_constant * (1.0 - phi**2)
        # Add small regularization to prevent complete degeneracy
        mobility = jnp.maximum(mobility, 1e-6 * mobility_constant)
    else:
        # Constant mobility
        mobility = mobility_constant * jnp.ones_like(phi)

    # Step 5: Calculate the gradient of the chemical potential
    grad_mu = jax_gradient(chemical_potential, dx, dy)  # Shape: (Nx, Ny, 2)

    # Step 6: Calculate ∇·(M(φ)∇μ) using the product rule
    # ∇·(M∇μ) = M∇²μ + ∇M·∇μ
    
    # First term: M∇²μ
    lap_mu = jax_laplacian(chemical_potential, dx, dy)
    term1 = mobility * lap_mu
    
    # Second term: ∇M·∇μ (only needed for variable mobility)
    if degenerate_mobility:
        grad_mobility = jax_gradient(mobility, dx, dy)  # Shape: (Nx, Ny, 2)
        term2 = grad_mobility[..., 0] * grad_mu[..., 0] + grad_mobility[..., 1] * grad_mu[..., 1]
    else:
        term2 = 0.0
    
    diffusion_term = term1 + term2

    # Step 7: Right-hand side of the Cahn-Hilliard equation
    # ∂φ/∂t = -∇·(φu) + (1/Pe)∇·(M∇μ)
    rhs_phi = -convective_term + diffusion_term / Pe

    # Step 8: Update phase field using explicit Euler
    phi_new = phi + current_dt * rhs_phi

    # Step 9: Apply boundary conditions
    phi_new = jax_apply_contact_angle_boundary_conditions(phi_new, dx, dy, contact_angle=contact_angle)
    
    # Step 10: Ensure phi stays within physical bounds [-1, 1] for stability
    # Cahn-Hilliard naturally conserves mass, but we still clip for numerical stability
    phi_new = jnp.clip(phi_new, -1.0, 1.0)

    return phi_new

# @jit
def update_phase_cahn_hilliard_stable(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon, mobility=0.1, chemical_potential_bc_strength=0.1):
    """Update the phase field using a numerically stable Cahn-Hilliard implementation.
    
    This version includes multiple stability measures:
    - Adaptive time stepping
    - Proper scaling of diffusion terms
    - Regularization of chemical potential
    - Conservative clipping
    
    Args:
        phi (np.ndarray): Current phase field (shape: (Nx, Ny)).
        U (np.ndarray): Velocity field (shape: (Nx, Ny, 2)).
        current_dt (float): Time step.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        contact_angle (float): Contact angle for boundary conditions.
        rho1 (float): Density of phase 1.
        rho2 (float): Density of phase 2.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness parameter.
        mobility (float): Mobility parameter M (default: 0.1).
    
    Returns:
        np.ndarray: Updated phase field.
    """
    # Store initial mass for conservation check
    initial_mass = jnp.sum(phi) * dx * dy
    
    # Step 1: Calculate the gradient of phi for convective term
    grad_phi = jax_gradient(phi, dx, dy)

    # Step 2: Calculate the convective term ∇·(φu)
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]

    # Step 3: Calculate the chemical potential with regularization
    lap_phi = jax_laplacian(phi, dx, dy)
    
    # Regularized chemical potential to prevent extreme values
    f_prime = jax_df_2(phi)
    # Clip f_prime to prevent extreme values
    f_prime = jnp.clip(f_prime, -10.0, 10.0)
    
    # Apply contact-angle Robin BC for chemical potential BEFORE computing chemical potential
    # This ensures the phase field satisfies ∂nφ = -(√2/η) cosθw (φ² - 1)
    # Using very conservative strength for stability
    phi = apply_chemical_potential_contact_angle_bc(phi, dx, dy, contact_angle, epsilon, strength=chemical_potential_bc_strength)
    
    # Recompute Laplacian after applying boundary condition
    lap_phi = jax_laplacian(phi, dx, dy)
    chemical_potential = f_prime - epsilon**2 * lap_phi
    # Additional regularization of chemical potential
    chemical_potential = jnp.clip(chemical_potential, -100.0, 100.0)

    # Step 4: Calculate diffusion term with proper scaling
    # Use smaller effective mobility and proper grid scaling
    grid_scale = min(dx, dy)
    effective_mobility = mobility * grid_scale**2 / epsilon**2
    
    # Calculate Laplacian of chemical potential
    lap_mu = jax_laplacian(chemical_potential, dx, dy)
    diffusion_term = effective_mobility * lap_mu

    # Step 5: Right-hand side with conservative scaling
    # Use much smaller scaling for the diffusion term
    diffusion_scale = 1.0 / (Pe * max(100.0, 1.0/epsilon**2))
    rhs_phi = -convective_term + diffusion_term * diffusion_scale

    # Step 6: Adaptive time stepping for stability
    # Estimate maximum safe time step
    max_rhs = jnp.max(jnp.abs(rhs_phi))
    safe_dt = jnp.where(max_rhs > 1e-12, 
                       min(0.1 / max_rhs, current_dt),
                       current_dt)
    
    # Additional CFL constraint for 4th order PDE
    cfl_dt = 0.001 * grid_scale**4 / (effective_mobility + 1e-12)
    stable_dt = min(safe_dt, cfl_dt, current_dt)
    stable_dt = current_dt

    # Step 7: Update with stable time step
    phi_new = phi + stable_dt * rhs_phi

    # Step 8: Apply boundary conditions
    phi_new = jax_apply_contact_angle_boundary_conditions(phi_new, dx, dy, contact_angle=contact_angle)
    
    # Step 9: Conservative clipping to maintain bounds
    phi_new = jnp.clip(phi_new, -1.0, 1.0)
    
    # Step 10: Mass conservation correction (optional)
    current_mass = jnp.sum(phi_new) * dx * dy
    mass_error = initial_mass - current_mass
    
    # Apply small mass correction if error is significant
    if jnp.abs(mass_error) > 1e-6:
        correction = mass_error / (phi_new.size * dx * dy)
        phi_new = phi_new + correction
        # Re-clip after correction
        phi_new = jnp.clip(phi_new, -1.0, 1.0)

    return phi_new

def correction_step(U, dx, dy, dt, correction_solver=None, div=None):
    print(f"DEBUG correction_step: Input U min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")
    
    if div is None:
        # Use finite differences for divergence calculation
        u_x = jax_dx(U[..., 0], h=dx)
        v_y = jax_dy(U[..., 1], h=dy)
        div = u_x + v_y
        
        print(f"DEBUG correction_step: U[..., 0] min/max: {np.array(U[..., 0]).min():.6f} / {np.array(U[..., 0]).max():.6f}")
        print(f"DEBUG correction_step: U[..., 1] min/max: {np.array(U[..., 1]).min():.6f} / {np.array(U[..., 1]).max():.6f}")
        print(f"DEBUG correction_step: dx = {dx:.10f}, dy = {dy:.10f}")
        print(f"DEBUG correction_step: u_x min/max: {np.array(u_x).min():.6f} / {np.array(u_x).max():.6f}")
        print(f"DEBUG correction_step: v_y min/max: {np.array(v_y).min():.6f} / {np.array(v_y).max():.6f}")
        print(f"DEBUG correction_step: div min/max: {np.array(div).min():.6f} / {np.array(div).max():.6f}")
        
        # Check boundary values specifically
        print(f"DEBUG correction_step: u_x[0, :5] (bottom): {np.array(u_x[0, :5])}")
        print(f"DEBUG correction_step: u_x[-1, :5] (top): {np.array(u_x[-1, :5])}")
        print(f"DEBUG correction_step: v_y[:5, 0] (left): {np.array(v_y[:5, 0])}")
        print(f"DEBUG correction_step: v_y[:5, -1] (right): {np.array(v_y[:5, -1])}")
    
    div = div - jnp.mean(div)
    print(f"DEBUG correction_step: After mean removal div min/max: {np.array(div).min():.6f} / {np.array(div).max():.6f}")
    
    # DEBUG: Check what we're passing to the solver
    rhs = -div / dt
    print(f"DEBUG solver: dt = {dt:.10f}")
    print(f"DEBUG solver: RHS min/max: {np.array(rhs).min():.6f} / {np.array(rhs).max():.6f}")
    print(f"DEBUG solver: RHS contains NaN: {np.any(np.isnan(np.array(rhs)))}")
    print(f"DEBUG solver: RHS contains Inf: {np.any(np.isinf(np.array(rhs)))}")
    
    correction_solver.set_rhs(rhs)
    correction_solver.solve()
    p_correction = correction_solver.get_solution()
    
    # DEBUG: Check the solution
    print(f"DEBUG solver: p_correction min/max: {np.array(p_correction).min():.6f} / {np.array(p_correction).max():.6f}")
    print(f"DEBUG solver: p_correction contains NaN: {np.any(np.isnan(np.array(p_correction)))}")
    print(f"DEBUG solver: p_correction contains Inf: {np.any(np.isinf(np.array(p_correction)))}")
    
    # DEBUG: Check pressure gradients using finite differences
    p_grad_x = jax_dx(p_correction, h=dx)
    p_grad_y = jax_dy(p_correction, h=dy)
    print(f"DEBUG solver: p_grad_x min/max: {np.array(p_grad_x).min():.6f} / {np.array(p_grad_x).max():.6f}")
    print(f"DEBUG solver: p_grad_y min/max: {np.array(p_grad_y).min():.6f} / {np.array(p_grad_y).max():.6f}")
    
    U = U.at[..., 0].set(U[..., 0] - dt * p_grad_x)
    U = U.at[..., 1].set(U[..., 1] - dt * p_grad_y)
    
    print(f"DEBUG solver: U_new min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")
    
    return U, p_correction

def damp_divergence(U, dx, dy, xi, dt):
    div = jax_divergence(U, dx, dy)
    div_grad = jax_gradient(div, dx, dy)
    U = U.at[..., 0].set(U[..., 0] - xi * dt * div_grad[..., 0])
    U = U.at[..., 1].set(U[..., 1] - xi * dt * div_grad[..., 1])
    return U

def ppe(U, dx, dy, dt, correction_solver=None, div_threshold=0.05):
    beta = 0.01
    # global correction_step    
    U, solution = correction_step(U, dx, dy, dt, correction_solver=correction_solver)
    U = apply_velocity_boundary_conditions(U, beta, dy)

    # local corrections
    divergence, max_div, mean_div = check_continuity(U, dx, dy)
    
    prev_mean_div = mean_div

    if mean_div > div_threshold or max_div > div_threshold:
        count = 0
        while mean_div > div_threshold or max_div > div_threshold:
            # Store current state before attempting correction
            prev_mean_div = mean_div
            prev_max_div = max_div
            
            U, solution = correction_step(U, dx, dy, dt, correction_solver=correction_solver, div=divergence)
            U = apply_velocity_boundary_conditions(U, beta, dy)
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            
            # Check if divergence is rising - break if it's getting worse
            if mean_div > 2 * prev_mean_div:
                sys.stdout.write(f"\nDivergence rising ({mean_div:.6f} > {prev_mean_div:.6f}), breaking\n")
                break
                
            if count % 20 == 0:
                sys.stdout.write(f"\rMax|mean div: {max_div:.6f}  | {mean_div:.6f}")
            count += 1
            
            # Safety exit after too many iterations
            if count > 100:
                sys.stdout.write(f"\nMax iterations reached\n")
                break
        sys.stdout.write(f"\nCorrected in {count} iterations\n")
    return U
    
def initialize_phase(Nx, Ny, radius):
    """Initialize the phase field with a semicircle droplet resting on the bottom boundary."""
    # Create a grid of coordinates
    x = np.linspace(0, 1, Nx)  # X coordinates from 0 to 1
    y = np.linspace(0, 1, Ny)   # Y coordinates from 0 to 1
    X, Y = np.meshgrid(x, y, indexing='ij')  # Create a meshgrid with correct indexing
    
    # Define the center of the semicircle at the bottom center of the domain
    center_x = 0.5
    center_y = 0  # Bottom of the domain
    
    # Calculate distance from the center
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Use tanh to create a smooth transition between phases
    # Positive inside the semicircle (phi = 1), negative outside (phi = -1)
    phi = -np.tanh((distance - radius) * (10/radius))  # Scale factor for sharpness
    
    # Apply boundary condition to ensure semicircle sits exactly on the bottom boundary
    phi[:,0] = phi[:,1]  # Copy the first interior row to the boundary
    
    return phi

def get_borders_of_droplet(phi):
    """Get the  borders of the droplet."""
    # Find the first and last non-zero elements in each row=
    start_of_droplet = 0
    end_of_droplet = phi.shape[0] - 1
    for i in range(0, phi.shape[0]):
        if phi[i, 0] > 0:
            start_of_droplet = i
            break
    for i in range(phi.shape[0] - 1, 0, -1):
        if phi[i, 0] > 0:
            end_of_droplet = i
            break
    return start_of_droplet, end_of_droplet

@jit
def cfl_dt(u_max, v_max, dx, dy, C=0.4):
    """Return Δt that gives desired CFL=C."""
    # More conservative CFL calculation
    max_velocity = jnp.maximum(jnp.abs(u_max), jnp.abs(v_max))
    min_grid_spacing = jnp.minimum(dx, dy)  # Use jnp.minimum instead of min()
    # Use jnp.where instead of if statement for JAX compatibility
    return jnp.where(max_velocity > 1e-12, 
                     C * min_grid_spacing / max_velocity, 
                     jnp.inf)

def create_sparse_solver(Nx, Ny, dx, dy, backend, boundary_conditions, args=None):
    solver = SparseSolverWrapper(Nx, Ny, dx, dy, backend, args)
    solver.set_top_boundary_condition(boundary_conditions[0])
    solver.set_bottom_boundary_condition(boundary_conditions[1])
    solver.set_left_boundary_condition(boundary_conditions[2])
    solver.set_right_boundary_condition(boundary_conditions[3])
    solver.create_sparse_matrix()
    return solver

def update_phase_field(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon, 
                      method="allen_cahn", chemical_potential_bc_strength=0.1, **kwargs):
    """Unified interface for updating phase field using different methods.
    
    Args:
        phi (np.ndarray): Current phase field.
        U (np.ndarray): Velocity field.
        current_dt (float): Time step.
        dx, dy (float): Grid spacings.
        contact_angle (float): Contact angle.
        rho1, rho2 (float): Phase densities.
        Pe (float): Peclet number.
        epsilon (float): Interface thickness.
        method (str): Method to use - "allen_cahn", "cahn_hilliard", or "cahn_hilliard_improved".
        **kwargs: Additional parameters for specific methods.
    
    Returns:
        np.ndarray: Updated phase field.
    
    Example usage:
        # Allen-Cahn (default)
        phi_new = update_phase_field(phi, U, dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon)
        
        # Basic Cahn-Hilliard
        phi_new = update_phase_field(phi, U, dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon,
                                   method="cahn_hilliard", mobility=0.5)
        
        # Improved Cahn-Hilliard with degenerate mobility
        phi_new = update_phase_field(phi, U, dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon,
                                   method="cahn_hilliard_improved", 
                                   mobility_constant=0.5, degenerate_mobility=True)
    """
    if method == "allen_cahn":
        return update_phase(phi, U, current_dt, dx, dy, contact_angle, rho1, rho2, Pe, epsilon)
    elif method == "cahn_hilliard":
        mobility = kwargs.get('mobility', 1.0)
        return update_phase_cahn_hilliard(phi, U, current_dt, dx, dy, contact_angle, 
                                        rho1, rho2, Pe, epsilon, mobility, chemical_potential_bc_strength)
    elif method == "cahn_hilliard_improved":
        mobility_constant = kwargs.get('mobility_constant', 1.0)
        degenerate_mobility = kwargs.get('degenerate_mobility', True)
        return update_phase_cahn_hilliard_improved(phi, U, current_dt, dx, dy, contact_angle,
                                                 rho1, rho2, Pe, epsilon, 
                                                 mobility_constant, degenerate_mobility, chemical_potential_bc_strength)
    elif method == "cahn_hilliard_stable":
        mobility = kwargs.get('mobility', 0.1)
        return update_phase_cahn_hilliard_stable(phi, U, current_dt, dx, dy, contact_angle,
                                                 rho1, rho2, Pe, epsilon, mobility, chemical_potential_bc_strength)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'allen_cahn', 'cahn_hilliard', 'cahn_hilliard_improved', or 'cahn_hilliard_stable'")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Droplet spreading simulation')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON format)')
    parser.add_argument('--output', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Access global variables
    global phi, Re1, Re2, Pe, epsilon, phase_penalty, contact_angle, Fr, g, atm_pressure
    
    # Extract parameters from config
    # Physical parameters
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    We1 = config["physical_params"]["We1"]
    We2 = config["physical_params"]["We2"]
    Pe = config["physical_params"]["Pe"]
    epsilon = config["physical_params"]["epsilon"]
    alpha = config["physical_params"]["alpha"]
    phase_penalty = config["physical_params"]["phase_penalty"]
    contact_angle = config["physical_params"]["contact_angle"]
    chemical_potential_bc_strength = config["physical_params"]["chemical_potential_bc_strength"]
    include_gravity = config["physical_params"]["include_gravity"]
    g = config["physical_params"]["g"]
    atm_pressure = config["physical_params"]["atm_pressure"]
    Fr = config["physical_params"]["Fr"]
    
    # Grid setup
    Lx, Ly = config["grid_params"]["Lx"], config["grid_params"]["Ly"]
    Nx, Ny = config["grid_params"]["Nx"], config["grid_params"]["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    
    # Time setup
    dt = config["time_params"]["dt"]
    t_max = config["time_params"]["t_max"]
    num_steps = int(t_max / dt)
    checkpoint_interval = config["time_params"]["checkpoint_interval"]
    dt_initial = config["time_params"]["dt_initial"]
    cfl_number = config["time_params"].get("cfl_number", 0.01)
    
    # Initial conditions
    radius = config["initial_conditions"]["droplet_radius"]
    
    # Restart information
    restart_from = config["restart"]["restart_from"]
    if restart_from == "None":
        restart_from = None
    
    # Create a directory for the experiment
    if args.output:
        # Use provided output directory
        login_dir = args.output
        os.makedirs(login_dir, exist_ok=True)
    else:
        # Use timestamped directory (original behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        login_dir = f"experiment_{timestamp}"
        os.makedirs(login_dir, exist_ok=True)

    # Initialize the phase field with a semicircle droplet
    phi = initialize_phase(Nx, Ny, radius)  # Initialize the phase field
    phi = jnp.array(phi)
    phi = jax_apply_contact_angle_boundary_conditions(phi, dx, dy, contact_angle=contact_angle)

    # Initialize fields
    U = np.zeros((Nx, Ny, 2))  # Velocity field (2D vector field)
    P = np.zeros((Nx, Ny))  # Pressure field

    # Visualization of the phase field
    plt.imshow(phi.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    plt.colorbar(label='Phase Field (phi)')
    plt.title('Phase Field Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f'{login_dir}/initial_phase_field.png', bbox_inches='tight')
    plt.clf()

    # Create a directory for checkpoints
    checkpoint_dir = os.path.join(login_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the configuration to a JSON file
    params_file = os.path.join(login_dir, "simulation_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(config, f, indent=4)
    sys.stdout.write(f"Simulation parameters saved to {params_file}\n")

    # Optional: restart from checkpoint
    if restart_from is not None:
        sys.stdout.write(f"Restarting from checkpoint: {restart_from} \n")
        checkpoint_data = load_checkpoint(restart_from)
        
        # Load simulation state
        start_step = checkpoint_data['step']
        phi = checkpoint_data['phi']
        U = checkpoint_data['U']
        P = checkpoint_data['P']
        
        sys.stdout.write(f"Loaded state from step {start_step}, continuing simulation... \n")
    else:
        # Initialize from scratch
        start_step = 0
    
    # Create solvers using configuration parameters
    # Extract solver parameters
    pressure_solver_config = config.get("solver_params", {}).get("pressure_solver", {})
    correction_solver_config = config.get("solver_params", {}).get("correction_solver", {})
    boundary_conditions = config.get("boundary_conditions", {})
    
    # Set default values if not specified
    pressure_backend = pressure_solver_config.get("backend", "pyamg")
    pressure_args = {
        'accel': pressure_solver_config.get("accel", "bicgstab"),
        'tol': pressure_solver_config.get("tol", 0.05),
        'maxiter': pressure_solver_config.get("maxiter", 10000)
    }
    
    correction_backend = correction_solver_config.get("backend", "pyamg")
    correction_args = {
        'accel': correction_solver_config.get("accel", "bicgstab"),
        'tol': correction_solver_config.get("tol", 0.05),
        'maxiter': correction_solver_config.get("maxiter", 10000)
    }
    
    # Extract boundary conditions
    pressure_bc = boundary_conditions.get("pressure", {})
    velocity_bc = boundary_conditions.get("velocity", {})
    
    # Map periodic BCs to supported types (periodic -> neumann for now)
    def map_bc(bc_type, default):
        if bc_type == "periodic":
            return "neumann"
        return bc_type if bc_type in ["dirichlet", "neumann"] else default
    
    # Create solvers with configuration parameters
    correction_solver = create_sparse_solver(Nx, Ny, dx, dy,
                                              correction_backend,
                                              [map_bc(pressure_bc.get("top"), "dirichlet"), 
                                               map_bc(pressure_bc.get("bottom"), "neumann"), 
                                               map_bc(pressure_bc.get("left"), "dirichlet"), 
                                               map_bc(pressure_bc.get("right"), "dirichlet")],
                                              correction_args)
    pressure_solver = create_sparse_solver(Nx, Ny, dx, dy,
                                              pressure_backend,
                                              [map_bc(pressure_bc.get("top"), "dirichlet"), 
                                               map_bc(pressure_bc.get("bottom"), "neumann"), 
                                               map_bc(pressure_bc.get("left"), "dirichlet"), 
                                               map_bc(pressure_bc.get("right"), "dirichlet")],
                                              pressure_args)

    times = []
    # Main simulation loop
    cur_t = 0
    step = 0
    while cur_t < t_max:
        # Use dt_initial for the first 100 steps
        # Compute derivatives and other terms
        
        start_time = time.time()
        current_dt = dt_initial if step < 500 else dt

        # More conservative time step selection
        cfl_computed_dt = cfl_dt(U[..., 0].max(), U[..., 1].max(), dx, dy, C=cfl_number)
        
        # Additional stability constraint for PPE
        ppe_dt = 100.0 * jnp.minimum(dx, dy)**2  # Diffusion-like constraint for pressure correction
        
        # Take the most conservative time step
        if cfl_computed_dt != np.inf:
            current_dt = min(current_dt, float(cfl_computed_dt), float(ppe_dt))
        else:
            current_dt = min(current_dt, float(ppe_dt))
        
        # Ensure minimum time step for numerical stability
        current_dt = max(current_dt, 1e-8)
        cur_t += current_dt

        surface_tension = jax_surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)
        
        print(f"DEBUG main: Before velocity update U min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")
        U = update_velocity(U, P, surface_tension, current_dt, dx, dy, rho1, rho2, include_gravity=include_gravity)
        print(f"DEBUG main: After velocity update U min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")

        # Project velocity to ensure continuity is satisfied
        beta = 0.01
        print(f"DEBUG main: Before boundary conditions U min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")
        U = apply_velocity_boundary_conditions(U, beta, dy)
        print(f"DEBUG main: After boundary conditions U min/max: {np.array(U).min():.6f} / {np.array(U).max():.6f}")

        # Add special handling for extreme divergence points
        _, max_div, mean_div = check_continuity(U, dx, dy)
            
        # Apply PPE with more conservative threshold
        U = ppe(U, dx, dy, current_dt, correction_solver, div_threshold=0.01)

        # beta = 0.01
        # Apply no-slip boundary conditions
        # U = apply_velocity_boundary_conditions(U, beta, dy)

        # Apply contact angle boundary conditions
        phi = update_phase_field(phi, U, current_dt, dx, dy, 
                                 contact_angle, rho1, rho2, Pe, epsilon,
                                 method="cahn_hilliard_stable",
                                 mobility=1.0,  # Conservative mobility for water-air interface
                                 chemical_potential_bc_strength=chemical_potential_bc_strength)
        
        # Recompute surface tension
        surface_tension = jax_surface_tension_force(phi, epsilon, We1, We2, dx, dy)
        surface_tension = jax_apply_surface_tension_boundary_conditions(surface_tension, phi, contact_angle=contact_angle)

        # Update pressure
        P = update_pressure(surface_tension, Nx, Ny, dx, dy, rho1, rho2, pressure_solver, include_gravity)

        end_time = time.time()
        times.append(end_time - start_time)
        
        # Print or log results for analysis
        if step % 10 == 0:  # Print every 10 steps
            sys.stdout.write(f"Step {step}, Time {cur_t:.2f} \n")
            sys.stdout.write(f"Min/Max of U: {U.min():.4f} / {U.max():.4f} \n")
            sys.stdout.write(f"Min/Max of P: {P.min():.4f} / {P.max():.4f} \n")
            sys.stdout.write(f"Min/Max of phi: {phi.min():.4f} / {phi.max():.4f} \n")
            divergence, max_div, mean_div = check_continuity(U, dx, dy)
            sys.stdout.write(f"Continuity check - Max |div(U)|: {max_div:.6f}, Mean |div(U)|: {mean_div:.6f} \n")
            sys.stdout.write(f"Time: {np.mean(times):.6f} \n")
            start_of_droplet, end_of_droplet = get_borders_of_droplet(phi)
            mass = np.sum(phi[phi > 0])
            sys.stdout.write(f"Droplet mass {mass} \n")
            sys.stdout.write(f"Start/end of the droplet on the surface: {start_of_droplet} - {end_of_droplet} \n")

        if step % checkpoint_interval == 0: 
            mass = np.sum(phi[phi > 0])
            # Extract plotting parameters from config
            plotting_params = config.get("plotting_params", {})
            # def create_joint_plot(phi, U, P, 
            # surface_tension, dt, step, dx, dy,
            # mass, rho1, rho2, save_path=None):

            create_joint_plot(
                phi=phi,
                U=U,
                P=P,
                surface_tension=surface_tension,
                dt=current_dt,
                step=step,
                dx=dx,
                dy=dy,
                mass=mass,
                rho1=rho1,
                rho2=rho2,
                cur_t=cur_t,
                save_path=f'{login_dir}/joint_plot_step_{step}.png',
                # plotting_params=plotting_params
            )

        # Save checkpoint at specified intervals
        if step % checkpoint_interval == 0:
            save_checkpoint(step, phi, U, P, directory=f"{login_dir}/checkpoints")
        step += 1

if __name__ == "__main__":
    main()