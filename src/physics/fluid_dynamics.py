"""
Fluid dynamics equations for droplet spreading simulation.
"""

import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_gradient, jax_divergence, jax_laplacian
from physics.properties import jax_calculate_density, jax_calculate_reynolds_number


@jit
def jax_compute_viscous_term(U, dx, dy, Re, f_1_grid, f_2_grid):
    """Calculate viscous term (1/Re)∇²U. f_1_grid, f_2_grid: (Nx, Ny)."""
    return jnp.stack([jax_laplacian(U[..., 0], dx, dy, f_1_grid, f_2_grid) / Re,
                      jax_laplacian(U[..., 1], dx, dy, f_1_grid, f_2_grid) / Re], axis=-1)


@jit
def jax_check_continuity(U, dx, dy, f_1_grid):
    """Check continuity equation. Curvilinear (x, eta): fluid-only grid. f_1_grid: terrain gradient (Nx, Ny)."""
    divergence_field = jax_divergence(U, dx, dy, f_1_grid)
    max_div = jnp.max(jnp.abs(divergence_field))
    mean_div = jnp.mean(jnp.abs(divergence_field))
    return divergence_field, max_div, mean_div


@jit
def jax_update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2,
                        Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid, include_gravity=False, psi=None):
    """Update velocity (predictor step). Grid is fluid-only (bottom-aligned). Ice (psi>0) zeroed when present."""
    ice_mask = psi > 0.0
    U = jnp.where(ice_mask[..., jnp.newaxis], 0.0, U)

    # Material properties
    Re = jax_calculate_reynolds_number(phi, Re1, Re2)
    rho = jax_calculate_density(phi, rho1, rho2)
    rho = jnp.maximum(rho, 1e-6)
    rho_stacked = jnp.stack([rho, rho], axis=-1)
    
    grad_U = jax_gradient(U, dx, dy, f_1_grid)
    p_grad = jax_gradient(p, dx, dy, f_1_grid)
    viscous_term = jax_compute_viscous_term(U, dx, dy, Re, f_1_grid, f_2_grid)
    convective_term = jnp.stack([
        U[..., 0] * grad_U[..., 0, 0] + U[..., 1] * grad_U[..., 0, 1],
        U[..., 0] * grad_U[..., 1, 0] + U[..., 1] * grad_U[..., 1, 1]
    ], axis=-1)
    
    # Pressure-capillary route: capillary contribution is carried by pressure update.
    # Keep argument for API compatibility, but avoid explicit capillary forcing here.
    _ = surface_tension
    # RHS
    rhs_U = (-p_grad / rho_stacked + viscous_term / rho_stacked - convective_term)
    
    # Gravity (uniform - buoyancy comes from hydrostatic pressure gradient).
    # Classical Froude convention: Fr = U/sqrt(gL), so body force scales as g/Fr^2.
    fr2 = jnp.maximum(Fr * Fr, 1e-12)
    gravity_term = (1.0 / fr2) * jnp.stack([
        jnp.zeros_like(U[..., 0]), 
        g * jnp.ones_like(U[..., 1])
    ], axis=-1)
    rhs_U = jnp.where(include_gravity, rhs_U + gravity_term, rhs_U)
    
    U = U + current_dt * rhs_U
    U = jnp.where(ice_mask[..., jnp.newaxis], 0.0, U)
    return U


class FluidDynamicsSolver:
    """Fluid dynamics solver for droplet spreading simulation."""
    
    def __init__(self, rho1, rho2, Re1, Re2, Fr, g):
        self.rho1 = rho1
        self.rho2 = rho2
        self.Re1 = Re1
        self.Re2 = Re2
        self.Fr = Fr
        self.g = g
    
    def set_velocity_backend(self, backend):
        """Attach a velocity backend (currently a no-op; kept for API compatibility)."""
        self._backend = backend
    
    def update_velocity(self, U, p, surface_tension, current_dt, dx, dy,
                       phi, geometry, include_gravity=False, use_jax=True, psi=None):
        """Update velocity field. Grid is fluid-only (bottom-aligned)."""
        import jax.numpy as jnp
        if psi is None:
            psi = jnp.zeros_like(phi)
        elif not isinstance(psi, jnp.ndarray):
            psi = jnp.array(psi)
        U_updated = jax_update_velocity(U, p, surface_tension, current_dt, dx, dy,
                                       self.rho1, self.rho2, self.Re1, self.Re2,
                                       self.Fr, self.g, phi,
                                       geometry.f_1_grid, geometry.f_2_grid,
                                       include_gravity=include_gravity, psi=psi)
        return U_updated

    def check_continuity(self, U, dx, dy, geometry):
        """Check continuity. Grid is fluid-only (bottom-aligned)."""
        return jax_check_continuity(U, dx, dy, geometry.f_1_grid)
