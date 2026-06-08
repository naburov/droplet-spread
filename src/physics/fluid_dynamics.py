"""
Fluid dynamics equations for droplet spreading simulation.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import (
    jax_gradient,
    jax_divergence,
    jax_laplacian,
    jax_dx,
    jax_dy,
)
from physics.properties import jax_calculate_density, jax_calculate_reynolds_number


@jit
def jax_compute_viscous_stress_divergence(U, dx, dy, mu, f_1_grid):
    """∇·τ for τ = μ(∇u + ∇uᵀ) with physical velocity components on terrain grid."""
    u = U[..., 0]
    v = U[..., 1]
    h1 = f_1_grid

    u_xi = jax_dx(u, h=dx)
    u_eta = jax_dy(u, h=dy)
    v_xi = jax_dx(v, h=dx)
    v_eta = jax_dy(v, h=dy)

    u_X = u_xi - h1 * u_eta
    u_Y = u_eta
    v_X = v_xi - h1 * v_eta
    v_Y = v_eta

    tau_xx = 2.0 * mu * u_X
    tau_xy = mu * (u_Y + v_X)
    tau_yy = 2.0 * mu * v_Y

    div_tau_x = (
        jax_dx(tau_xx, h=dx)
        - h1 * jax_dy(tau_xx, h=dy)
        + jax_dy(tau_xy, h=dy)
    )
    div_tau_y = (
        jax_dx(tau_xy, h=dx)
        - h1 * jax_dy(tau_xy, h=dy)
        + jax_dy(tau_yy, h=dy)
    )
    return jnp.stack([div_tau_x, div_tau_y], axis=-1)


def _effective_viscosity(Re, rho, mu_convention):
    inv_re = 1.0 / jnp.maximum(Re, 1e-12)
    if mu_convention == "rho_over_re":
        return rho * inv_re
    return inv_re


@partial(jit, static_argnums=(6, 8))
def jax_compute_viscous_term(
    U, dx, dy, Re, f_1_grid, f_2_grid, viscous_form="component_laplacian",
    rho=None, mu_convention="inv_re",
):
    """Viscous term for momentum RHS (before division by density).

    component_laplacian: legacy (1/Re)∇²U via terrain scalar Laplacian.
    stress_divergence: ∇·τ with τ = μ(∇u + ∇uᵀ), μ from Re (and optionally ρ).
    """
    if viscous_form == "component_laplacian":
        return jnp.stack(
            [
                jax_laplacian(U[..., 0], dx, dy, f_1_grid, f_2_grid) / Re,
                jax_laplacian(U[..., 1], dx, dy, f_1_grid, f_2_grid) / Re,
            ],
            axis=-1,
        )

    if rho is None:
        rho = jnp.ones_like(Re)
    mu = _effective_viscosity(Re, rho, mu_convention)
    return jax_compute_viscous_stress_divergence(U, dx, dy, mu, f_1_grid)


@jit
def jax_check_continuity(U, dx, dy, f_1_grid):
    """Check continuity equation. Curvilinear (x, eta): fluid-only grid. f_1_grid: terrain gradient (Nx, Ny)."""
    divergence_field = jax_divergence(U, dx, dy, f_1_grid)
    max_div = jnp.max(jnp.abs(divergence_field))
    mean_div = jnp.mean(jnp.abs(divergence_field))
    return divergence_field, max_div, mean_div


@partial(jit, static_argnums=(15, 16, 17))
def jax_update_velocity(U, p, surface_tension, current_dt, dx, dy, rho1, rho2,
                        Re1, Re2, Fr, g, phi, f_1_grid, f_2_grid,
                        include_gravity=False, psi=None,
                        viscous_form="component_laplacian", mu_convention="inv_re"):
    """Update velocity (predictor step). Grid is fluid-only (bottom-aligned). Ice (psi>0) zeroed when present."""
    if psi is None:
        psi = jnp.zeros_like(phi)
    ice_mask = psi > 0.0
    U = jnp.where(ice_mask[..., jnp.newaxis], 0.0, U)

    # Material properties
    Re = jax_calculate_reynolds_number(phi, Re1, Re2)
    rho = jax_calculate_density(phi, rho1, rho2)
    rho = jnp.maximum(rho, 1e-6)
    rho_stacked = jnp.stack([rho, rho], axis=-1)
    
    grad_U = jax_gradient(U, dx, dy, f_1_grid)
    p_grad = jax_gradient(p, dx, dy, f_1_grid)
    viscous_term = jax_compute_viscous_term(
        U, dx, dy, Re, f_1_grid, f_2_grid, viscous_form, rho=rho, mu_convention=mu_convention,
    )
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
    
    def __init__(self, rho1, rho2, Re1, Re2, Fr, g, config=None):
        self.rho1 = rho1
        self.rho2 = rho2
        self.Re1 = Re1
        self.Re2 = Re2
        self.Fr = Fr
        self.g = g
        from diagnostics.viscous_form import normalize_mu_convention, normalize_viscous_form

        solver_params = (config or {}).get("solver_params", {})
        self.viscous_form = normalize_viscous_form(solver_params.get("viscous_form"))
        self.mu_convention = normalize_mu_convention(
            solver_params.get("viscous_mu_convention")
        )
    
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
        U_updated = jax_update_velocity(
            U, p, surface_tension, current_dt, dx, dy,
            self.rho1, self.rho2, self.Re1, self.Re2,
            self.Fr, self.g, phi,
            geometry.f_1_grid, geometry.f_2_grid,
            include_gravity=include_gravity, psi=psi,
            viscous_form=self.viscous_form, mu_convention=self.mu_convention,
        )
        return U_updated

    def check_continuity(self, U, dx, dy, geometry):
        """Check continuity. Grid is fluid-only (bottom-aligned)."""
        return jax_check_continuity(U, dx, dy, geometry.f_1_grid)
