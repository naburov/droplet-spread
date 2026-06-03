"""
Temperature evolution equation for water-ice phase transition.
"""

import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_gradient, jax_laplacian
from physics.properties import jax_advection_function


@jit
def jax_update_temperature(T, psi, U, current_dt, dx, dy, geometry, alpha_water, alpha_ice,
                           L, c_p_water, c_p_ice, T_melt, threshold=0.1):
    """Temperature evolution (simplified version without explicit latent heat).
    
    ∂T/∂t = -A(ψ)u·∇T + ∇·(α∇T)
    
    Args:
        T: Temperature field.
        psi: Ice phase field.
        U: Velocity field.
        current_dt: Time step.
        dx, dy: Grid spacing.
        alpha_water, alpha_ice: Thermal diffusivities.
        L: Latent heat (unused in this version).
        c_p_water, c_p_ice: Specific heats (unused in this version).
        T_melt: Melting temperature (unused in this version).
    
    Returns:
        Updated temperature.
    """
    psi_mapped = (psi + 1) / 2.0
    alpha = alpha_water * (1 - psi_mapped) + alpha_ice * psi_mapped
    c_p = c_p_water * (1 - psi_mapped) + c_p_ice * psi_mapped
    
    grad_T = jax_gradient(T, dx, dy, geometry.f_1_grid)
    advective_term = U[..., 0] * grad_T[..., 0] + U[..., 1] * grad_T[..., 1]
    lap_T = jax_laplacian(T, dx, dy, geometry.f_1_grid, geometry.f_2_grid)
    grad_alpha = jax_gradient(alpha, dx, dy, geometry.f_1_grid)
    diffusion_term = alpha * lap_T + grad_alpha[..., 0] * grad_T[..., 0] + grad_alpha[..., 1] * grad_T[..., 1]
    
    rhs_T = -advective_term + diffusion_term
    T_new = T + current_dt * rhs_T
    
    return T_new


@jit
def jax_update_temperature_with_latent_heat(T, psi, psi_old, U, current_dt, dx, dy, geometry,
                                            alpha_water, alpha_ice, L, c_p_water, c_p_ice, T_melt, threshold=0.1):
    """Temperature update with explicit latent heat from phase change.
    
    ∂T/∂t = -A(ψ)u·∇T + ∇·(α∇T) + (L/c_p)(1/2)∂ψ/∂t
    
    Args:
        T: Temperature field.
        psi: Current ice phase field.
        psi_old: Previous ice phase field.
        U: Velocity field.
        current_dt: Time step.
        dx, dy: Grid spacing.
        alpha_water, alpha_ice: Thermal diffusivities.
        L: Latent heat.
        c_p_water, c_p_ice: Specific heats.
        T_melt: Melting temperature.
    
    Returns:
        Updated temperature.
    """
    psi_mapped = (psi + 1) / 2.0
    alpha = alpha_water * (1 - psi_mapped) + alpha_ice * psi_mapped
    c_p = c_p_water * (1 - psi_mapped) + c_p_ice * psi_mapped
    
    grad_T = jax_gradient(T, dx, dy, geometry.f_1_grid)
    A = jax_advection_function(psi, threshold=threshold)
    advective_term = A * (U[..., 0] * grad_T[..., 0] + U[..., 1] * grad_T[..., 1])
    lap_T = jax_laplacian(T, dx, dy, geometry.f_1_grid, geometry.f_2_grid)
    grad_alpha = jax_gradient(alpha, dx, dy, geometry.f_1_grid)
    diffusion_term = alpha * lap_T + grad_alpha[..., 0] * grad_T[..., 0] + grad_alpha[..., 1] * grad_T[..., 1]
    
    # Latent heat: freezing (dpsi_dt > 0) releases heat
    dpsi_dt = (psi - psi_old) / current_dt
    latent_heat_term = 0.5 * (L / c_p) * dpsi_dt
    
    rhs_T = -advective_term + diffusion_term + latent_heat_term
    T_new = T + current_dt * rhs_T
    T_new = jnp.clip(T_new, 200.0, 400.0)
    
    return T_new


class TemperatureSolver:
    """Temperature solver with latent heat coupling."""
    
    def __init__(self, alpha_water, alpha_ice, L, c_p_water, c_p_ice, T_melt, config=None):
        self.alpha_water = alpha_water
        self.alpha_ice = alpha_ice
        self.L = L
        self.c_p_water = c_p_water
        self.c_p_ice = c_p_ice
        self.T_melt = T_melt
        self.config = config
        
        if config is not None:
            ice_params = config.get("ice_water_params", {})
            self.include_latent_heat = ice_params.get("include_latent_heat", True)
            self.T_min = ice_params.get("T_min", 200.0)
            self.T_max = ice_params.get("T_max", 400.0)
            self.advection_threshold = ice_params.get("advection_threshold", 0.1)
            
            from boundary_conditions.temperature_bc import TemperatureBoundaryConditions
            self.bc_manager = TemperatureBoundaryConditions(config)
        else:
            self.include_latent_heat = True
            self.T_min = 200.0
            self.T_max = 400.0
            self.advection_threshold = 0.1
            self.bc_manager = None
    
    def update(self, T, psi, U, current_dt, dx, dy, geometry, psi_old=None, use_jax=True):
        """Update temperature field. geometry: from state."""
        if self.bc_manager is not None:
            T = self.bc_manager.apply_boundary_conditions(T, dx, dy, use_jax=use_jax)
        if psi_old is not None and self.include_latent_heat:
            T_new = jax_update_temperature_with_latent_heat(
                T, psi, psi_old, U, current_dt, dx, dy, geometry,
                self.alpha_water, self.alpha_ice, self.L,
                self.c_p_water, self.c_p_ice, self.T_melt,
                threshold=self.advection_threshold
            )
        else:
            T_new = jax_update_temperature(
                T, psi, U, current_dt, dx, dy, geometry,
                self.alpha_water, self.alpha_ice, self.L,
                self.c_p_water, self.c_p_ice, self.T_melt,
                threshold=self.advection_threshold
            )
        
        # Apply BCs after update
        if self.bc_manager is not None:
            T_new = self.bc_manager.apply_boundary_conditions(T_new, dx, dy, use_jax=use_jax)
        
        # Clip temperature to reasonable range
        T_new = jnp.clip(T_new, self.T_min, self.T_max)
        
        return T_new
