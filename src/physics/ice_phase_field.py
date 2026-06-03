"""
Ice phase field equations for water-ice phase transition.

Supports both Allen-Cahn (non-conservative) and Cahn-Hilliard (conservative) equations.
"""

import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_gradient, jax_laplacian
from physics.properties import jax_df_2, jax_advection_function


@jit
def jax_update_ice_phase_allen_cahn(psi, T, U, current_dt, dx, dy, geometry, M_psi, epsilon_psi,
                                     T_melt, transition_width, phi=None, threshold=0.1):
    """Allen-Cahn equation for ice phase field.
    
    ∂ψ/∂t = -M_ψ[f'(ψ) - ε²∇²ψ] - u·∇ψ + λ(T_melt - T)
    
    Args:
        psi: Ice phase field (-1=water, +1=ice).
        T: Temperature field.
        U: Velocity field.
        current_dt: Time step.
        dx, dy: Grid spacing.
        M_psi: Mobility coefficient.
        epsilon_psi: Interface thickness.
        T_melt: Melting temperature.
        transition_width: Temperature width for phase transition.
        phi: Liquid-gas phase field (optional, to restrict ice to water).
    
    Returns:
        Updated ice phase field.
    """
    grad_psi = jax_gradient(psi, dx, dy, geometry.f_1_grid)
    A = jax_advection_function(psi, threshold=threshold)
    convective_term = A * (U[..., 0] * grad_psi[..., 0] + U[..., 1] * grad_psi[..., 1])
    lap_psi = jax_laplacian(psi, dx, dy, geometry.f_1_grid, geometry.f_2_grid)
    df_dpsi = jax_df_2(psi)
    
    # Allen-Cahn relaxation
    relaxation_term = -M_psi * (df_dpsi - epsilon_psi**2 * lap_psi)
    
    # Temperature coupling: λ(T_melt - T) drives freezing/melting
    T_diff = T_melt - T
    lambda_coupling = M_psi / (epsilon_psi * transition_width)
    temperature_coupling = lambda_coupling * T_diff
    
    rhs_psi = relaxation_term - convective_term + temperature_coupling
    psi_new = psi + current_dt * rhs_psi
    psi_new = jnp.clip(psi_new, -1.0, 1.0)
    
    # Restrict ice to water regions (phi < 0)
    if phi is not None:
        water_mask = phi < 0
        psi_new = jnp.where(water_mask, psi_new, -1.0)
    
    return psi_new


@jit
def jax_update_ice_phase_cahn_hilliard(psi, T, U, current_dt, dx, dy, geometry, Pe_psi, epsilon_psi,
                                        T_melt, transition_width, phi=None, T_tolerance=0.1, threshold=0.1):
    """Cahn-Hilliard equation for ice phase field (conservative).
    
    ∂ψ/∂t = ∇·(M∇μ) - u·∇ψ, where μ = f'(ψ) - ε²∇²ψ + coupling(T)
    
    Args:
        psi: Ice phase field (-1=water, +1=ice).
        T: Temperature field.
        U: Velocity field.
        current_dt: Time step.
        dx, dy: Grid spacing.
        Pe_psi: Peclet number (controls mobility).
        epsilon_psi: Interface thickness.
        T_melt: Melting temperature.
        transition_width: Temperature width for phase transition.
        phi: Liquid-gas phase field (optional, to restrict ice to water).
    
    Returns:
        Updated ice phase field.
    """
    grad_psi = jax_gradient(psi, dx, dy, geometry.f_1_grid)
    A = 0.5 * (1.0 - jnp.tanh(psi / 0.1))
    convective_term = A * (U[..., 0] * grad_psi[..., 0] + U[..., 1] * grad_psi[..., 1])
    lap_psi = jax_laplacian(psi, dx, dy, geometry.f_1_grid, geometry.f_2_grid)
    df_dpsi = jax_df_2(psi)
    
    # Temperature coupling with tolerance for near-T_melt
    T_diff = T_melt - T
    T_diff_effective = jnp.where(
        jnp.abs(T_diff) < T_tolerance,
        jnp.sign(T_diff) * T_tolerance,
        T_diff
    )
    lambda_coupling = 1.0 / (epsilon_psi * transition_width)
    temperature_coupling = -lambda_coupling * T_diff_effective
    
    # Phase field potential
    phase_field_potential = df_dpsi - epsilon_psi**2 * lap_psi
    
    from boundary_conditions.chemical_potential_bc import jax_apply_chemical_potential_zero_flux_bc
    phase_field_potential = jax_apply_chemical_potential_zero_flux_bc(phase_field_potential, dx, dy)
    
    lagrange_multiplier = jnp.mean(phase_field_potential)
    
    # Cahn-Hilliard source term
    source_term = -1.0 / Pe_psi * (phase_field_potential - lagrange_multiplier + temperature_coupling)
    
    rhs_psi = -convective_term + source_term
    psi_new = psi + current_dt * rhs_psi
    psi_new = jnp.clip(psi_new, -1.0, 1.0)
    
    # Restrict ice to water regions
    if phi is not None:
        water_mask = phi < 0
        psi_new = jnp.where(water_mask, psi_new, -1.0)
    
    return psi_new


class IcePhaseFieldSolver:
    """Ice phase field solver for water-ice phase transition."""
    
    def __init__(self, M_psi, epsilon_psi, T_melt, config=None):
        self.M_psi = M_psi
        self.epsilon_psi = epsilon_psi
        self.T_melt = T_melt
        self.config = config
        
        if config is not None:
            ice_params = config.get("ice_water_params", {})
            self.use_allen_cahn = ice_params.get("use_allen_cahn", True)
            self.Pe_psi = ice_params.get("Pe_psi", 10.0)
            self.transition_width = ice_params.get("transition_width", 1.0)
            self.T_tolerance = ice_params.get("T_tolerance", 0.1)
            self.advection_threshold = ice_params.get("advection_threshold", 0.1)
            
            from boundary_conditions.ice_phase_field_bc import IcePhaseFieldBoundaryConditions
            self.bc_manager = IcePhaseFieldBoundaryConditions(config)
        else:
            self.use_allen_cahn = True
            self.Pe_psi = 10.0
            self.transition_width = 1.0
            self.T_tolerance = 0.1
            self.advection_threshold = 0.1
            self.bc_manager = None
    
    def update(self, psi, T, U, current_dt, dx, dy, geometry, use_jax=True, phi=None):
        """Update ice phase field. geometry: from state."""
        if self.use_allen_cahn:
            psi_new = jax_update_ice_phase_allen_cahn(
                psi, T, U, current_dt, dx, dy, geometry,
                self.M_psi, self.epsilon_psi, self.T_melt, self.transition_width,
                phi=phi, threshold=self.advection_threshold
            )
        else:
            psi_new = jax_update_ice_phase_cahn_hilliard(
                psi, T, U, current_dt, dx, dy, geometry,
                self.Pe_psi, self.epsilon_psi, self.T_melt, self.transition_width,
                phi=phi, T_tolerance=self.T_tolerance, threshold=self.advection_threshold
            )
        
        if self.bc_manager is not None:
            psi_new = self.bc_manager.apply_boundary_conditions(psi_new, T, dx, dy, use_jax=use_jax)
        
        return psi_new
