"""
Pressure field calculations for droplet spreading simulation.

Pressure is split into two components:
- p_dynamic: from surface tension (capillary pressure)
- p_hydrostatic: from gravity (ρgh)

Total pressure: p = p_dynamic + p_hydrostatic
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from numerics.finite_differences import jax_divergence
from physics.properties import jax_calculate_density


@jit
def compute_hydrostatic_pressure(rho, g, dy, Fr, atm_pressure=0.0):
    """Compute hydrostatic pressure by integrating ρg from top.
    
    Hydrostatic equilibrium: dp/dy = ρ * g_y
    Integrating from top (p=atm) downward: p(y) = atm + ∫_{y}^{top} ρ|g|dy'
    
    Args:
        rho: Density field (Nx, Ny).
        g: Gravitational acceleration (negative for downward).
        dy: Grid spacing in y.
        Fr: Froude number.
        atm_pressure: Pressure at top boundary.
    
    Returns:
        Hydrostatic pressure field (Nx, Ny).
    """
    Nx, Ny = rho.shape
    
    # Initialize with atmospheric pressure at top
    p_hydro = jnp.zeros_like(rho)
    p_hydro = p_hydro.at[:, -1].set(atm_pressure)
    
    # Integrate from top to bottom: p[j] = p[j+1] - ρ[j] * g * dy
    # Note: g is negative (pointing down), so -g*dy is positive (pressure increases going down)
    # We use (1/Fr) scaling consistent with momentum equation
    def integrate_step(carry, j):
        p_prev = carry
        # Use density at current level for integration
        p_curr = p_prev - rho[:, j] * g * dy / Fr
        return p_curr, p_curr
    
    # Scan from Ny-2 down to 0
    indices = jnp.arange(Ny - 2, -1, -1)
    _, p_columns = jax.lax.scan(integrate_step, p_hydro[:, -1], indices)
    
    # Reconstruct full pressure field
    # p_columns[k] corresponds to index Ny-2-k
    p_hydro = p_hydro.at[:, :-1].set(p_columns[::-1].T)
    
    return p_hydro


def update_pressure_jax(surface_tension, dx, dy, geometry, rho1, rho2, phi, g, Fr, atm_pressure,
                        pressure_solver, include_gravity=False, has_dirichlet_bc=False,
                        dirichlet_rhs=None):
    """Update pressure. Terrain divergence of surface_tension. p_total = p_dynamic + p_hydrostatic.
    dirichlet_rhs: optional dict e.g. {"top": 0, "left": 0, "right": 0} so RHS at Dirichlet
    boundaries is set to the prescribed value (avoids spurious pressure gradients)."""
    sf_grad = jax_divergence(surface_tension, dx, dy, geometry.f_1_grid)
    # Set RHS at all Dirichlet boundaries to prescribed value so solution satisfies P = value there
    if dirichlet_rhs is not None:
        for side, value in dirichlet_rhs.items():
            if side == "top":
                sf_grad = sf_grad.at[:, -1].set(float(value))
            elif side == "bottom":
                sf_grad = sf_grad.at[:, 0].set(float(value))
            elif side == "left":
                sf_grad = sf_grad.at[0, :].set(float(value))
            elif side == "right":
                sf_grad = sf_grad.at[-1, :].set(float(value))
    else:
        sf_grad = sf_grad.at[:, -1].set(atm_pressure)
    sf_grad_np = np.array(sf_grad)

    pressure_solver.set_rhs(sf_grad_np)
    pressure_solver.solve()
    P_dynamic = jnp.array(pressure_solver.get_solution())

    if not has_dirichlet_bc:
        P_top_mean = float(jnp.mean(P_dynamic[:, -1]))
        if abs(P_top_mean - atm_pressure) > 1e-6:
            P_dynamic = P_dynamic - P_top_mean + atm_pressure

    if include_gravity:
        rho = jax_calculate_density(phi, rho1, rho2)
        P_hydrostatic = compute_hydrostatic_pressure(rho, g, dy, Fr, atm_pressure)
        return P_dynamic + P_hydrostatic
    return P_dynamic


class PressureSolver:
    """Pressure solver for droplet spreading simulation."""
    
    def __init__(self, rho1, rho2, g, atm_pressure, Fr=1.0, include_gravity=False):
        self.rho1 = rho1
        self.rho2 = rho2
        self.g = g
        self.atm_pressure = atm_pressure
        self.Fr = Fr
        self.include_gravity = include_gravity
    
    def update_pressure(self, surface_tension, dx, dy, geometry, phi, pressure_solver, has_dirichlet_bc=False,
                        dirichlet_rhs=None):
        """Update pressure. geometry: from state (terrain divergence of ST).
        dirichlet_rhs: optional dict {side: value} so RHS at Dirichlet boundaries matches BC (avoids spurious gradients)."""
        return update_pressure_jax(
            surface_tension, dx, dy, geometry, self.rho1, self.rho2, phi,
            self.g, self.Fr, self.atm_pressure, pressure_solver,
            include_gravity=self.include_gravity,
            has_dirichlet_bc=has_dirichlet_bc,
            dirichlet_rhs=dirichlet_rhs
        )
