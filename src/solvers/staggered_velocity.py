"""
Staggered-grid (MAC) velocity predictor helper for the main two-phase simulation.

This module advances MAC face velocities (u, v) using the same operators as the
standalone staggered solvers. The main simulation is responsible for:
  - converting between collocated U and faces when needed, and
  - enforcing boundary conditions (currently via the collocated BC machinery).

When P and geometry are provided, the pressure gradient -grad P/rho is included.
That is required for gravity to produce down-slope motion on a tilted surface:
hydrostatic P balances gravity in the normal direction, leaving the along-slope
component; without -grad P in the predictor that component never appears.
"""

from __future__ import annotations

import jax.numpy as jnp

from numerics.staggered_mac import advect_u, advect_v, laplacian_u, laplacian_v, grad_p_to_faces


def staggered_predictor_step(
    u,
    v,
    surface_tension,
    dt: float,
    dx: float,
    dy: float,
    Re2: float,
    Fr: float,
    g: float,
    include_gravity: bool = False,
    include_advection: bool = True,
    P=None,
    phi=None,
    rho1=None,
    rho2=None,
    geometry=None,
):
    """
    Advance velocity one predictor step using MAC operators.

    Args:
        u, v: MAC face velocities (u: (Nx+1, Ny), v: (Nx, Ny+1)) as JAX arrays.
        surface_tension: Surface tension force at cell centers, shape (Nx, Ny, 2).
        dt, dx, dy: Time step and grid spacing.
        Re2: Reynolds number of the outer phase (used as an effective Re).
        Fr, g: Froude number and gravity parameter for vertical forcing.
        include_gravity: Whether to include gravity in the predictor.
        include_advection: Whether to include convective term.
        P, phi, rho1, rho2, geometry: Optional. When provided, -grad P/rho is added
            (terrain gradient from geometry). Needed for gravity-driven sliding on tilted surfaces.

    Returns:
        Tuple of updated MAC face velocities (u_new, v_new).
    """
    # Advection
    if include_advection:
        Au = advect_u(u, v, dx, dy)
        Av = advect_v(u, v, dx, dy)
    else:
        Au = jnp.zeros_like(u)
        Av = jnp.zeros_like(v)

    # Viscous diffusion with a single effective kinematic viscosity ν ≈ 1 / Re2
    nu = 1.0 / max(Re2, 1e-6)
    Lu = laplacian_u(u, dx, dy)
    Lv = laplacian_v(v, dx, dy)

    u_star = u + dt * (-Au + nu * Lu)
    v_star = v + dt * (-Av + nu * Lv)

    # Pressure-capillary route: capillary effects are represented via pressure update.
    # Keep explicit surface_tension argument for API compatibility, but do not add
    # direct capillary forcing in the predictor to avoid double counting.
    _ = surface_tension

    # Pressure gradient -grad P/rho.
    # For flat MAC runs, use face-consistent discretization:
    #   u += -dt * (1/rho_u_face) * dpdx_face, v += -dt * (1/rho_v_face) * dpdy_face
    # to stay consistent with variable-density projection.
    if P is not None and phi is not None and rho1 is not None and rho2 is not None and geometry is not None:
        from physics.properties import jax_calculate_density
        Nx, Ny = u.shape[0] - 1, u.shape[1]
        rho = jax_calculate_density(phi, rho1, rho2)
        rho = jnp.maximum(rho, 1e-6)
        if getattr(geometry, "has_geometry", False):
            from numerics.finite_differences import jax_gradient

            grad_P = jax_gradient(P, dx, dy, geometry.f_1_grid)
            acc = -grad_P / rho[..., jnp.newaxis]
            ax = acc[..., 0]
            ay = acc[..., 1]
            ax_u = jnp.zeros((Nx + 1, Ny), dtype=ax.dtype)
            ax_u = ax_u.at[1:Nx, :].set(0.5 * (ax[1:, :] + ax[:-1, :]))
            ay_v = jnp.zeros((Nx, Ny + 1), dtype=ay.dtype)
            ay_v = ay_v.at[:, 1:Ny].set(0.5 * (ay[:, 1:] + ay[:, :-1]))
        else:
            inv_rho = 1.0 / rho
            inv_rho_u = jnp.zeros((Nx + 1, Ny), dtype=inv_rho.dtype)
            inv_rho_v = jnp.zeros((Nx, Ny + 1), dtype=inv_rho.dtype)
            inv_rho_u = inv_rho_u.at[1:Nx, :].set(0.5 * (inv_rho[1:, :] + inv_rho[:-1, :]))
            inv_rho_u = inv_rho_u.at[0, :].set(inv_rho[0, :])
            inv_rho_u = inv_rho_u.at[Nx, :].set(inv_rho[-1, :])
            inv_rho_v = inv_rho_v.at[:, 1:Ny].set(0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1]))
            inv_rho_v = inv_rho_v.at[:, 0].set(inv_rho[:, 0])
            inv_rho_v = inv_rho_v.at[:, Ny].set(inv_rho[:, -1])

            dpdx_face, dpdy_face = grad_p_to_faces(P, dx, dy)
            ax_u = -inv_rho_u * dpdx_face
            ay_v = -inv_rho_v * dpdy_face
        u_star = u_star + dt * ax_u
        v_star = v_star + dt * ay_v

    # Gravity (vertical component in (x, eta)); pressure gradient above provides down-slope component when tilted.
    # Classical Froude convention: Fr = U/sqrt(gL), so body force scales as g/Fr^2.
    if include_gravity:
        gravity_v = (1.0 / max(Fr * Fr, 1e-12)) * g * jnp.ones_like(v_star)
        v_star = v_star + dt * gravity_v

    return u_star, v_star

