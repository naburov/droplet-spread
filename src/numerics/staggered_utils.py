"""
Utilities to convert between collocated (cell-centered) velocity and MAC staggered faces.

Collocated:
  Ucc[i,j] = (u_x, u_y) at cell centers, shape (Nx, Ny, 2)

Staggered (MAC):
  u_face[i,j] at x-faces, shape (Nx+1, Ny)
  v_face[i,j] at y-faces, shape (Nx, Ny+1)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


@jit
def to_staggered(Ucc):
    """
    Convert collocated U (Nx,Ny,2) -> (u_face, v_face).

    Mapping:
      u_face[i,j] ~ Ux at face between cells i-1 and i
      v_face[i,j] ~ Uy at face between cells j-1 and j
    """
    Nx, Ny, _ = Ucc.shape
    Ux = Ucc[..., 0]
    Uy = Ucc[..., 1]

    u = jnp.zeros((Nx + 1, Ny), dtype=Ucc.dtype)
    v = jnp.zeros((Nx, Ny + 1), dtype=Ucc.dtype)

    # interior faces: average adjacent cell centers
    u = u.at[1:Nx, :].set(0.5 * (Ux[1:, :] + Ux[:-1, :]))
    v = v.at[:, 1:Ny].set(0.5 * (Uy[:, 1:] + Uy[:, :-1]))

    # boundary faces: copy nearest cell center (best-effort; BC application should override)
    u = u.at[0, :].set(Ux[0, :])
    u = u.at[Nx, :].set(Ux[-1, :])
    v = v.at[:, 0].set(Uy[:, 0])
    v = v.at[:, Ny].set(Uy[:, -1])

    return u, v


@jit
def to_collocated(u_face, v_face):
    """Convert MAC faces -> collocated U (Nx,Ny,2) by averaging to centers."""
    Nx1, Ny = u_face.shape
    Nx = Nx1 - 1
    Ny1 = v_face.shape[1]
    Nyc = Ny1 - 1
    # sanity: Nyc should equal Ny
    Ux = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    Uy = 0.5 * (v_face[:, 1:] + v_face[:, :-1])
    U = jnp.stack([Ux, Uy], axis=-1)
    return U

