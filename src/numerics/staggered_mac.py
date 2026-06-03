"""
Staggered (MAC) grid operators for incompressible flow.

Layout (2D):
  - p[i,j]   at cell centers, shape (Nx, Ny)
  - u[i,j]   at x-faces,     shape (Nx+1, Ny)   (i = 0..Nx)
  - v[i,j]   at y-faces,     shape (Nx, Ny+1)   (j = 0..Ny)

JAX ops are JIT-compiled; pressure solve uses NumPy/pyamg and converts at the boundary.
"""

from __future__ import annotations

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit

# ---- JAX (JIT) MAC operators ----

@partial(jit, static_argnums=(0, 1))
def zeros_mac(Nx: int, Ny: int):
    """Return (u, v, p) zero fields for MAC grid. JAX arrays. Nx, Ny must be Python ints (static)."""
    u = jnp.zeros((Nx + 1, Ny))
    v = jnp.zeros((Nx, Ny + 1))
    p = jnp.zeros((Nx, Ny))
    return u, v, p


@jit
def divergence(u, v, dx: float, dy: float):
    """Divergence at cell centers from face velocities. (u,v) JAX arrays."""
    du_dx = (u[1:, :] - u[:-1, :]) / dx
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy
    return du_dx + dv_dy


@jit
def grad_p_to_faces(p, dx: float, dy: float):
    """Pressure gradient to faces: (dpdx_on_u, dpdy_on_v). p is JAX array."""
    Nx, Ny = p.shape
    dpdx = jnp.zeros((Nx + 1, Ny), dtype=p.dtype)
    dpdx = dpdx.at[1:Nx, :].set((p[1:, :] - p[:-1, :]) / dx)
    dpdy = jnp.zeros((Nx, Ny + 1), dtype=p.dtype)
    dpdy = dpdy.at[:, 1:Ny].set((p[:, 1:] - p[:, :-1]) / dy)
    return dpdx, dpdy


@jit
def laplacian_u(u, dx: float, dy: float):
    """Vector Laplacian on u-faces (interior)."""
    out = jnp.zeros_like(u)
    out = out.at[1:-1, 1:-1].set(
        (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
        + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    )
    return out


@jit
def laplacian_v(v, dx: float, dy: float):
    """Vector Laplacian on v-faces (interior)."""
    out = jnp.zeros_like(v)
    out = out.at[1:-1, 1:-1].set(
        (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / (dx * dx)
        + (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) / (dy * dy)
    )
    return out


@jit
def advect_u(u, v, dx: float, dy: float):
    """Upwind advection for u on u-faces (vectorized)."""
    Nx1, Ny = u.shape
    Nx = Nx1 - 1
    out = jnp.zeros_like(u)
    u_c = u[1:Nx, 1 : Ny - 1]
    ux = u_c
    vy = 0.25 * (
        v[0 : Nx - 1, 1 : Ny - 1]
        + v[0 : Nx - 1, 2:Ny]
        + v[1:Nx, 1 : Ny - 1]
        + v[1:Nx, 2:Ny]
    )
    du_dx_pos = (u[1:Nx, 1 : Ny - 1] - u[0 : Nx - 1, 1 : Ny - 1]) / dx
    du_dx_neg = (u[2 : Nx + 1, 1 : Ny - 1] - u[1:Nx, 1 : Ny - 1]) / dx
    du_dx = jnp.where(ux >= 0, du_dx_pos, du_dx_neg)
    du_dy_pos = (u[1:Nx, 1 : Ny - 1] - u[1:Nx, 0 : Ny - 2]) / dy
    du_dy_neg = (u[1:Nx, 2:Ny] - u[1:Nx, 1 : Ny - 1]) / dy
    du_dy = jnp.where(vy >= 0, du_dy_pos, du_dy_neg)
    conv = ux * du_dx + vy * du_dy
    out = out.at[1:Nx, 1 : Ny - 1].set(conv)
    return out


@jit
def advect_v(u, v, dx: float, dy: float):
    """Upwind advection for v on v-faces (vectorized)."""
    Nx, Ny1 = v.shape
    Ny = Ny1 - 1
    out = jnp.zeros_like(v)
    v_c = v[1 : Nx - 1, 1:Ny]
    vy = v_c
    ux = 0.25 * (
        u[1 : Nx - 1, 0 : Ny - 1]
        + u[2:Nx, 0 : Ny - 1]
        + u[1 : Nx - 1, 1:Ny]
        + u[2:Nx, 1:Ny]
    )
    dv_dx_pos = (v[1 : Nx - 1, 1:Ny] - v[0 : Nx - 2, 1:Ny]) / dx
    dv_dx_neg = (v[2:Nx, 1:Ny] - v[1 : Nx - 1, 1:Ny]) / dx
    dv_dx = jnp.where(ux >= 0, dv_dx_pos, dv_dx_neg)
    dv_dy_pos = (v[1 : Nx - 1, 1:Ny] - v[1 : Nx - 1, 0 : Ny - 1]) / dy
    dv_dy_neg = (v[1 : Nx - 1, 2 : Ny + 1] - v[1 : Nx - 1, 1:Ny]) / dy
    dv_dy = jnp.where(vy >= 0, dv_dy_pos, dv_dy_neg)
    conv = ux * dv_dx + vy * dv_dy
    out = out.at[1 : Nx - 1, 1:Ny].set(conv)
    return out


@partial(jit, static_argnums=(3, 4, 5))
def apply_velocity_bcs(
    u,
    v,
    u_inlet_profile,
    top_bc: str = "no_slip",  # "no_slip" | "free_slip" | "open"
    bottom_bc: str = "no_slip",  # "no_slip" | "free_slip" | "open"
    outflow_right: bool = True,
):
    """Apply BCs: left inlet, right outflow, top/bottom no-slip or free-slip. Returns (u, v)."""
    Nx1, Ny = u.shape
    Nx = Nx1 - 1
    u = u.at[0, :].set(u_inlet_profile)
    v = v.at[0, :].set(0.0)
    if outflow_right:
        u = u.at[Nx, :].set(u[Nx - 1, :])
        v = v.at[Nx - 1, :].set(v[Nx - 2, :])
    # bottom wall (y=0): v[:,0] is boundary face, u[:,0] adjacent to wall
    if bottom_bc == "no_slip":
        v = v.at[:, 0].set(0.0)
        u = u.at[:, 0].set(0.0)
    elif bottom_bc == "free_slip":
        v = v.at[:, 0].set(0.0)
        u = u.at[:, 0].set(u[:, 1])  # du/dy = 0 at wall-adjacent row
    elif bottom_bc == "open":
        # allow penetration: dv/dy = 0, du/dy = 0 at boundary
        v = v.at[:, 0].set(v[:, 1])
        u = u.at[:, 0].set(u[:, 1])

    # top wall (y=Ly): v[:,-1] boundary face, u[:,-1] adjacent to wall
    if top_bc == "no_slip":
        v = v.at[:, -1].set(0.0)
        u = u.at[:, -1].set(0.0)
    elif top_bc == "free_slip":
        v = v.at[:, -1].set(0.0)
        u = u.at[:, -1].set(u[:, -2])  # du/dy = 0 at wall-adjacent row
    elif top_bc == "open":
        # allow penetration: dv/dy = 0, du/dy = 0 at boundary
        v = v.at[:, -1].set(v[:, -2])
        u = u.at[:, -1].set(u[:, -2])
    return u, v


# ---- Pressure Poisson: pyamg (NumPy) ----

def make_ppe_solver_pyamg(Nx: int, Ny: int, dx: float, dy: float):
    """Build a sparse PPE solver: Neumann left/top/bottom, Dirichlet p=0 on right."""
    from solvers.sparse_solver import SparseSolverWrapper

    solver = SparseSolverWrapper(
        Nx, Ny, dx, dy, backend="pyamg", solver_params={"tol": 1e-8, "maxiter": 200}
    )
    solver.set_bcs(
        left="neumann", right="dirichlet", top="neumann", bottom="neumann"
    )
    return solver


def solve_pressure_poisson_pyamg(
    rhs, dx: float, dy: float, solver, p0=None
) -> tuple[np.ndarray, int, float]:
    """
    Solve ∇²p = rhs with Neumann (left/top/bottom) and Dirichlet p=0 (right).
    rhs: (Nx, Ny) array (NumPy or JAX; converted to NumPy).
    solver: from make_ppe_solver_pyamg(Nx, Ny, dx, dy).
    Returns (p, iters_used, residual); iters/residual from AMG are not exposed, so iters=-1, res=0.
    """
    rhs_np = np.asarray(rhs, dtype=np.float64).copy()
    Nx, Ny = rhs_np.shape
    rhs_np[-1, :] = 0.0  # Dirichlet p=0 on right column
    x0 = np.asarray(p0, dtype=np.float64).copy() if p0 is not None else None
    if x0 is not None:
        x0[-1, :] = 0.0
    p = solver.solve(rhs_np, x0=x0)
    return p, -1, 0.0


# ---- Legacy SOR (NumPy, slow) ----

def solve_pressure_poisson_sor(
    rhs: np.ndarray,
    dx: float,
    dy: float,
    *,
    p0: np.ndarray | None = None,
    omega: float = 1.7,
    max_iter: int = 5000,
    tol: float = 1e-6,
):
    """
    Solve ∇²p = rhs using SOR. Neumann left/top/bottom, Dirichlet p=0 right.
    Returns (p, iters, residual_linf). Kept for fallback; prefer solve_pressure_poisson_pyamg.
    """
    Nx, Ny = rhs.shape
    p = np.zeros_like(rhs) if p0 is None else np.asarray(p0, dtype=np.float64).copy()

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    denom = 2.0 * (inv_dx2 + inv_dy2)

    def apply_neumann(p_):
        p_[0, :] = p_[1, :]
        p_[:, 0] = p_[:, 1]
        p_[:, -1] = p_[:, -2]
        return p_

    for it in range(1, max_iter + 1):
        p = apply_neumann(p)
        p[-1, :] = 0.0
        max_res = 0.0
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                lap = (
                    (p[i + 1, j] + p[i - 1, j]) * inv_dx2
                    + (p[i, j + 1] + p[i, j - 1]) * inv_dy2
                )
                p_new = (lap - rhs[i, j]) / denom
                p_old = p[i, j]
                p[i, j] = (1.0 - omega) * p_old + omega * p_new
                r = (
                    (p[i + 1, j] - 2.0 * p[i, j] + p[i - 1, j]) * inv_dx2
                    + (p[i, j + 1] - 2.0 * p[i, j] + p[i, j - 1]) * inv_dy2
                    - rhs[i, j]
                )
                max_res = max(max_res, abs(r))
        if max_res < tol:
            return p, it, max_res
    return p, max_iter, max_res
