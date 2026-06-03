"""
Finite difference methods for numerical derivatives.

Primary operators (used in physics/solvers): JAX-compiled jax_*.
- jax_gradient(f, dx, dy, f_1_grid): gradient in terrain coords (φ_x - f' φ_η, φ_η).
  For flat geometry (f_1_grid=0) this is Cartesian. There is no separate "gradient_terrain".
- jax_divergence(f, dx, dy, f_1_grid): terrain divergence u_x - f' u_η + v_η.
- jax_laplacian(f, dx, dy, f_1_grid, f_2_grid): terrain Laplacian.

NumPy helpers (Cartesian-only): numerical_derivative, laplacian, divergence, gradient, norm.
Used for visualization (e.g. plotting) and legacy scripts; not in the main simulation path.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


# --- NumPy helpers (Cartesian-only; used in plotting/scripts) ---

def numerical_derivative(f, axis=0, h=1e-5, dtype='float32'):
    """Calculate the numerical derivative of f along a specified axis using central difference.
    
    Args:
        f (np.ndarray): Function to differentiate.
        axis (int): Axis along which to differentiate.
        h (float): Step size.
        dtype (str): Data type for the result.
    
    Returns:
        np.ndarray: Numerical derivative.
    """
    return ((np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)).astype(dtype)


@jit
def jax_dx(f, h=1e-5):
    """JAX-compiled x-derivative using central differences.
    
    Uses central differences in interior, one-sided at boundaries.
    
    Args:
        f (jnp.ndarray): Function to differentiate. Expected shape: (Nx, Ny)
        h (float): Step size.
    
    Returns:
        jnp.ndarray: x-derivative. Shape: (Nx, Ny)
    """
    # Central differences for interior
    df = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * h)
    # One-sided at boundaries
    df = df.at[0, :].set((f[1, :] - f[0, :]) / h)
    df = df.at[-1, :].set((f[-1, :] - f[-2, :]) / h)
    return df


@jit
def jax_dy(f, h=1e-5):
    """JAX-compiled y-derivative using central differences.
    
    Uses central differences in interior, one-sided at boundaries.
    
    Args:
        f (jnp.ndarray): Function to differentiate. Expected shape: (Nx, Ny)
        h (float): Step size.
    
    Returns:
        jnp.ndarray: y-derivative. Shape: (Nx, Ny)
    """
    # Central differences for interior
    df = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * h)
    # One-sided at boundaries
    df = df.at[:, 0].set((f[:, 1] - f[:, 0]) / h)
    df = df.at[:, -1].set((f[:, -1] - f[:, -2]) / h)
    return df


def laplacian(f, dx=1e-5, dy=1e-5):
    """Calculate the Laplacian of f at all points in a 2D field using finite differences.
    
    Args:
        f (np.ndarray): Function to differentiate.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Laplacian of f.
    """
    # Interior points using finite difference
    f_padded = np.pad(f.copy(), ((1, 1), (1, 1)), mode="edge")
    d2x = (f_padded[1:-1, 2:] - 2*f_padded[1:-1, 1:-1] + f_padded[1:-1, :-2]) / dx**2
    d2y = (f_padded[2:, 1:-1] - 2*f_padded[1:-1, 1:-1] + f_padded[:-2, 1:-1]) / dy**2

    return d2x + d2y


@jit
def jax_laplacian(f, dx, dy, f_1_grid, f_2_grid):
    """Laplacian in terrain-following coords: φ_xx - 2 f' φ_xη - f'' φ_η + (1+f'²) φ_ηη. Flat → Cartesian (f_1,f_2 zeros)."""
    f_1 = f_1_grid
    f_2 = f_2_grid
    phi_xx = jax_dx(jax_dx(f, h=dx), h=dx)
    phi_x_eta = jax_dx(jax_dy(f, h=dy), h=dx)
    phi_eta = jax_dy(f, h=dy)
    phi_eta_eta = jax_dy(jax_dy(f, h=dy), h=dy)
    return phi_xx - 2.0 * f_1 * phi_x_eta - f_2 * phi_eta + (1.0 + f_1**2) * phi_eta_eta


def divergence(f, dx, dy):
    """Calculate the divergence of a 2D vector field f at all points.
    
    Args:
        f (np.ndarray): 2D vector field with shape (M, N, 2).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Divergence field.
    """
    div = np.zeros(f.shape[:2])  # Assuming f is a 2D vector field with shape (M, N, 2)
    div[1:-1, 1:-1] = (np.roll(f[..., 0], -1, axis=1)[1:-1, 1:-1] - np.roll(f[..., 0], 1, axis=1)[1:-1, 1:-1]) / (2 * dy) + \
                      (np.roll(f[..., 1], -1, axis=0)[1:-1, 1:-1] - np.roll(f[..., 1], 1, axis=0)[1:-1, 1:-1]) / (2 * dx)
    
    # Handle boundaries (Neumann condition: zero-gradient)
    div[:, 0] = div[:, 1]  # Bottom boundary
    div[:, -1] = div[:, -2]  # Top boundary
    div[0, :] = div[1, :]  # Left boundary
    div[-1, :] = div[-2, :]  # Right boundary
    
    return div


@jit
def jax_divergence(f, dx, dy, f_1_grid):
    """Divergence in terrain coords: u_x - f' u_η + v_η. Grid is fluid-only (bottom-aligned)."""
    u, v = f[..., 0], f[..., 1]
    f_1 = f_1_grid
    du_dx = jax_dx(u, h=dx)
    dv_dy = jax_dy(v, h=dy)
    u_eta = jax_dy(u, h=dy)
    return du_dx - f_1 * u_eta + dv_dy


def gradient(f, dx, dy):
    """Calculate the gradient of a scalar field f at all points in a 2D field.
    
    Args:
        f (np.ndarray): Scalar field.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        np.ndarray: Gradient field with shape (*f.shape, 2).
    """
    grad = np.zeros((*f.shape, 2))
    grad[:, :, 0] = numerical_derivative(f, axis=0, h=dx)  # Gradient in x-direction
    grad[:, :, 1] = numerical_derivative(f, axis=1, h=dy)  # Gradient in y-direction
    return grad


@jit
def jax_gradient(f, dx, dy, f_1_grid):
    """Gradient in terrain coords: (φ_x - f' φ_η, φ_η). f_1_grid (Nx,Ny). Flat → Cartesian (f_1 zeros)."""
    phi_x = jax_dx(f, h=dx)
    phi_eta = jax_dy(f, h=dy)
    # Broadcast f_1 for vector fields: (Nx,Ny) * (Nx,Ny,2) -> (Nx,Ny,2)
    f_1 = jnp.expand_dims(f_1_grid, axis=-1) if phi_eta.ndim > f_1_grid.ndim else f_1_grid
    grad_x = phi_x - f_1 * phi_eta
    grad_y = phi_eta
    return jnp.stack([grad_x, grad_y], axis=-1)


def norm(f):
    """Calculate the norm of a 2D vector field f at all points.
    
    Args:
        f (np.ndarray): 2D vector field with shape (M, N, 2).
    
    Returns:
        np.ndarray: Norm field.
    """
    return np.sqrt(f[:, :, 0]**2 + f[:, :, 1]**2)


@jit
def jax_norm(f):
    """JAX-compiled norm calculation.
    
    Args:
        f (jnp.ndarray): 2D vector field with shape (M, N, 2).
    
    Returns:
        jnp.ndarray: Norm field.
    """
    return jnp.sqrt(f[..., 0]**2 + f[..., 1]**2)
