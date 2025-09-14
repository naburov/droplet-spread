"""
Finite difference methods for numerical derivatives.

This module contains implementations of finite difference schemes
for calculating gradients, divergences, and Laplacians.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


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
    
    Args:
        f (jnp.ndarray): Function to differentiate.
        h (float): Step size.
    
    Returns:
        jnp.ndarray: x-derivative.
    """
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * h)


@jit
def jax_dy(f, h=1e-5):
    """JAX-compiled y-derivative using central differences.
    
    Args:
        f (jnp.ndarray): Function to differentiate.
        h (float): Step size.
    
    Returns:
        jnp.ndarray: y-derivative.
    """
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * h)


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
def jax_laplacian(f, dx=1e-5, dy=1e-5):
    """JAX-compiled Laplacian calculation.
    
    Args:
        f (jnp.ndarray): Function to differentiate.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Laplacian of f.
    """
    f_padded = jnp.pad(f.copy(), ((1, 1), (1, 1)), mode="edge")
    d2x = (f_padded[1:-1, 2:] - 2*f_padded[1:-1, 1:-1] + f_padded[1:-1, :-2]) / dx**2
    d2y = (f_padded[2:, 1:-1] - 2*f_padded[1:-1, 1:-1] + f_padded[:-2, 1:-1]) / dy**2

    return d2x + d2y


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
def jax_divergence(f, dx, dy):
    """JAX-compiled divergence calculation.
    
    Args:
        f (jnp.ndarray): 2D vector field with shape (M, N, 2).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Divergence field.
    """
    return jax_dx(f[..., 0], h=dy) + jax_dy(f[..., 1], h=dx)


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
def jax_gradient(f, dx, dy):
    """JAX-compiled gradient calculation.
    
    Args:
        f (jnp.ndarray): Scalar field.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
    
    Returns:
        jnp.ndarray: Gradient field with shape (*f.shape, 2).
    """
    grad_x = jax_dx(f, h=dy)  # Gradient in x-direction
    grad_y = jax_dy(f, h=dx)  # Gradient in y-direction
    grad = jnp.stack([grad_x, grad_y], axis=-1)  # Shape: (*f.shape, 2)
    return grad


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
