"""
Material properties calculations for two-phase flow.

This module contains functions to calculate material properties
based on the phase field, including density, Reynolds number,
and Weber number calculations.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


def calculate_density(phi, rho1, rho2):
    """Calculate the density based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        rho1 (float): Density for phase 1.
        rho2 (float): Density for phase 2.
    
    Returns:
        np.ndarray: Calculated density (shape: (Nx, Ny)).
    """
    # Calculate the density using the provided formula
    phi_mapped = (phi + 1) / 2.0
    rho = 1 / ((1 + phi_mapped) / (2 * rho2) + (1 - phi_mapped) / (2 * rho1))
    return rho


@jit
def jax_calculate_density(phi, rho1, rho2):
    """JAX-compiled version of density calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        rho1 (float): Density for phase 1.
        rho2 (float): Density for phase 2.
    
    Returns:
        jnp.ndarray: Calculated density (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    rho = 1 / ((1 + phi_mapped) / (2 * rho2) + (1 - phi_mapped) / (2 * rho1))
    return rho


def calculate_reynolds_number(phi, Re1, Re2):
    """Calculate the Reynolds number based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        Re1 (float): Reynolds number for phase 1.
        Re2 (float): Reynolds number for phase 2.
    
    Returns:
        np.ndarray: Calculated Reynolds number (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    Re = 1 / ((1 + phi_mapped) / (2 * Re2) + (1 - phi_mapped) / (2 * Re1))
    return Re


@jit
def jax_calculate_reynolds_number(phi, Re1, Re2):
    """JAX-compiled version of Reynolds number calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        Re1 (float): Reynolds number for phase 1.
        Re2 (float): Reynolds number for phase 2.
    
    Returns:
        jnp.ndarray: Calculated Reynolds number (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    Re = 1 / ((1 + phi_mapped) / (2 * Re2) + (1 - phi_mapped) / (2 * Re1))
    return Re


def calculate_weber_number(phi, We1, We2):
    """Calculate the Weber number based on the phase field.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
        We1 (float): Weber number for phase 1.
        We2 (float): Weber number for phase 2.
    
    Returns:
        np.ndarray: Calculated Weber number (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    We = 1 / ((1 + phi_mapped) / (2 * We2) + (1 - phi_mapped) / (2 * We1))
    return We


@jit
def jax_calculate_weber_number(phi, We1, We2):
    """JAX-compiled version of Weber number calculation.
    
    Args:
        phi (jnp.ndarray): Phase field (shape: (Nx, Ny)).
        We1 (float): Weber number for phase 1.
        We2 (float): Weber number for phase 2.
    
    Returns:
        jnp.ndarray: Calculated Weber number (shape: (Nx, Ny)).
    """
    phi_mapped = (phi + 1) / 2.0
    We = 1 / ((1 + phi_mapped) / (2 * We2) + (1 - phi_mapped) / (2 * We1))
    return We


def f_1(phi):
    """Double-well potential function with minimas at phi = 0 and phi = 1.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Value of the double-well potential (shape: (Nx, Ny)).
    """
    return phi**2 * (1 - phi)**2


def f_2(phi):
    """Double-well potential function with minimas at phi = -1 and phi = 1.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Value of the double-well potential (shape: (Nx, Ny)).
    """
    return 1./4 * (phi**2 - 1)**2


def df_1(phi):
    """Derivative of the double-well potential function.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Derivative of the double-well potential (shape: (Nx, Ny)).
    """
    return 2 * phi * (1 - phi) * (1 - 2 * phi)


def df_2(phi):
    """Derivative of the double-well potential function.
    
    Args:
        phi (np.ndarray): Phase field (shape: (Nx, Ny)).
    
    Returns:
        np.ndarray: Derivative of the double-well potential (shape: (Nx, Ny)).
    """
    return phi * (phi**2 - 1)


@jit
def jax_f_1(phi):
    """JAX-compiled version of f_1."""
    return phi**2 * (1 - phi)**2


@jit
def jax_f_2(phi):
    """JAX-compiled version of f_2."""
    return 1./4 * (phi**2 - 1)**2


@jit
def jax_df_1(phi):
    """JAX-compiled version of df_1."""
    return 2 * phi * (1 - phi) * (1 - 2 * phi)


@jit
def jax_df_2(phi):
    """JAX-compiled version of df_2."""
    return phi * (phi**2 - 1)
