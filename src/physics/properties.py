"""
Material properties calculations for two-phase flow.

Property interpolations evaluate phi as-is: keeping phi within [-1, 1] is the
phase solver's responsibility (e.g. flory_huggins potential with the
log_entropy convex split). Out-of-range phi yields out-of-range properties on
purpose, so bound violations surface loudly instead of being masked.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


def calculate_density(phi, rho1, rho2):
    """Calculate density using linear interpolation.
    
    Args:
        phi: Phase field. phi=-1: liquid (rho2), phi=+1: air (rho1)
        rho1: Density of air.
        rho2: Density of liquid.
    
    Returns:
        Density field.
    """
    w = (phi + 1) / 2.0
    # Linear interpolation: rho = (1 - w) * rho2 + w * rho1, w = air fraction
    return (1 - w) * rho2 + w * rho1


@jit
def jax_calculate_density(phi, rho1, rho2):
    """JAX version of density calculation using linear interpolation."""
    w = (phi + 1) / 2.0
    return (1 - w) * rho2 + w * rho1


def calculate_reynolds_number(phi, Re1, Re2):
    """Calculate Reynolds number based on phase field.
    
    Args:
        phi: Phase field. phi=-1: liquid (Re2), phi=+1: air (Re1)
        Re1: Reynolds number for air (phase 1).
        Re2: Reynolds number for liquid (phase 2).
    """
    w = (phi + 1) / 2.0
    # Harmonic mean interpolation: Re = 1 / ((1-w)/Re2 + w/Re1)
    return 1 / ((1 - w) / Re2 + w / Re1)


@jit
def jax_calculate_reynolds_number(phi, Re1, Re2):
    """JAX version of Reynolds number calculation.
    
    Args:
        phi: Phase field. phi=-1: liquid (Re2), phi=+1: air (Re1)
        Re1: Reynolds number for air (phase 1).
        Re2: Reynolds number for liquid (phase 2).
    """
    w = (phi + 1) / 2.0
    return 1 / ((1 - w) / Re2 + w / Re1)


def calculate_weber_number(phi, We1, We2):
    """Calculate Weber number based on phase field.
    
    Args:
        phi: Phase field. phi=-1: liquid (We2), phi=+1: air (We1)
        We1: Weber number for air (phase 1).
        We2: Weber number for liquid (phase 2).
    """
    w = (phi + 1) / 2.0
    # Harmonic mean interpolation: We = 1 / ((1-w)/We2 + w/We1)
    return 1 / ((1 - w) / We2 + w / We1)


@jit
def jax_calculate_weber_number(phi, We1, We2):
    """JAX version of Weber number calculation.
    
    Args:
        phi: Phase field. phi=-1: liquid (We2), phi=+1: air (We1)
        We1: Weber number for air (phase 1).
        We2: Weber number for liquid (phase 2).
    """
    w = (phi + 1) / 2.0
    return 1 / ((1 - w) / We2 + w / We1)


def df_2(phi):
    """Derivative of double-well potential f(phi) = (1/4)(phi^2 - 1)^2."""
    return phi * (phi**2 - 1)


@jit
def jax_df_2(phi):
    """JAX version of df_2."""
    return phi * (phi**2 - 1)


@jit
def jax_advection_function(psi, threshold=0.1):
    """Advection function A(ψ) for ice-water phase field.
    
    A(ψ) = 0.5 * (1 - tanh(ψ / threshold))
    - A ≈ 1 in liquid (ψ < 0): full advection
    - A ≈ 0 in ice (ψ > 0): no advection (solid)
    
    Args:
        psi: Ice phase field (-1=water, +1=ice).
        threshold: Transition width (default: 0.1).
    
    Returns:
        Advection coefficient field (0 to 1).
    """
    return 0.5 * (1.0 - jnp.tanh(psi / threshold))
