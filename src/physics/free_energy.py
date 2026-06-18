"""
Homogeneous free energy of the Cahn-Hilliard model.

Shared between the phase-field solver and the surface tension force so that
both always use the same thermodynamic potential.

Potentials (codes used inside jitted kernels):
    0 -- polynomial double well        f(phi) = (1 - phi^2)^2 / 4
    1 -- Flory-Huggins (logarithmic)   f(phi) = (theta/2)[(1+phi)ln(1+phi)
                                       + (1-phi)ln(1-phi)] - (theta_c/2) phi^2
"""

import numpy as np
import jax.numpy as jnp
from jax import jit

POTENTIAL_CODES = {
    "polynomial": 0,
    "flory_huggins": 1,
    "log": 1,
    "logarithmic": 1,
}


def potential_code_from_name(name):
    """Map a config potential name to the integer code used in jitted kernels."""
    key = str(name).lower()
    if key not in POTENTIAL_CODES:
        raise ValueError(
            f"Unknown phase potential '{name}'. "
            f"Supported: {sorted(POTENTIAL_CODES)}"
        )
    return POTENTIAL_CODES[key]


@jit
def jax_free_energy_derivative(
    phi,
    potential_code=0,
    log_theta=0.25,
    log_theta_c=1.0,
    log_delta=1e-6,
):
    """Configurable homogeneous free-energy derivative.

    Polynomial: f'(phi) = phi (phi^2 - 1), evaluated at phi as-is.  The
    polynomial potential does not bound phi; boundedness must come from the
    time-stepping scheme (e.g. the logarithmic convex split).

    Logarithmic (Flory-Huggins): f'(phi) = (theta/2) log((1+phi)/(1-phi))
    - theta_c phi.  ``log_delta`` is the standard interior regularization of
    the singular entropy term (Copetti–Elliott): the logarithm is evaluated no
    closer than ``delta`` to the pure phases so that stray evaluations at
    phi = ±1 (where mobility is degenerate and the flux M*grad(mu) is finite)
    do not produce infinities.  It is a solver regularization, not a model
    change: converged convex-split iterates stay in the interior.
    """
    polynomial = phi * (phi**2 - 1.0)
    delta = jnp.maximum(jnp.asarray(log_delta, dtype=phi.dtype), 1e-12)
    phi_log = jnp.clip(phi, -1.0 + delta, 1.0 - delta)
    logarithmic = (
        0.5
        * jnp.asarray(log_theta, dtype=phi.dtype)
        * jnp.log((1.0 + phi_log) / (1.0 - phi_log))
        - jnp.asarray(log_theta_c, dtype=phi.dtype) * phi_log
    )
    return jnp.where(jnp.asarray(potential_code) == 1, logarithmic, polynomial)


def binodal_phi(potential_code, log_theta=0.25, log_theta_c=1.0):
    """Positive binodal (bulk equilibrium) composition of the potential.

    Polynomial well: exactly 1.  Flory-Huggins: the nonzero root of
    f'(phi) = 0, i.e. the fixed point phi = tanh((theta_c/theta) phi).
    """
    if potential_code == 0:
        return 1.0
    ratio = float(log_theta_c) / float(log_theta)
    if ratio <= 1.0:
        raise ValueError(
            f"Flory-Huggins potential with theta_c/theta = {ratio} <= 1 has no "
            "phase separation (single well); binodal undefined."
        )
    phi = 0.9
    for _ in range(200):
        phi_new = np.tanh(ratio * phi)
        if abs(phi_new - phi) < 1e-15:
            break
        phi = phi_new
    return float(phi)


def _bulk_free_energy(phi, potential_code, log_theta, log_theta_c):
    """f(phi) on a numpy grid (used only for the one-off sigma quadrature)."""
    phi = np.asarray(phi, dtype=np.float64)
    if potential_code == 0:
        return 0.25 * (1.0 - phi**2) ** 2
    one_p = 1.0 + phi
    one_m = 1.0 - phi
    entropy = one_p * np.log(one_p) + one_m * np.log(one_m)
    return 0.5 * float(log_theta) * entropy - 0.5 * float(log_theta_c) * phi**2


def diffuse_interface_sigma(epsilon, potential_code, log_theta=0.25, log_theta_c=1.0):
    """Surface tension of the equilibrium planar interface of the CH model.

    sigma = epsilon * integral_{-phi_b}^{phi_b} sqrt(2 (f(phi) - f(phi_b))) dphi

    For the polynomial well this is the classic (2 sqrt(2)/3) epsilon.
    """
    phi_b = binodal_phi(potential_code, log_theta, log_theta_c)
    # Endpoints excluded: integrand -> 0 there, and log terms are finite on
    # the open interval.  20001 interior points give ~1e-8 relative accuracy.
    phi = np.linspace(-phi_b, phi_b, 20001)[1:-1]
    delta_f = _bulk_free_energy(phi, potential_code, log_theta, log_theta_c) - _bulk_free_energy(
        phi_b, potential_code, log_theta, log_theta_c
    )
    if delta_f.min() < -1e-12:
        raise RuntimeError(
            f"Negative excess free energy ({delta_f.min():.3e}) inside the "
            "miscibility gap; binodal solve is inconsistent with f(phi)."
        )
    integrand = np.sqrt(2.0 * np.maximum(delta_f, 0.0))
    sigma = float(epsilon) * float(np.trapezoid(integrand, phi))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"Diffuse-interface sigma quadrature failed: {sigma}")
    return sigma
