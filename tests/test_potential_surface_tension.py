"""Tests for the energy-consistent potential-form surface tension force."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp

from physics.free_energy import (
    binodal_phi,
    diffuse_interface_sigma,
    potential_code_from_name,
)
from physics.surface_tension import SurfaceTensionSolver
from simulation.geometry import Geometry


LOG_PARAMS = {
    "phase_potential": "flory_huggins",
    "phase_log_theta": 0.25,
    "phase_log_theta_c": 1.0,
    "phase_log_delta": 1e-6,
}
POLY_PARAMS = {"phase_potential": "polynomial"}


def make_solver(epsilon, We2, potential_params, contact_angle=120):
    return SurfaceTensionSolver(
        epsilon=epsilon,
        We1=1.0,
        We2=We2,
        contact_angle=contact_angle,
        force_form="potential",
        potential_params=potential_params,
    )


def test_sigma_polynomial_matches_analytic():
    eps = 0.04
    sigma = diffuse_interface_sigma(eps, potential_code_from_name("polynomial"))
    assert abs(sigma - (2.0 * np.sqrt(2.0) / 3.0) * eps) < 1e-6 * eps


def test_binodal_flory_huggins():
    phi_b = binodal_phi(1, log_theta=0.25, log_theta_c=1.0)
    # Fixed point of tanh(4 phi); known value ~0.99933
    assert abs(phi_b - np.tanh(4.0 * phi_b)) < 1e-12
    assert 0.999 < phi_b < 0.9995


def test_force_zero_in_bulk_phases():
    nx, ny = 64, 64
    eps = 0.04
    geom = Geometry.flat(nx, ny)
    for params in (POLY_PARAMS, LOG_PARAMS):
        solver = make_solver(eps, We2=1.0, potential_params=params)
        code = potential_code_from_name(params["phase_potential"])
        phi_b = binodal_phi(code, params.get("phase_log_theta", 0.25), params.get("phase_log_theta_c", 1.0))
        for sign in (1.0, -1.0):
            phi = jnp.full((nx, ny), sign * phi_b)
            force = solver.calculate_force(phi, 0.02, 0.02, geom)
            force = solver.apply_boundary_conditions(force, phi, geometry=geom)
            assert float(jnp.max(jnp.abs(force))) == pytest.approx(0.0, abs=1e-12)


def test_laplace_pressure_jump_polynomial_droplet():
    """Static circular droplet: integral of radial force = sigma_eff / R."""
    nx = ny = 256
    L = 2.0
    dx = dy = L / nx
    eps = 0.04
    We2 = 1.0
    R = 0.5
    x = (np.arange(nx) + 0.5) * dx - L / 2
    y = (np.arange(ny) + 0.5) * dy - L / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    # Liquid (phi=-1) inside, gas (+1) outside; tanh is the exact polynomial
    # equilibrium profile, so mu is the pure curvature (Gibbs-Thomson) part.
    phi = jnp.asarray(np.tanh((r - R) / (np.sqrt(2.0) * eps)))

    solver = make_solver(eps, We2=We2, potential_params=POLY_PARAMS)
    force = solver.calculate_force(phi, dx, dy, Geometry.flat(nx, ny))
    force = np.asarray(solver.apply_boundary_conditions(force, phi, geometry=Geometry.flat(nx, ny)))

    # Integrate F . e_r along the +x axis through the interface
    i_axis = np.argmin(np.abs(y))
    fx = force[:, i_axis, 0]
    mask = x > 0
    delta_p = -np.trapezoid(fx[mask], x[mask])

    sigma_eff = 3.0 * np.sqrt(2.0) * eps / (4.0 * We2)
    expected = sigma_eff / R
    assert delta_p == pytest.approx(expected, rel=0.05)


def _curl_z_wall_band(force, dx, dy):
    """curl_z on rows 1..5 (the wall film band), x-wrap columns cropped."""
    fx, fy = force[..., 0], force[..., 1]
    dfy_dx = (np.roll(fy, -1, axis=0) - np.roll(fy, 1, axis=0)) / (2 * dx)
    dfx_dy = (np.roll(fx, -1, axis=1) - np.roll(fx, 1, axis=1)) / (2 * dy)
    return (dfy_dx - dfx_dy)[2:-2, 1:6]


def test_wall_film_force_is_irrotational():
    """An x-uniform wall wetting film must not drive flow: any capillary force
    on it has to be a pure gradient (absorbed by the pressure projection).

    With grid-level noise, CSF saturates kappa at ~1/dy with sign flips along
    the wall, producing an O(1) rotational force in pure gas (the spurious
    under-droplet vortices seen in impact runs).  The potential form responds
    only through mu, so its curl stays at the noise level.
    """
    rng = np.random.default_rng(7)
    nx, ny = 128, 160
    Lx, Ly = 1.0, 1.2
    dx, dy = Lx / nx, Ly / ny
    eps = 0.04
    geom = Geometry.flat(nx, ny)

    phi_b = binodal_phi(1, 0.25, 1.0)
    phi = np.full((nx, ny), phi_b)
    phi[:, 0] = 0.93  # wall-energy wetting layer value observed in production
    phi += 1e-4 * rng.standard_normal((nx, ny))  # grid-level noise
    phi = jnp.asarray(phi)

    csf = SurfaceTensionSolver(
        epsilon=eps, We1=1.0, We2=1.0, contact_angle=120, force_form="csf"
    )
    pot = make_solver(eps, We2=1.0, potential_params=LOG_PARAMS)

    f_csf = np.asarray(csf.calculate_force(phi, dx, dy, geom))
    f_pot = np.asarray(pot.calculate_force(phi, dx, dy, geom))

    curl_csf = np.abs(_curl_z_wall_band(f_csf, dx, dy)).max()
    curl_pot = np.abs(_curl_z_wall_band(f_pot, dx, dy)).max()

    # Measured: curl_csf ~ 4.8e2 (kappa saturated at 1/dy with sign flips),
    # curl_pot ~ 2.1e1 (noise amplified once through the mu Laplacian only).
    assert curl_csf > 10.0 * curl_pot


def test_potential_requires_params():
    with pytest.raises(ValueError):
        SurfaceTensionSolver(
            epsilon=0.04, We1=1.0, We2=1.0, contact_angle=90, force_form="potential"
        )
