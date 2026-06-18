"""Tests for the conservative wavelength-2h checkerboard filter."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from physics.phase_field import jax_checkerboard_filter


def test_preserves_global_sum():
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.uniform(-1.0, 1.0, size=(32, 24)))
    out = jax_checkerboard_filter(phi, 0.05)
    np.testing.assert_allclose(float(jnp.sum(out)), float(jnp.sum(phi)), rtol=0, atol=1e-12)


def test_damps_pure_checkerboard_by_expected_factor():
    n = 32
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cb = jnp.asarray((-1.0) ** (i + j), dtype=jnp.float64)
    s = 0.05
    out = jax_checkerboard_filter(cb, s)
    # one (1 - s) factor per axis on the doubly-alternating mode, interior
    interior = np.asarray(out)[2:-2, 2:-2]
    expected = np.asarray(cb)[2:-2, 2:-2] * (1.0 - s) ** 2
    np.testing.assert_allclose(interior, expected, rtol=0, atol=1e-12)


def test_leaves_constant_field_unchanged():
    phi = jnp.full((16, 16), 0.7, dtype=jnp.float64)
    out = jax_checkerboard_filter(phi, 0.05)
    np.testing.assert_allclose(np.asarray(out), 0.7, rtol=0, atol=1e-14)


def test_smooth_tanh_profile_nearly_untouched():
    n = 128
    y = (np.arange(n) + 0.5) / n
    profile = np.tanh((y - 0.5) / (np.sqrt(2.0) * 0.04))
    phi = jnp.asarray(np.tile(profile, (n, 1)))
    out = jax_checkerboard_filter(phi, 0.05)
    # interface resolved with ~5 cells: 4th-difference damping stays tiny
    assert float(jnp.max(jnp.abs(out - phi))) < 5e-3


def test_zero_strength_is_identity():
    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.uniform(-1.0, 1.0, size=(8, 8)))
    out = jax_checkerboard_filter(phi, 0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(phi), rtol=0, atol=0)


def test_solver_rejects_out_of_range_strength():
    from physics.phase_field import PhaseFieldSolverGhostCell

    config = {
        "solver_params": {
            "phase_field_solver": "ghost_cell",
            "phase_checkerboard_filter": 1.5,
        },
        "boundary_conditions": {"phase_field": {}},
    }
    with pytest.raises(ValueError, match="phase_checkerboard_filter"):
        PhaseFieldSolverGhostCell(Pe=20.0, epsilon=0.04, contact_angle=90, config=config)
