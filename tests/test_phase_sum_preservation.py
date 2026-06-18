"""Interface-weighted global phi-sum preservation.

The wall/BC corrections may change the global phi sum; the compensation must
not be dumped uniformly into the bulk phases, where degenerate mobility
(M ~ (1-phi^2)^p) cannot relax it.  A uniform shift makes the gas bulk creep
from the binodal to the log-potential clamp over thousands of steps, which
feeds the potential-form capillary force and chainsaws the interface.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from physics.phase_field import PhaseFieldSolverGhostCell


class _StubContactBC:
    conserve_phi_sum = True


class _StubBCManager:
    contact_angle_bc = _StubContactBC()


def _solver_with_stub_bc():
    solver = PhaseFieldSolverGhostCell.__new__(PhaseFieldSolverGhostCell)
    solver.bc_manager = _StubBCManager()
    return solver


def _droplet_phi(n=64, radius=0.25, eps=0.04, binodal=0.99933):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + Y**2)
    return jnp.asarray(binodal * np.tanh((r - radius) / (np.sqrt(2.0) * eps)))


def test_phase_sum_correction_preserves_mean():
    solver = _solver_with_stub_bc()
    phi_old = _droplet_phi()
    # Mimic a wall update that removed some phi mass near the bottom row.
    phi_new = phi_old.at[:, 0].add(-1e-3)
    phi_corr = solver._preserve_phase_sum_if_requested(phi_old, phi_new)
    np.testing.assert_allclose(
        float(jnp.mean(phi_corr)), float(jnp.mean(phi_old)), rtol=0, atol=1e-12
    )


def test_phase_sum_correction_leaves_bulk_at_binodal():
    solver = _solver_with_stub_bc()
    binodal = 0.99933
    phi_old = _droplet_phi(binodal=binodal)
    phi_new = phi_old.at[:, 0].add(-1e-3)
    phi_corr = solver._preserve_phase_sum_if_requested(phi_old, phi_new)

    # Bulk gas cells (phi ~ +binodal, far from the droplet) must not absorb
    # the correction: their weight (1 - phi^2) is ~1e-3, so the shift they
    # receive must be orders of magnitude below the uniform-shift baseline.
    bulk = np.abs(np.asarray(phi_new)) > 0.999
    shift_bulk = np.abs(np.asarray(phi_corr - phi_new))[bulk].max()
    uniform_shift = abs(float(jnp.mean(phi_old) - jnp.mean(phi_new)))
    assert shift_bulk < 0.05 * uniform_shift

    # The interface band absorbs it instead.
    band = np.abs(np.asarray(phi_new)) < 0.9
    shift_band = np.abs(np.asarray(phi_corr - phi_new))[band].max()
    assert shift_band > uniform_shift


def test_phase_sum_correction_uniform_fallback_without_interface():
    solver = _solver_with_stub_bc()
    phi_old = jnp.full((32, 32), 1.0)
    phi_new = jnp.full((32, 32), 1.0 - 1e-4)
    phi_corr = solver._preserve_phase_sum_if_requested(phi_old, phi_new)
    np.testing.assert_allclose(np.asarray(phi_corr), 1.0, rtol=0, atol=1e-12)
