import os
import sys
import unittest

import numpy as np
import jax.numpy as jnp


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from boundary_conditions.advection_bc import AdvectionBoundaryConditions
from boundary_conditions.chemical_potential_bc import (
    BC_DIRICHLET,
    BC_NEUMANN,
    jax_apply_chemical_potential_bc,
)


class BoundaryConditionRegressionTests(unittest.TestCase):
    def test_top_impermeable_advection_copies_from_interior(self):
        config = {
            "boundary_conditions": {
                "advection": {
                    "top": "impermeable",
                    "bottom": "open",
                    "left": "open",
                    "right": "open",
                    "velocity_threshold": 1e-10,
                }
            }
        }
        bc = AdvectionBoundaryConditions(config)
        phi = jnp.arange(25.0).reshape(5, 5)
        U = jnp.zeros((5, 5, 2))
        U = U.at[:, -1, 1].set(0.25)

        updated = np.asarray(bc.apply_boundary_conditions(phi, U, 1e-3, 0.1, 0.1))
        np.testing.assert_allclose(updated[1:-1, -1], np.asarray(phi)[1:-1, -2])

    def test_configured_chemical_potential_bcs_are_applied(self):
        mu = jnp.arange(25.0).reshape(5, 5)
        updated = np.asarray(
            jax_apply_chemical_potential_bc(
                mu,
                0.1,
                0.1,
                top_bc=BC_DIRICHLET,
                bottom_bc=BC_NEUMANN,
                left_bc=BC_NEUMANN,
                right_bc=BC_NEUMANN,
                top_value=7.5,
            )
        )

        np.testing.assert_allclose(updated[:, -1], 7.5)
        np.testing.assert_allclose(updated[:, 0], updated[:, 1])
        np.testing.assert_allclose(updated[0, :], updated[1, :])
        np.testing.assert_allclose(updated[-1, :], updated[-2, :])


if __name__ == "__main__":
    unittest.main()
