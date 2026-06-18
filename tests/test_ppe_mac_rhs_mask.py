import numpy as np

from boundary_conditions.velocity_bc.staggered import StaggeredVelocityBoundaryConditions
from solvers.ppe import _mask_ppe_rhs_mac_boundary_artifacts


def _velocity_bc(mask=False):
    return StaggeredVelocityBoundaryConditions(
        {
            "boundary_conditions": {
                "velocity": {
                    "top": "do_nothing",
                    "bottom": "no_slip",
                    "left": "slip_symmetry",
                    "right": "slip_symmetry",
                    "mask_ppe_boundary_artifacts": mask,
                }
            }
        }
    )


def test_mac_ppe_rhs_keeps_wall_cells_by_default():
    rhs = np.arange(20.0).reshape(4, 5)

    masked = _mask_ppe_rhs_mac_boundary_artifacts(rhs.copy(), _velocity_bc(mask=False))

    np.testing.assert_array_equal(masked, rhs)


def test_mac_ppe_rhs_legacy_mask_is_explicit_opt_in():
    rhs = np.ones((4, 5))

    masked = _mask_ppe_rhs_mac_boundary_artifacts(rhs.copy(), _velocity_bc(mask=True))

    assert np.all(masked[:, 0] == 0.0)
    assert np.all(masked[0, 1:] == 1.0)
    assert np.all(masked[-1, 1:] == 1.0)
