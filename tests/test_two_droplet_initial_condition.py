import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulation.initial_conditions import initialize_phase_two_droplets_touching


def test_two_droplets_touch_but_do_not_overlap():
    config = {
        "initial_conditions": {
            "droplet_center_x": 0.5,
            "droplet_center_y": 0.0,
            "droplet_radius": 0.1,
            "inter_droplet_gap": 0.0,
        }
    }

    phi = initialize_phase_two_droplets_touching(401, 201, radius=0.1, epsilon=0.002, config=config)
    bottom_row = phi[:, 1]
    x = np.linspace(0.0, 1.0, phi.shape[0])

    liquid_x = x[bottom_row < 0.0]
    left_component = liquid_x[liquid_x < 0.5]
    right_component = liquid_x[liquid_x > 0.5]

    assert left_component.size > 0
    assert right_component.size > 0
    assert bottom_row[phi.shape[0] // 2] > 0.0
    assert left_component.max() < right_component.min()
    assert np.isclose(left_component.max(), 0.5, atol=5e-3)
    assert np.isclose(right_component.min(), 0.5, atol=5e-3)


def test_two_droplets_reject_overlap():
    config = {
        "initial_conditions": {
            "left_droplet_center_x": 0.45,
            "right_droplet_center_x": 0.54,
            "left_droplet_radius": 0.05,
            "right_droplet_radius": 0.05,
        }
    }

    try:
        initialize_phase_two_droplets_touching(64, 64, radius=0.05, config=config)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Expected overlapping droplets to raise ValueError")
