#!/usr/bin/env python3
"""
Step-3 smoke checks for PPE BC derivation refactor.
"""

from __future__ import annotations

import json
from pathlib import Path

from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config, derive_ppe_bcs_from_velocity_bcs


def main() -> None:
    cases = []

    # Case 1: standard static droplet configs
    for cfg_path in [
        "configs/static_droplet_contact_angle/contact_angle_60.json",
        "configs/static_droplet_contact_angle/contact_angle_120.json",
    ]:
        cfg = json.loads(Path(cfg_path).read_text())
        cases.append({
            "name": cfg_path,
            "ppe_bcs": derive_ppe_bcs_from_config(cfg),
        })

    # Case 2: explicit inlet/outlet synthetic setup
    vel = {"left": "dirichlet", "right": "do_nothing", "top": "neumann", "bottom": "no_slip"}
    cases.append({
        "name": "synthetic_inlet_outlet",
        "velocity_bcs": vel,
        "ppe_bcs": derive_ppe_bcs_from_velocity_bcs(vel),
    })

    # Case 3: pressure-open override should pin top
    vel2 = {"left": "neumann", "right": "neumann", "top": "neumann", "bottom": "neumann"}
    p2 = {"top": "open", "bottom": "neumann", "left": "neumann", "right": "neumann"}
    cases.append({
        "name": "synthetic_pressure_open_top",
        "velocity_bcs": vel2,
        "pressure_bcs": p2,
        "ppe_bcs": derive_ppe_bcs_from_velocity_bcs(vel2, p2),
    })

    print(json.dumps({"step": "refactor_step3_ppe_bc_derivation", "cases": cases}, indent=2))


if __name__ == "__main__":
    main()

