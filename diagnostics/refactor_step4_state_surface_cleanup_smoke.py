#!/usr/bin/env python3
"""
Step-4 smoke check after removing dead state init config surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

from simulation.state import SimulationState


def main() -> None:
    cfg_path = Path("configs/static_droplet_contact_angle/contact_angle_120.json")
    cfg = json.loads(cfg_path.read_text())
    state = SimulationState.from_config(cfg, restart_from=None)
    state.ensure_face_velocities()
    state.sync_collocated_from_faces()

    out = {
        "step": "refactor_step4_state_surface_cleanup",
        "config": str(cfg_path),
        "ok": True,
        "shapes": {
            "phi": list(state.phi.shape),
            "U": list(state.U.shape),
            "P": list(state.P.shape),
            "u_face": list(state.u_face.shape),
            "v_face": list(state.v_face.shape),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

