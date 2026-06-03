#!/usr/bin/env python3
"""
Step-1 refactor smoke test: SimulationState.from_config initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.state import SimulationState


def _run_one(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    state = SimulationState.from_config(cfg, restart_from=None)
    state.ensure_face_velocities()
    state.sync_collocated_from_faces()
    return {
        "config": str(config_path),
        "Nx": int(state.Nx),
        "Ny": int(state.Ny),
        "phi_shape": tuple(state.phi.shape),
        "U_shape": tuple(state.U.shape),
        "P_shape": tuple(state.P.shape),
        "u_face_shape": tuple(state.u_face.shape),
        "v_face_shape": tuple(state.v_face.shape),
        "P_min": float(np.min(np.array(state.P))),
        "P_max": float(np.max(np.array(state.P))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/static_droplet_contact_angle/contact_angle_60.json",
            "configs/static_droplet_contact_angle/contact_angle_120.json",
        ],
    )
    args = ap.parse_args()

    results = []
    for cfg in args.configs:
        results.append(_run_one(Path(cfg)))

    print(json.dumps({"step": "refactor_step1_state_init", "results": results}, indent=2))


if __name__ == "__main__":
    main()

