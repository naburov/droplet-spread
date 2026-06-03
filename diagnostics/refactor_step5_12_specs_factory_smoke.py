#!/usr/bin/env python3
"""
Smoke test for refactor steps 1+2:
- specs/bundles attached to state
- from_config routed through state_factory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.state import SimulationState


def run_one(config_path: str) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    s = SimulationState.from_config(cfg, restart_from=None)
    s.ensure_face_velocities()
    s.sync_collocated_from_faces()

    return {
        "config": config_path,
        "phi_shape": list(s.phi.shape),
        "U_shape": list(s.U.shape),
        "P_shape": list(s.P.shape),
        "u_face_shape": list(s.u_face.shape),
        "v_face_shape": list(s.v_face.shape),
        "context_present": s.context is not None,
        "solver_bundle_present": s.solver_bundle is not None,
        "bc_bundle_present": s.bc_bundle is not None,
        "context_grid_Nx": int(s.context.grid.Nx) if s.context is not None else None,
        "context_grid_Ny": int(s.context.grid.Ny) if s.context is not None else None,
        "phi_mean": float(np.mean(np.array(s.phi))),
        "P_mean": float(np.mean(np.array(s.P))),
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

    out = {"step": "refactor_steps_1_2", "results": [run_one(c) for c in args.configs]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

