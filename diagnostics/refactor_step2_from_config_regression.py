#!/usr/bin/env python3
"""
Step-2 regression check for SimulationState.from_config decomposition.

Runs deterministic initialization summaries for selected configs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.state import SimulationState


def summarize(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    s = SimulationState.from_config(cfg, restart_from=None)
    s.ensure_face_velocities()
    s.sync_collocated_from_faces()
    phi = np.array(s.phi)
    P = np.array(s.P)
    U = np.array(s.U)
    return {
        "config": str(config_path),
        "phi_shape": list(phi.shape),
        "U_shape": list(U.shape),
        "P_shape": list(P.shape),
        "u_face_shape": list(s.u_face.shape),
        "v_face_shape": list(s.v_face.shape),
        "phi_mean": float(np.mean(phi)),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "P_min": float(np.min(P)),
        "P_max": float(np.max(P)),
        "P_mean": float(np.mean(P)),
        "U_abs_mean": float(np.mean(np.abs(U))),
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

    out = {"step": "refactor_step2_from_config_regression", "results": []}
    for c in args.configs:
        out["results"].append(summarize(Path(c)))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

