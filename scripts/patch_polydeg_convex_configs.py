#!/usr/bin/env python3
"""Patch configs to the bounded log-degenerate production scheme.

Applies the wall-energy ghost contact law and the Flory-Huggins (logarithmic)
potential with the log_entropy convex split and degenerate mobility to the
given config files or directories. The logarithmic entropy keeps phi strictly
inside (-1, 1) without any clip-based bound enforcement; the removed
phase_potential_clip / degenerate_mobility_clip keys are dropped.
"""

import argparse
import json
import sys
from pathlib import Path

PHASE_FIELD_BC_PATCH = {
    "contact_angle_ghost_law": "wall_energy",
    "contact_angle_full_wall": True,
    "contact_angle_wall_energy_scale": 1.0,
    "contact_angle_wall_tangent_regularization": 0.0,
}

SOLVER_PARAMS_PATCH = {
    "semi_implicit_contact_split": "implicit_wall_energy",
    "use_degenerate_mobility": True,
    "degenerate_mobility_blend": 0.0,
    "degenerate_mobility_power": 2.0,
    "degenerate_mobility_imex": True,
    "degenerate_mobility_imex_mref": 0.0,
    "phase_potential": "flory_huggins",
    "phase_log_theta": 0.25,
    "phase_log_theta_c": 1.0,
    "phase_log_delta": 1e-06,
    "phase_convex_stabilization": 0.0,
    "phase_convex_split": "log_entropy",
    "phase_convex_split_maxiter": 8,
    "phase_convex_split_tol": 1e-07,
    "phase_convex_split_damping": 0.8,
}

REMOVED_SOLVER_KEYS = (
    "phase_potential_clip",
    "degenerate_mobility_clip",
)

# Energy-consistent capillary force F = lambda * mu * grad(phi).  Replaces the
# CSF form whose normalized-gradient curvature saturates at 1/dy on the wall
# wetting film and produces spurious O(1) forces in pure gas (impact runs).
# The potential form needs no curvature smoothing and no wall-row overwrite.
SURFACE_TENSION_PATCH = {
    "force_form": "potential",
}

REMOVED_SURFACE_TENSION_KEYS = (
    "smooth_curvature",
    "smoothing_radius",
    "apply_boundary_overwrite",
)


def patch_config(path: Path) -> bool:
    config = json.loads(path.read_text(encoding="utf-8"))

    phase_bc = config["boundary_conditions"]["phase_field"]
    solver = config["solver_params"]
    surface_tension = config["physical_params"].setdefault("surface_tension", {})

    before = (
        json.dumps(phase_bc, sort_keys=True),
        json.dumps(solver, sort_keys=True),
        json.dumps(surface_tension, sort_keys=True),
    )
    phase_bc.update(PHASE_FIELD_BC_PATCH)
    solver.update(SOLVER_PARAMS_PATCH)
    for key in REMOVED_SOLVER_KEYS:
        solver.pop(key, None)
    surface_tension.update(SURFACE_TENSION_PATCH)
    for key in REMOVED_SURFACE_TENSION_KEYS:
        surface_tension.pop(key, None)
    after = (
        json.dumps(phase_bc, sort_keys=True),
        json.dumps(solver, sort_keys=True),
        json.dumps(surface_tension, sort_keys=True),
    )

    if before == after:
        return False
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Config files or directories")
    parser.add_argument(
        "--glob", default="*.json", help="Glob for directories (default: *.json)"
    )
    args = parser.parse_args()

    files = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob(args.glob)))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"No such config file or directory: {p}")

    changed = 0
    for f in files:
        if patch_config(f):
            print(f"PATCHED  {f}")
            changed += 1
        else:
            print(f"OK       {f}")
    print(f"Patched {changed}/{len(files)} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
