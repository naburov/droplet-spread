#!/usr/bin/env python3
"""Patch production JSON configs: stable ghost delta split, soft contact mask, checkpoint_interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTACT_SPLIT_FIX = {
    "semi_implicit_contact_split": "explicit_delta",
}

CONSERVATIVE_MITIGATION = {
    "contact_angle_full_wall": False,
    "contact_mask_soft_band": 0.8,
    "contact_mask_grad_scale": 0.5,
}

AGGRESSIVE_MITIGATION = {
    **CONSERVATIVE_MITIGATION,
    "contact_angle_ghost_law": "wall_energy",
}

SURFACE_TENSION = {
    "smooth_curvature": True,
    "smoothing_radius": 1,
    "force_form": "csf",
}

DEFAULT_CHECKPOINT_INTERVAL = 100


def patch_file(
    path: Path,
    dry_run: bool,
    aggressive: bool,
    checkpoint_interval: int,
) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    pf = data.setdefault("boundary_conditions", {}).setdefault("phase_field", {})
    solver = data.setdefault("solver_params", {})
    time_params = data.setdefault("time_params", {})
    mitigation = AGGRESSIVE_MITIGATION if aggressive else CONSERVATIVE_MITIGATION
    changed = False
    for key, value in CONTACT_SPLIT_FIX.items():
        if solver.get(key) != value:
            solver[key] = value
            changed = True
    for key, value in mitigation.items():
        if pf.get(key) != value:
            pf[key] = value
            changed = True
    st = data.setdefault("physical_params", {}).setdefault("surface_tension", {})
    for key, value in SURFACE_TENSION.items():
        if st.get(key) != value:
            st[key] = value
            changed = True
    if time_params.get("checkpoint_interval") != checkpoint_interval:
        time_params["checkpoint_interval"] = checkpoint_interval
        changed = True
    if changed and not dry_run:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path("configs/generated_sliding_long_realpaper_longrun"),
            Path("configs/generated_sliding_terrain_realpaper_longrun"),
            Path("configs/generated_falling_impact"),
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    args = parser.parse_args()
    n = 0
    for root in args.roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            if patch_file(path, args.dry_run, args.aggressive, args.checkpoint_interval):
                print(("would patch" if args.dry_run else "patched"), path)
                n += 1
    print(f"{'would update' if args.dry_run else 'updated'} {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
