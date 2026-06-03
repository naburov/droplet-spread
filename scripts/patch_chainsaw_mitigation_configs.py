#!/usr/bin/env python3
"""Patch generated longrun JSON configs with chainsaw-mitigation phase-field settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
}


def patch_file(path: Path, dry_run: bool, aggressive: bool) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    pf = data.setdefault("boundary_conditions", {}).setdefault("phase_field", {})
    mitigation = AGGRESSIVE_MITIGATION if aggressive else CONSERVATIVE_MITIGATION
    changed = False
    for key, value in mitigation.items():
        if pf.get(key) != value:
            pf[key] = value
            changed = True
    st = data.setdefault("physical_params", {}).setdefault("surface_tension", {})
    for key, value in SURFACE_TENSION.items():
        if st.get(key) != value:
            st[key] = value
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
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Also switch contact_angle_ghost_law to wall_energy",
    )
    args = parser.parse_args()
    n = 0
    for root in args.roots:
        for path in sorted(root.glob("*.json")):
            if patch_file(path, args.dry_run, args.aggressive):
                print(("would patch" if args.dry_run else "patched"), path)
                n += 1
    print(f"{'would update' if args.dry_run else 'updated'} {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
