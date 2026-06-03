#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compare_slide_mass_loss import run_variant


def main():
    parser = argparse.ArgumentParser(description="Replay a checkpoint with baseline vs ghost-cell contact-angle BC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--restart-from", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    variants = []

    baseline = copy.deepcopy(base_cfg)
    variants.append(("baseline", baseline))

    ghost = copy.deepcopy(base_cfg)
    ghost.setdefault("solver_params", {})
    ghost["solver_params"]["phase_field_solver"] = "ghost_cell"
    ghost["boundary_conditions"]["phase_field"]["contact_angle_method"] = "ghost_cell"
    ghost["boundary_conditions"]["phase_field"].pop("contact_angle_relaxation", None)
    variants.append(("ghost_cell", ghost))

    for name, cfg in variants:
        print(f"\n=== Running {name} ===")
        result = run_variant(
            name,
            cfg,
            output_dir,
            args.steps,
            args.log_every,
            patch_relaxed_simple=False,
            restart_from=args.restart_from,
        )
        print(
            f"{name}: mass {result['mass0']:.8f} -> {result['massN']:.8f} "
            f"({result['mass_delta_pct']:+.3f}%)"
        )


if __name__ == "__main__":
    main()
