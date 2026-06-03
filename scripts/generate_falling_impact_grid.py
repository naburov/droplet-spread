#!/usr/bin/env python3
"""Generate a compact falling-droplet impact experiment grid."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "config_falling_droplet.json"
OUT_DIR = ROOT / "configs" / "generated_falling_impact"


HEIGHT_CASES = [
    {"name": "h0p13", "center_y": 0.25},
    {"name": "h0p28", "center_y": 0.40},
    {"name": "h0p48", "center_y": 0.60},
]

FLUID_CASES = [
    {"name": "base", "Re2": 100.0, "We2": 0.50, "Pe": 15.0, "Fr": 0.35},
    {"name": "inertial", "Re2": 200.0, "We2": 1.00, "Pe": 20.0, "Fr": 0.35},
]

CONTACT_ANGLES = [60, 120]


def load_base() -> dict:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


def configure_case(base: dict, *, height: dict, fluid: dict, contact_angle: int) -> tuple[str, dict]:
    cfg = copy.deepcopy(base)
    case_name = f"impact_{height['name']}_ca{contact_angle}_{fluid['name']}"

    cfg["description"] = (
        f"Falling droplet impact: gap={height['name'][1:].replace('p', '.')}, "
        f"CA={contact_angle}, regime={fluid['name']}."
    )

    phys = cfg["physical_params"]
    phys["rho1"] = 0.001225
    phys["rho2"] = 1.0
    phys["Re1"] = max(10.0, fluid["Re2"] * phys["rho1"])
    phys["Re2"] = fluid["Re2"]
    phys["We1"] = max(0.001, fluid["We2"] * phys["rho1"])
    phys["We2"] = fluid["We2"]
    phys["Pe"] = fluid["Pe"]
    phys["epsilon"] = 0.04
    phys["contact_angle"] = contact_angle
    phys["include_gravity"] = True
    phys["Fr"] = fluid["Fr"]
    phys["g"] = -1.0
    phys["surface_tension"] = {
        "smooth_curvature": False,
        "smoothing_radius": 1,
    }

    grid = cfg["grid_params"]
    grid["Lx"] = 1.0
    grid["Ly"] = 1.2
    grid["Nx"] = 128
    grid["Ny"] = 160

    time = cfg["time_params"]
    time["dt"] = 0.0005
    time["dt_initial"] = 0.0002
    time["t_max"] = 0.40
    time["checkpoint_interval"] = 2500
    time["cfl_number"] = 0.10
    time["capillary_cfl_number"] = 0.10
    time["curvature_cfl_number"] = 0.10

    ic = cfg["initial_conditions"]
    ic["type"] = "droplet"
    ic["droplet_radius"] = 0.12
    ic["droplet_center_x"] = 0.50
    ic["droplet_center_y"] = height["center_y"]
    ic["is_bubble"] = False
    ic["initial_velocity"] = {"u": 0.0, "v": 0.0}

    bcs = cfg["boundary_conditions"]
    bcs["pressure"] = {
        "top": "open",
        "bottom": "neumann",
        "left": "neumann",
        "right": "neumann",
        "open_pressure": 0.0,
    }
    bcs["velocity"] = {
        "top": "do_nothing",
        "bottom": "no_slip",
        "left": "slip_symmetry",
        "right": "slip_symmetry",
    }
    bcs["phase_field"] = {
        "top": "neumann",
        "bottom": "contact_angle",
        "left": "neumann",
        "right": "neumann",
        "contact_angle_method": "ghost_cell",
        "use_cox_voinov": False,
        "contact_angle_ghost_law": "analytic_gradient",
        "contact_mask_soft_band": 0.8,
        "contact_mask_grad_scale": 0.5,
        "contact_angle_full_wall": True,
    }
    bcs["chemical_potential"] = {
        "top": "zero_flux",
        "bottom": "zero_flux",
        "left": "zero_flux",
        "right": "zero_flux",
    }
    bcs["advection"] = {
        "top": "open",
        "bottom": "impermeable",
        "left": "impermeable",
        "right": "impermeable",
        "cout": 1.0,
    }

    solver = cfg["solver_params"]
    solver["phase_field_solver"] = "ghost_cell"
    solver["phase_update_mode"] = "semi_implicit_ch"
    solver["phase_diffusion_solver_backend"] = "pyamg"
    solver["phase_diffusion_solver_tol"] = 1e-8
    solver["phase_diffusion_solver_maxiter"] = 500
    solver["pressure_solver"] = {
        "backend": "pyamg",
        "accel": "bicgstab",
        "tol": 0.05,
        "maxiter": 1000,
    }
    solver["correction_solver"] = {
        "backend": "pyamg",
        "accel": "bicgstab",
        "tol": 0.05,
        "maxiter": 1000,
    }
    solver["ppe"] = {
        "mean_div_threshold": 0.15,
        "max_div_threshold": 0.25,
        "use_local_ppe": False,
        "local_threshold_factor": 0.2,
        "max_iterations": 1000,
        "buffer_size": 5,
    }
    solver["physical_pressure"] = {
        "boundary_conditions": {
            "top": "neumann",
            "bottom": "neumann",
            "left": "neumann",
            "right": "neumann",
        }
    }

    cfg["visualization"] = {"use_pyvista": False}
    cfg["restart"] = {"restart_from": None}

    return case_name, cfg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()

    generated = []
    for height in HEIGHT_CASES:
        for fluid in FLUID_CASES:
            for contact_angle in CONTACT_ANGLES:
                case_name, cfg = configure_case(
                    base,
                    height=height,
                    fluid=fluid,
                    contact_angle=contact_angle,
                )
                target = OUT_DIR / f"{case_name}.json"
                with target.open("w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                    f.write("\n")
                generated.append(target.name)

    print(f"Generated {len(generated)} configs in {OUT_DIR}")
    for name in generated:
        print(name)


if __name__ == "__main__":
    main()
