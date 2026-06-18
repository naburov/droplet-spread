#!/usr/bin/env python3
"""Generate chainsaw A/B debug configs from a shared reference template."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def base_config(*, nx: int = 128, t_max: float = 0.04) -> dict:
    return {
        "description": "Chainsaw A/B reference (user mitigation stack, analytic ghost).",
        "physical_params": {
            "rho1": 0.001225,
            "rho2": 1.0,
            "Re1": 35.0,
            "Re2": 500.0,
            "We1": 0.01,
            "We2": 2.0,
            "Pe": 20.0,
            "epsilon": 0.04,
            "contact_angle": 90,
            "include_gravity": True,
            "Fr": 1.0,
            "g": -1.0,
            "atm_pressure": 0.0,
            "lambda_willmore": 0.0,
            "epsilon_willmore": 0.0,
            "surface_tension": {"smooth_curvature": True, "smoothing_radius": 1},
        },
        "grid_params": {"Lx": 1.0, "Ly": 1.0, "Nx": nx, "Ny": nx},
        "time_params": {
            "dt": 0.0005,
            "dt_initial": 0.00025,
            "t_max": t_max,
            "checkpoint_interval": 500 if nx <= 64 else 800,
            "cfl_number": 0.1,
            "capillary_cfl_number": 0.1,
            "curvature_cfl_number": 0.1,
        },
        "initial_conditions": {
            "type": "droplet",
            "droplet_radius": 0.15,
            "droplet_center_x": 0.5,
            "droplet_center_y": 0.0,
            "pressure_drop": 0.01,
        },
        "boundary_conditions": {
            "pressure": {
                "top": "neumann",
                "bottom": "neumann",
                "left": "neumann",
                "right": "dirichlet",
                "dirichlet_values": {"right": 0.0},
            },
            "velocity": {
                "top": "dirichlet",
                "bottom": "navier_slip",
                "slip_length": 0.005,
                "left": "dirichlet",
                "right": "do_nothing",
                "dirichlet_values": {
                    "top": {"u": 1.0, "v": 0.0},
                    "left": {"u": 1.0, "v": 0.0},
                },
                "dirichlet_profiles": {
                    "left": {
                        "type": "boundary_layer",
                        "subtype": "slip_blasius",
                        "include_normal_velocity": False,
                        "reynolds_number": "Re2",
                        "characteristic_length": "Ly",
                        "bl_exponent": 0.5,
                    }
                },
            },
            "phase_field": {
                "top": "neumann",
                "bottom": "contact_angle",
                "left": "neumann",
                "right": "neumann",
                "contact_angle_method": "ghost_cell",
                "use_cox_voinov": True,
                "cox_voinov_coefficient": 1.0,
                "cox_voinov_exponent": 0.333,
                "contact_angle_ghost_law": "analytic_gradient",
                "contact_angle_full_wall": False,
                "contact_mask_soft_band": 0.8,
                "contact_mask_grad_scale": 0.5,
            },
            "chemical_potential": {
                "top": "zero_flux",
                "bottom": "zero_flux",
                "left": "zero_flux",
                "right": "zero_flux",
            },
            "advection": {
                "top": "open",
                "bottom": "impermeable",
                "left": "impermeable",
                "right": "impermeable",
                "cout": 1.0,
            },
        },
        "solver_params": {
            "velocity_layout": "staggered",
            "phase_field_solver": "ghost_cell",
            "phase_update_mode": "semi_implicit_ch",
            "phase_diffusion_solver_backend": "pyamg",
            "phase_diffusion_solver_tol": 1e-8,
            "phase_diffusion_solver_maxiter": 500,
            "chainsaw_diagnostics": True,
            "pressure_solver": {
                "backend": "pyamg",
                "accel": "bicgstab",
                "tol": 0.05,
                "maxiter": 1000,
            },
            "correction_solver": {
                "backend": "pyamg",
                "accel": "bicgstab",
                "tol": 0.05,
                "maxiter": 1000,
            },
            "ppe": {
                "mean_div_threshold": 0.15,
                "max_div_threshold": 0.25,
                "convergence_mode": "interior",
                "use_local_ppe": False,
                "max_iterations": 1000,
                "buffer_size": 5,
            },
        },
        "visualization": {"use_pyvista": False},
        "restart": {"restart_from": None},
    }


def fast_wang_base(*, nx: int = 64) -> dict:
    """Accelerated chainsaw repro: Wang-like shear + larger dt cap + weaker kappa smooth."""
    cfg = base_config(nx=nx, t_max=0.055)
    cfg["description"] = f"[{nx}²] Fast chainsaw driver (Wang-like shear, larger dt, full_wall optional per variant)"
    cfg["physical_params"].update(
        {
            "Re1": 52.5,
            "Re2": 750.0,
            "We1": 0.0225,
            "We2": 4.5,
            "contact_angle": 120,
            "surface_tension": {"smooth_curvature": False, "smoothing_radius": 1},
        }
    )
    cfg["initial_conditions"]["droplet_radius"] = 0.12
    cfg["initial_conditions"]["droplet_center_x"] = 0.45
    cfg["boundary_conditions"]["velocity"]["slip_length"] = 0.02
    cfg["boundary_conditions"]["velocity"]["dirichlet_values"] = {
        "top": {"u": 1.5, "v": 0.0},
        "left": {"u": 1.5, "v": 0.0},
    }
    cfg["time_params"].update(
        {
            "dt": 0.001,
            "dt_initial": 0.0008,
            "t_max": 0.055,
            "checkpoint_interval": 400,
            "cfl_number": 0.35,
            "capillary_cfl_number": 0.35,
            "curvature_cfl_number": 0.3,
        }
    )
    return cfg


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Only write accelerated fast_* configs (for quicker onset)",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "configs" / "debug" / "chainsaw_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "ref": ("Reference: analytic ghost + Cox-Voinov + smooth curvature", {}),
        "A_no_cox": ("Experiment A: Cox-Voinov off", {"phase_field": {"use_cox_voinov": False}}),
        "B_wall_energy": (
            "Experiment B: wall_energy ghost law",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "wall_energy",
                    "contact_angle_full_wall": False,
                    "contact_mask_soft_band": 0.8,
                    "contact_mask_grad_scale": 0.5,
                }
            },
        ),
        "C_no_smooth_kappa": (
            "Experiment C: smooth_curvature off",
            {"physical_params": {"surface_tension": {"smooth_curvature": False, "smoothing_radius": 1}}},
        ),
        "D_strict_ppe": (
            "Experiment D: stricter PPE",
            {
                "solver_params": {
                    "ppe": {
                        "convergence_mode": "both",
                        "max_iterations": 20,
                        "under_relaxation": 0.7,
                    },
                    "correction_solver": {"tol": 1e-4},
                }
            },
        ),
    }

    def deep_update(base: dict, patch: dict) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value

    if not args.fast_only:
        for grid_label, nx in (("64", 64), ("128", 128)):
            for name, (desc, patch) in variants.items():
                cfg = copy.deepcopy(base_config(nx=nx, t_max=0.04 if nx == 128 else 0.02))
                cfg["description"] = f"[{grid_label}²] {desc}"
                deep_update(cfg, patch)
                path = out_dir / f"chainsaw_ab_{name}_{grid_label}.json"
                path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {path}")

    fast_variants = {
        "fast_analytic_fullwall": (
            "Fast onset: analytic_gradient + contact_angle_full_wall (production failure mode)",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                }
            },
        ),
        "fast_analytic_masked": (
            "Fast onset: analytic + soft mask (no full_wall), same Wang shear/dt",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": False,
                    "contact_mask_soft_band": 0.8,
                    "contact_mask_grad_scale": 0.5,
                }
            },
        ),
        "fast_wall_energy": (
            "Fast onset: wall_energy + masked (expected stable)",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "wall_energy",
                    "contact_angle_full_wall": False,
                    "contact_mask_soft_band": 0.8,
                    "contact_mask_grad_scale": 0.5,
                },
                "physical_params": {
                    "surface_tension": {"smooth_curvature": True, "smoothing_radius": 1}
                },
            },
        ),
        "fast_no_cox_voinov": (
            "Fast onset: full_wall analytic but Cox-Voinov off",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                    "use_cox_voinov": False,
                }
            },
        ),
        "fast_analytic_fullwall_no_surface_tension_bc_overwrite": (
            "Fast onset: full_wall analytic, no ST boundary overwrite",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "physical_params": {
                    "surface_tension": {
                        "smooth_curvature": False,
                        "smoothing_radius": 1,
                        "apply_boundary_overwrite": False,
                    }
                },
            },
        ),
        "fast_analytic_fullwall_no_curvature_smoothing": (
            "Fast onset: full_wall analytic (explicit no kappa smooth; same as default fast fullwall)",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "physical_params": {
                    "surface_tension": {"smooth_curvature": False, "smoothing_radius": 1}
                },
            },
        ),
    }

    for grid_label, nx in (("64", 64), ("128", 128)):
        for name, (desc, patch) in fast_variants.items():
            cfg = copy.deepcopy(fast_wang_base(nx=nx))
            cfg["description"] = f"[{grid_label}²] {desc}"
            deep_update(cfg, patch)
            path = out_dir / f"chainsaw_ab_{name}_{grid_label}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")

    stage_ab = {
        "fast_analytic_fullwall_stage": (
            "Stage trace: full_wall analytic (baseline)",
            {
                "solver_params": {
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                }
            },
        ),
        "fast_analytic_fullwall_freeze_wall_after_solve": (
            "Stage A/B: freeze phi[:,0] after linear solve",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "solver_params": {
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                    "phase_debug": {"freeze_wall_after_solve": True},
                },
            },
        ),
        "fast_analytic_fullwall_zero_phase_advection": (
            "Stage A/B: zero phase advection",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "solver_params": {
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                    "phase_debug": {"zero_phase_advection": True},
                },
            },
        ),
        "fast_analytic_fullwall_skip_ch_diffusion": (
            "Stage A/B: skip explicit CH diffusion in RHS",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "solver_params": {
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                    "phase_debug": {"skip_ch_diffusion": True},
                },
            },
        ),
        "fast_analytic_fullwall_no_conserve_phi_sum": (
            "Stage A/B: disable global phi-sum preservation",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                    "conserve_phi_sum": False,
                },
                "solver_params": {
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                },
            },
        ),
        "fast_analytic_fullwall_explicit_monolithic": (
            "Stage A/B: explicit monolithic ghost update (not semi_implicit_ch)",
            {
                "phase_field": {
                    "contact_angle_ghost_law": "analytic_gradient",
                    "contact_angle_full_wall": True,
                    "contact_mask_soft_band": 0.0,
                    "contact_mask_grad_scale": 0.0,
                },
                "solver_params": {
                    "phase_update_mode": "monolithic",
                    "phase_stage_diagnostics": True,
                    "chainsaw_diagnostics": True,
                },
            },
        ),
    }

    for grid_label, nx in (("64", 64),):
        for name, (desc, patch) in stage_ab.items():
            cfg = copy.deepcopy(fast_wang_base(nx=nx))
            cfg["description"] = f"[{grid_label}²] {desc}"
            deep_update(cfg, patch)
            path = out_dir / f"chainsaw_ab_{name}_{grid_label}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")

    split_base_pf = {
        "contact_angle_ghost_law": "analytic_gradient",
        "contact_angle_full_wall": True,
        "contact_mask_soft_band": 0.0,
        "contact_mask_grad_scale": 0.0,
    }
    split_base_solver = {
        "phase_stage_diagnostics": True,
        "chainsaw_diagnostics": True,
        "phase_field_solver": "ghost_cell",
        "phase_update_mode": "semi_implicit_ch",
    }

    contact_split_experiments = {
        "fast_split_explicit_delta": (
            "Contact split baseline (explicit_delta)",
            {"solver_params": {"semi_implicit_contact_split": "explicit_delta"}},
        ),
        "fast_split_no_delta": (
            "Contact split: omit contact_delta_term",
            {"solver_params": {"semi_implicit_contact_split": "no_delta"}},
        ),
        "fast_split_filtered_delta": (
            "Contact split: x-lowpass contact_delta near wall",
            {"solver_params": {"semi_implicit_contact_split": "filtered_delta"}},
        ),
        "fast_split_damped_delta_beta0": (
            "Contact split: damp bottom contact_delta_term (beta=0)",
            {
                "solver_params": {
                    "semi_implicit_contact_split": "damped_delta",
                    "semi_implicit_contact_delta_beta": 0.0,
                }
            },
        ),
        "fast_split_damped_delta_beta025": (
            "Contact split: damp bottom contact_delta_term (beta=0.25)",
            {
                "solver_params": {
                    "semi_implicit_contact_split": "damped_delta",
                    "semi_implicit_contact_delta_beta": 0.25,
                }
            },
        ),
        "fast_split_damped_delta_beta05": (
            "Contact split: damp bottom contact_delta_term (beta=0.5)",
            {
                "solver_params": {
                    "semi_implicit_contact_split": "damped_delta",
                    "semi_implicit_contact_delta_beta": 0.5,
                }
            },
        ),
        "fast_split_implicit_wall_energy": (
            "Contact split: semi-implicit wall_energy bottom (A_phi diagonal)",
            {"solver_params": {"semi_implicit_contact_split": "implicit_wall_energy"}},
        ),
        "fast_split_explicit_ghost": (
            "Contact split: bypass semi-implicit (jax_update_phase_ghost)",
            {"solver_params": {"semi_implicit_contact_split": "explicit_ghost"}},
        ),
    }

    for grid_label, nx in (("64", 64),):
        for name, (desc, patch) in contact_split_experiments.items():
            cfg = copy.deepcopy(fast_wang_base(nx=nx))
            cfg["description"] = f"[{grid_label}²] {desc}"
            deep_update(cfg, {"phase_field": split_base_pf, "solver_params": split_base_solver})
            deep_update(cfg, patch)
            path = out_dir / f"chainsaw_ab_{name}_{grid_label}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
