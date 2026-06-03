#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from numerics.finite_differences import jax_gradient
from simulation.two_phase import TwoPhaseSimulation
import boundary_conditions.contact_angle_bc as cab_mod


def liquid_mass(phi: np.ndarray, dx: float, dy: float) -> float:
    liquid_fraction = np.clip(0.5 * (1.0 - np.asarray(phi)), 0.0, 1.0)
    return float(np.sum(liquid_fraction) * dx * dy)


def interface_area(phi: np.ndarray, dx: float, dy: float, threshold: float = 0.9) -> float:
    return float(np.sum(np.abs(np.asarray(phi)) < threshold) * dx * dy)


def negative_phi_area(phi: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(np.asarray(phi) < 0.0) * dx * dy)


@contextmanager
def patch_simple_contact_angle(use_relaxation: bool):
    original = cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax

    def relaxed_simple(self, phi, dx, dy, geometry, psi=None, U=None, bottom_velocity_bc="no_slip"):
        theta_effective = self._get_effective_contact_angle_jax(
            psi, U=U, contact_line_velocity=None, bottom_velocity_bc=bottom_velocity_bc
        )
        grad_phi = jax_gradient(phi, dx, dy, geometry.f_1_grid)
        grad_phi_x = grad_phi[:, 1, 0]
        grad_phi_y = grad_phi[:, 1, 1]
        norm_grad_phi = jnp.sqrt(grad_phi_x**2 + grad_phi_y**2)
        norm_grad_phi = jnp.where(norm_grad_phi < 1e-10, 1e-10, norm_grad_phi)
        normal_derivative = -jnp.cos(theta_effective) * norm_grad_phi

        phi_bottom = phi[:, 0]
        phi_above = phi[:, 1]
        phi_contact_target = phi_above - normal_derivative * dy
        phi_neumann_target = phi_above

        phi_crosses_zero = (phi_bottom * phi_above) < 0.0
        phi_near_zero = (jnp.abs(phi_bottom) < 0.5) | (jnp.abs(phi_above) < 0.5)
        has_interface = norm_grad_phi > 1e-3
        contact_mask = (phi_crosses_zero | phi_near_zero) & has_interface

        if use_relaxation:
            alpha = jnp.clip(self.contact_angle_relaxation, 0.0, 1.0)
            phi_contact_new = (1.0 - alpha) * phi_bottom + alpha * phi_contact_target
            phi_noncontact_new = (1.0 - alpha) * phi_bottom + alpha * phi_neumann_target
            phi_bottom_new = jnp.where(contact_mask, phi_contact_new, phi_noncontact_new)
        else:
            phi_bottom_new = jnp.where(contact_mask, phi_contact_target, phi_neumann_target)

        phi_new = phi.at[:, 0].set(phi_bottom_new)
        phi_new = phi_new.at[:, -1].set(phi_new[:, -2])
        phi_new = phi_new.at[0, :].set(phi_new[1, :])
        phi_new = phi_new.at[-1, :].set(phi_new[-2, :])
        return phi_new

    try:
        cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax = relaxed_simple
        yield
    finally:
        cab_mod.ContactAngleBoundaryCondition._simple_contact_angle_jax = original


def build_variants(base_cfg: dict):
    variants = []

    baseline = copy.deepcopy(base_cfg)
    variants.append(("baseline", baseline, False))

    no_cox = copy.deepcopy(base_cfg)
    no_cox["boundary_conditions"]["phase_field"]["use_cox_voinov"] = False
    variants.append(("no_cox", no_cox, False))

    relaxed = copy.deepcopy(base_cfg)
    relaxed["boundary_conditions"]["phase_field"]["contact_angle_relaxation"] = 0.2
    variants.append(("relaxed_alpha_0p2", relaxed, True))

    relaxed_no_cox = copy.deepcopy(base_cfg)
    relaxed_no_cox["boundary_conditions"]["phase_field"]["contact_angle_relaxation"] = 0.2
    relaxed_no_cox["boundary_conditions"]["phase_field"]["use_cox_voinov"] = False
    variants.append(("relaxed_alpha_0p2_no_cox", relaxed_no_cox, True))

    return variants


def run_variant(
    name: str,
    cfg: dict,
    output_dir: Path,
    steps: int,
    log_every: int,
    patch_relaxed_simple: bool,
    restart_from: Optional[str] = None,
):
    variant_dir = output_dir / name
    os.makedirs(variant_dir, exist_ok=True)

    if restart_from is not None:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("restart", {})
        cfg["restart"]["restart_from"] = restart_from

    with patch_simple_contact_angle(use_relaxation=patch_relaxed_simple):
        sim = TwoPhaseSimulation(cfg, output_dir=str(variant_dir))
        if restart_from is None:
            sim.state.phi = sim.state.phase_field_bc.apply_boundary_conditions(
                sim.state.phi, sim.state.dx, sim.state.dy, use_jax=True, geometry=sim.state.geometry
            )
        start_step = int(sim.state.step)

        rows = []
        for _ in range(steps + 1):
            phi_np = np.asarray(sim.state.phi)
            rows.append(
                {
                    "step": int(sim.state.step),
                    "time": float(sim.state.t),
                    "mass": liquid_mass(phi_np, sim.state.dx, sim.state.dy),
                    "neg_area": negative_phi_area(phi_np, sim.state.dx, sim.state.dy),
                    "interface_area": interface_area(phi_np, sim.state.dx, sim.state.dy),
                    "phi_min": float(phi_np.min()),
                    "phi_max": float(phi_np.max()),
                }
            )
            if (sim.state.step - start_step) >= steps:
                break

            sim.state.dt = sim.dt_initial if sim.state.step < 500 else sim.dt_normal
            sim.step()
            sim._after_step_cleanup()
            sim.state.step += 1

        csv_path = variant_dir / "mass_loss.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows[::max(1, log_every)]:
                writer.writerow(row)
            if rows[-1] is not rows[::max(1, log_every)][-1]:
                writer.writerow(rows[-1])

    first = rows[0]
    last = rows[-1]
    return {
        "variant": name,
        "steps": steps,
        "mass0": first["mass"],
        "massN": last["mass"],
        "mass_delta": last["mass"] - first["mass"],
        "mass_delta_pct": 100.0 * (last["mass"] - first["mass"]) / first["mass"],
        "neg_area0": first["neg_area"],
        "neg_areaN": last["neg_area"],
        "interface0": first["interface_area"],
        "interfaceN": last["interface_area"],
        "phi_min0": first["phi_min"],
        "phi_minN": last["phi_min"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare local slide mass loss across BC/Cox–Voinov variants")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--restart-from", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variants to run: baseline no_cox relaxed_alpha_0p2 relaxed_alpha_0p2_no_cox",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    selected_variants = set(args.variants) if args.variants else None

    results = []
    for name, cfg, patch_relaxed_simple in build_variants(base_cfg):
        if selected_variants is not None and name not in selected_variants:
            continue
        print(f"\n=== Running {name} ===")
        result = run_variant(
            name,
            cfg,
            output_dir,
            args.steps,
            args.log_every,
            patch_relaxed_simple,
            restart_from=args.restart_from,
        )
        results.append(result)
        print(
            f"{name}: mass {result['mass0']:.8f} -> {result['massN']:.8f} "
            f"({result['mass_delta_pct']:+.3f}%)"
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for result in results:
        print(
            f"{result['variant']:24s} "
            f"mass_delta={result['mass_delta']:+.8f} "
            f"({result['mass_delta_pct']:+.3f}%) "
            f"phi_min: {result['phi_min0']:.4f} -> {result['phi_minN']:.4f} "
            f"interface: {result['interface0']:.6f} -> {result['interfaceN']:.6f}"
        )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
