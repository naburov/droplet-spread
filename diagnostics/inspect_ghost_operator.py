import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from boundary_conditions.contact_angle_bc import ContactAngleBoundaryCondition
from boundary_conditions.chemical_potential_bc import (
    BC_NEUMANN,
    jax_apply_chemical_potential_bc,
)
from numerics.finite_differences import jax_gradient, jax_laplacian, jax_norm
from physics.phase_field import (
    jax_apply_chemical_potential_bc_skip_bottom,
    jax_laplacian_flat_bottom_ghost,
)
from physics.properties import jax_df_2


def load_config(path):
    return json.loads(Path(path).read_text())


def make_contact_angle_bc(config):
    phys = config["physical_params"]
    bc = config["boundary_conditions"]["phase_field"]
    return ContactAngleBoundaryCondition(
        contact_angle=phys["contact_angle"],
        method=bc.get("contact_angle_method", "simple"),
        epsilon=phys.get("epsilon"),
        contact_angle_ice=bc.get("contact_angle_ice"),
        use_ice_aware=bc.get("use_ice_aware", False),
        use_geometry_aware=bc.get("use_geometry_aware", False),
        use_cox_voinov=bc.get("use_cox_voinov", False),
        cox_voinov_coefficient=bc.get("cox_voinov_coefficient", 1.0),
        cox_voinov_exponent=bc.get("cox_voinov_exponent", 1.0 / 3.0),
        contact_angle_relaxation=bc.get("contact_angle_relaxation", 1.0),
    )


def zero_crossings(arr):
    out = []
    for i in range(len(arr) - 1):
        a = float(arr[i])
        b = float(arr[i + 1])
        if a == 0.0:
            out.append(i)
        elif a * b < 0.0:
            t = abs(a) / (abs(a) + abs(b))
            out.append(i + t)
    return out


def fit_contact_angle(phi, dx, dy, rows=range(0, 7)):
    left = []
    right = []
    for j in rows:
        z = zero_crossings(phi[:, j])
        y = j * dy
        if len(z) >= 1:
            left.append((z[0] * dx, y))
        if len(z) >= 2:
            right.append((z[-1] * dx, y))

    def calc(pts, side):
        if len(pts) < 2:
            return None
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope, _ = np.polyfit(ys, xs, 1)
        angle = np.degrees(np.arctan2(1.0, slope if side == "left" else -slope))
        return float(angle)

    return calc(left, "left"), calc(right, "right")


def analyze_checkpoint(checkpoint_path, config_path):
    config = load_config(config_path)
    data = np.load(checkpoint_path, allow_pickle=True)
    phi = jnp.asarray(data["phi"])
    U = jnp.asarray(data["U"])

    nx, ny = phi.shape
    dx = config["grid_params"]["Lx"] / nx
    dy = config["grid_params"]["Ly"] / ny
    epsilon = config["physical_params"]["epsilon"]
    pe = config["physical_params"]["Pe"]

    zeros = jnp.zeros((nx, ny), dtype=phi.dtype)
    geometry = SimpleNamespace(f_1_grid=zeros, f_2_grid=zeros)

    contact_angle_bc = make_contact_angle_bc(config)
    bottom_velocity_bc = config["boundary_conditions"]["velocity"].get("bottom", "no_slip")
    theta_eff = contact_angle_bc._get_effective_contact_angle_jax(
        None, U=U, contact_line_velocity=None, bottom_velocity_bc=bottom_velocity_bc
    )
    bottom_ghost = contact_angle_bc.build_bottom_ghost_row_jax(
        phi, dx, dy, geometry, psi=None, U=U, bottom_velocity_bc=bottom_velocity_bc
    )

    grad_phi = jax_gradient(phi, dx, dy, zeros)
    convective_term = U[..., 0] * grad_phi[..., 0] + U[..., 1] * grad_phi[..., 1]
    grad_mag = jax_norm(grad_phi)

    lap_simple = jax_laplacian(phi, dx, dy, zeros, zeros)
    lap_ghost = jax_laplacian_flat_bottom_ghost(phi, dx, dy, bottom_ghost)

    mu_simple_raw = jax_df_2(phi) - epsilon**2 * lap_simple
    mu_ghost_raw = jax_df_2(phi) - epsilon**2 * lap_ghost

    mu_simple_bc = jax_apply_chemical_potential_bc(
        mu_simple_raw, dx, dy,
        top_bc=BC_NEUMANN, bottom_bc=BC_NEUMANN, left_bc=BC_NEUMANN, right_bc=BC_NEUMANN,
    )
    mu_ghost_bc = jax_apply_chemical_potential_bc_skip_bottom(
        mu_ghost_raw, dx, dy,
        top_bc=BC_NEUMANN, bottom_bc=BC_NEUMANN, left_bc=BC_NEUMANN, right_bc=BC_NEUMANN,
    )

    mu_simple_centered = mu_simple_bc - jnp.mean(mu_simple_bc)
    mu_ghost_centered = mu_ghost_bc - jnp.mean(mu_ghost_bc)

    diff_simple = (1.0 / pe) * jax_laplacian(mu_simple_centered, dx, dy, zeros, zeros)
    diff_ghost = (1.0 / pe) * jax_laplacian_flat_bottom_ghost(mu_ghost_centered, dx, dy, mu_ghost_centered[:, 0])

    phi_np = np.asarray(phi)
    report = {
        "checkpoint": str(checkpoint_path),
        "step": int(data["step"]),
        "phi_min": float(phi_np.min()),
        "phi_max": float(phi_np.max()),
        "mass": float((0.5 * (1.0 - phi_np)).sum() * dx * dy),
        "left_angle_deg": fit_contact_angle(phi_np, dx, dy)[0],
        "right_angle_deg": fit_contact_angle(phi_np, dx, dy)[1],
        "theta_eff_deg_min": float(np.degrees(np.asarray(theta_eff)).min()),
        "theta_eff_deg_max": float(np.degrees(np.asarray(theta_eff)).max()),
        "u_wall_min": float(np.asarray(U[:, 0, 0]).min()),
        "u_wall_max": float(np.asarray(U[:, 0, 0]).max()),
        "ghost_minus_wall_max": float(np.max(np.abs(np.asarray(bottom_ghost - phi[:, 0])))),
        "lap_bottom_simple_max": float(np.max(np.abs(np.asarray(lap_simple[:, 0])))),
        "lap_bottom_ghost_max": float(np.max(np.abs(np.asarray(lap_ghost[:, 0])))),
        "lap_row1_simple_max": float(np.max(np.abs(np.asarray(lap_simple[:, 1])))),
        "lap_row1_ghost_max": float(np.max(np.abs(np.asarray(lap_ghost[:, 1])))),
        "mu_raw_bottom_diff_max": float(np.max(np.abs(np.asarray(mu_ghost_raw[:, 0] - mu_simple_raw[:, 0])))),
        "mu_bc_bottom_diff_max": float(np.max(np.abs(np.asarray(mu_ghost_bc[:, 0] - mu_simple_bc[:, 0])))),
        "mu_bc_row1_diff_max": float(np.max(np.abs(np.asarray(mu_ghost_bc[:, 1] - mu_simple_bc[:, 1])))),
        "diff_bottom_diff_max": float(np.max(np.abs(np.asarray(diff_ghost[:, 0] - diff_simple[:, 0])))),
        "diff_row1_diff_max": float(np.max(np.abs(np.asarray(diff_ghost[:, 1] - diff_simple[:, 1])))),
        "convective_bottom_max": float(np.max(np.abs(np.asarray(convective_term[:, 0])))),
        "gradmag_bottom_max": float(np.max(np.abs(np.asarray(grad_mag[:, 0])))),
    }
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    for checkpoint in args.checkpoints:
        report = analyze_checkpoint(checkpoint, args.config)
        print(json.dumps(report, indent=2))
