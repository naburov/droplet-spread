import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from physics.phase_field import PhaseFieldSolverGhostCell, PhaseFieldSolverSimple


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


def build_couette_field(nx, ny, ly, u_top):
    y = (np.arange(ny) + 0.5) * (ly / ny)
    u = u_top * y / ly
    U = np.zeros((nx, ny, 2), dtype=np.float32)
    U[..., 0] = u[None, :]
    return jnp.asarray(U)


def run_replay(config_path, checkpoint_path, steps, u_top):
    config = json.loads(Path(config_path).read_text())
    data = np.load(checkpoint_path, allow_pickle=True)
    phi = jnp.asarray(data["phi"])

    nx, ny = phi.shape
    lx = config["grid_params"]["Lx"]
    ly = config["grid_params"]["Ly"]
    dx = lx / nx
    dy = ly / ny
    dt = float(config["time_params"]["dt"])

    geometry = SimpleNamespace(
        f_1_grid=jnp.zeros((nx, ny), dtype=phi.dtype),
        f_2_grid=jnp.zeros((nx, ny), dtype=phi.dtype),
    )
    U = build_couette_field(nx, ny, ly, u_top)

    phys = config["physical_params"]
    solver_cls = PhaseFieldSolverGhostCell if config["solver_params"]["phase_field_solver"] == "ghost_cell" else PhaseFieldSolverSimple
    solver = solver_cls(phys["Pe"], phys["epsilon"], phys["contact_angle"], config=config)

    records = []
    for step in range(steps + 1):
        phi_np = np.asarray(phi)
        mass = float((0.5 * (1.0 - phi_np)).sum() * dx * dy)
        left_angle, right_angle = fit_contact_angle(phi_np, dx, dy)
        records.append(
            {
                "step": step,
                "mass": mass,
                "phi_min": float(phi_np.min()),
                "phi_max": float(phi_np.max()),
                "left_angle_deg": left_angle,
                "right_angle_deg": right_angle,
            }
        )
        if step == steps:
            break
        phi = solver.update(phi, U, dt, dx, dy, geometry, use_jax=True, psi=None)

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--u-top", type=float, default=1.0)
    args = parser.parse_args()

    records = run_replay(args.config, args.checkpoint, args.steps, args.u_top)
    print(json.dumps(records, indent=2))
