#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from numerics.staggered_mac import divergence as mac_divergence


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "simulation_parameters.json", "r") as f:
        return json.load(f)


def load_checkpoint(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def liquid_fraction(phi: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * (1.0 - phi), 0.0, 1.0)


def summarize_checkpoint(path: Path, dx: float, dy: float) -> dict:
    ck = load_checkpoint(path)
    phi = np.asarray(ck["phi"], dtype=np.float64)
    u_face = np.asarray(ck["u_face"], dtype=np.float64)
    v_face = np.asarray(ck["v_face"], dtype=np.float64)
    div = np.asarray(mac_divergence(u_face, v_face, dx, dy), dtype=np.float64)

    abs_div = np.abs(div)
    liquid = liquid_fraction(phi)
    Ny = phi.shape[1]
    Nx = phi.shape[0]

    strips = {
        "left": abs_div[:2, :],
        "right": abs_div[-2:, :],
        "bottom": abs_div[:, :2],
        "top": abs_div[:, -2:],
        "interior": abs_div[2:-2, 2:-2] if Nx > 4 and Ny > 4 else abs_div,
        "top_left_corner": abs_div[:4, -4:],
        "bottom_left_corner": abs_div[:4, :4],
        "top_right_corner": abs_div[-4:, -4:],
        "bottom_right_corner": abs_div[-4:, :4],
    }

    argmax = np.unravel_index(np.argmax(abs_div), abs_div.shape)
    i_max, j_max = int(argmax[0]), int(argmax[1])

    return {
        "step": int(ck["step"]),
        "time": float(ck.get("t", 0.0)),
        "dt": float(ck.get("dt", 0.0)),
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
        "liquid_mass": float(liquid.sum() * dx * dy),
        "negative_phi_area": float((phi < 0.0).sum() * dx * dy),
        "global_max_div": float(abs_div.max()),
        "global_mean_div": float(abs_div.mean()),
        "interior_max_div": float(np.max(strips["interior"])),
        "left_max_div": float(np.max(strips["left"])),
        "right_max_div": float(np.max(strips["right"])),
        "bottom_max_div": float(np.max(strips["bottom"])),
        "top_max_div": float(np.max(strips["top"])),
        "top_left_corner_max_div": float(np.max(strips["top_left_corner"])),
        "bottom_left_corner_max_div": float(np.max(strips["bottom_left_corner"])),
        "top_right_corner_max_div": float(np.max(strips["top_right_corner"])),
        "bottom_right_corner_max_div": float(np.max(strips["bottom_right_corner"])),
        "max_div_i": i_max,
        "max_div_j": j_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    cfg = load_config(exp_dir)
    dx = float(cfg["grid_params"]["Lx"]) / int(cfg["grid_params"]["Nx"])
    dy = float(cfg["grid_params"]["Ly"]) / int(cfg["grid_params"]["Ny"])

    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.npz"))
    if args.steps is not None and len(args.steps) > 0:
        requested = {f"checkpoint_{step:06d}.npz" for step in args.steps}
        checkpoints = [path for path in checkpoints if path.name in requested]

    rows = [summarize_checkpoint(path, dx, dy) for path in checkpoints]
    if not rows:
        raise SystemExit("No checkpoints selected")

    first = rows[0]
    print(f"Experiment: {exp_dir}")
    print(f"dx={dx:.6f} dy={dy:.6f}")
    print("step time liquid_mass mass_delta neg_phi_area global_max interior_max left_max top_max bottom_max right_max max_loc")
    for row in rows:
        mass_delta = row["liquid_mass"] - first["liquid_mass"]
        print(
            f"{row['step']:6d} "
            f"{row['time']:.6f} "
            f"{row['liquid_mass']:.8f} "
            f"{mass_delta:+.8f} "
            f"{row['negative_phi_area']:.8f} "
            f"{row['global_max_div']:.6f} "
            f"{row['interior_max_div']:.6f} "
            f"{row['left_max_div']:.6f} "
            f"{row['top_max_div']:.6f} "
            f"{row['bottom_max_div']:.6f} "
            f"{row['right_max_div']:.6f} "
            f"({row['max_div_i']},{row['max_div_j']})"
        )


if __name__ == "__main__":
    main()
