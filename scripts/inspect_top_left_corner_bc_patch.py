#!/usr/bin/env python3
"""
Inspect top-left corner patch (3x3 cells) on staggered grid and BC application effects.

Focus:
  - first interior row near top boundary
  - top-left corner compatibility for mixed BCs (top dirichlet + left slip_symmetry)
  - optional top-strip compatibility modifications
"""

import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.state import SimulationState
from numerics.staggered_mac import divergence as mac_divergence


def _fmt(arr: np.ndarray, name: str) -> str:
    return f"{name} shape={arr.shape}\n{np.array2string(arr, precision=6, suppress_small=False)}"


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect top-left BC patch on staggered grid")
    p.add_argument("--config", required=True, help="Path to simulation config JSON")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint npz")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    state = SimulationState.from_config(cfg, restart_from=args.checkpoint)
    if state.u_face is None or state.v_face is None:
        raise RuntimeError("Checkpoint does not contain u_face/v_face")

    u_pre = jnp.array(state.u_face)
    v_pre = jnp.array(state.v_face)
    phi = state.phi
    dx, dy = float(state.dx), float(state.dy)
    Nx = u_pre.shape[0] - 1
    Ny = u_pre.shape[1]

    # Apply BCs exactly as simulation does on staggered faces.
    u_post, v_post = state.velocity_bc.apply_to_faces(
        u_pre,
        v_pre,
        dx,
        dy,
        psi=None,
        geometry=state.geometry,
        phi=phi,
    )

    u_pre_np = np.array(u_pre)
    v_pre_np = np.array(v_pre)
    u_post_np = np.array(u_post)
    v_post_np = np.array(v_post)

    # Top-left 3x3 cell patch => i=0..2, j=Ny-3..Ny-1.
    i0, i1 = 0, 2
    j0, j1 = Ny - 3, Ny - 1

    # Needed faces for those cells:
    # u faces: i=0..3, j=Ny-3..Ny-1
    # v faces: i=0..2, j=Ny-3..Ny
    u_slice = (slice(i0, i1 + 2), slice(j0, j1 + 1))
    v_slice = (slice(i0, i1 + 1), slice(j0, j1 + 2))

    print("=== Configuration BCs ===")
    vcfg = cfg.get("boundary_conditions", {}).get("velocity", {})
    print("top:", vcfg.get("top"), "left:", vcfg.get("left"), "right:", vcfg.get("right"), "bottom:", vcfg.get("bottom"))
    print("strip_compatibility_top:", vcfg.get("strip_compatibility_top", True))
    print("strip_compatibility_width:", vcfg.get("strip_compatibility_width", 2))
    print("top dirichlet values:", vcfg.get("dirichlet_values", {}).get("top", {}))
    print()

    print("=== Top-left 3x3 cell patch indices ===")
    print(f"cells i={i0}..{i1}, j={j0}..{j1} (Ny={Ny})")
    print()

    print("=== u-face patch (before) ===")
    print(_fmt(u_pre_np[u_slice], "u_pre[i=0..3, j=Ny-3..Ny-1]"))
    print("=== u-face patch (after BC) ===")
    print(_fmt(u_post_np[u_slice], "u_post[i=0..3, j=Ny-3..Ny-1]"))
    print("=== u-face patch delta ===")
    print(_fmt(u_post_np[u_slice] - u_pre_np[u_slice], "du"))
    print()

    print("=== v-face patch (before) ===")
    print(_fmt(v_pre_np[v_slice], "v_pre[i=0..2, j=Ny-3..Ny]"))
    print("=== v-face patch (after BC) ===")
    print(_fmt(v_post_np[v_slice], "v_post[i=0..2, j=Ny-3..Ny]"))
    print("=== v-face patch delta ===")
    print(_fmt(v_post_np[v_slice] - v_pre_np[v_slice], "dv"))
    print()

    div_pre = np.array(mac_divergence(u_pre, v_pre, dx, dy))
    div_post = np.array(mac_divergence(u_post, v_post, dx, dy))
    div_patch = (slice(i0, i1 + 1), slice(j0, j1 + 1))
    print("=== Cell divergence in top-left 3x3 (before) ===")
    print(_fmt(div_pre[div_patch], "div_pre"))
    print("=== Cell divergence in top-left 3x3 (after BC) ===")
    print(_fmt(div_post[div_patch], "div_post"))
    print("=== Cell divergence delta ===")
    print(_fmt(div_post[div_patch] - div_pre[div_patch], "ddiv"))
    print()

    # Explicit corner checks relevant to suspected incompatibility.
    top_vals = vcfg.get("dirichlet_values", {}).get("top", {})
    u_top_target = float(top_vals.get("u", 0.0))
    v_top_target = float(top_vals.get("v", 0.0))
    strip_w = max(1, int(vcfg.get("strip_compatibility_width", 2)))
    strip_enabled = bool(vcfg.get("strip_compatibility_top", True))

    print("=== Corner/BC consistency checks (after BC) ===")
    print(f"u_post[0, Ny-1] (left+top corner) = {u_post_np[0, Ny-1]:+.6e} (top dirichlet target u={u_top_target:+.6e})")
    print(f"v_post[0, Ny]   (top boundary)      = {v_post_np[0, Ny]:+.6e} (top dirichlet target v={v_top_target:+.6e})")
    print(f"left slip check at top boundary: v_post[0, Ny] - v_post[1, Ny] = {(v_post_np[0, Ny] - v_post_np[1, Ny]):+.6e}")
    print(
        f"left slip check first interior-top row: "
        f"v_post[0, Ny-1] - v_post[1, Ny-1] = {(v_post_np[0, Ny-1] - v_post_np[1, Ny-1]):+.6e}"
    )
    if strip_enabled:
        print(f"top-strip enabled width={strip_w}; affected i=0..{min(strip_w-1, Nx-1)} at j=Ny-1")
        for i in range(min(strip_w, Nx)):
            print(f"  i={i:2d}: v_post[{i},Ny-1]={v_post_np[i,Ny-1]:+.6e}, u_post[{i},Ny-1]={u_post_np[i,Ny-1]:+.6e}, u_post[{i+1},Ny-1]={u_post_np[i+1,Ny-1]:+.6e}")
    else:
        print("top-strip compatibility disabled")


if __name__ == "__main__":
    main()
