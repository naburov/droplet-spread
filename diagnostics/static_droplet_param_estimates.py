#!/usr/bin/env python3
"""
Quick parameter estimator for static-droplet setups.

This script does not run CFD. It extracts key nondimensional groups from config
and reports practical ranges for "collect/not spread" behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float(dct, key, default):
    try:
        return float(dct.get(key, default))
    except Exception:
        return float(default)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument(
        "--we-candidates",
        nargs="*",
        type=float,
        default=[0.005, 0.002, 0.001, 5e-4],
        help="Candidate Weber values to evaluate",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    pp = cfg.get("physical_params", {})
    gp = cfg.get("grid_params", {})
    ic = cfg.get("initial_conditions", {})
    bc_vel = cfg.get("boundary_conditions", {}).get("velocity", {})

    we1 = _safe_float(pp, "We1", 1.0)
    we2 = _safe_float(pp, "We2", we1)
    re1 = _safe_float(pp, "Re1", 1.0)
    re2 = _safe_float(pp, "Re2", re1)
    fr = _safe_float(pp, "Fr", 1.0)
    theta = _safe_float(pp, "contact_angle", 90.0)

    lx = _safe_float(gp, "Lx", 1.0)
    ly = _safe_float(gp, "Ly", 1.0)
    nx = int(gp.get("Nx", 1))
    ny = int(gp.get("Ny", 1))
    dx = lx / max(nx, 1)
    dy = ly / max(ny, 1)

    r = _safe_float(ic, "droplet_radius", 0.1)
    slip = _safe_float(bc_vel, "slip_length", 0.0)
    slip_l = _safe_float(bc_vel, "slip_length_liquid", slip)
    slip_g = _safe_float(bc_vel, "slip_length_gas", slip)

    # Common nondimensional proxies (based on standard definitions).
    # Bo ~ We/Fr^2; Ca ~ We/Re.
    bo1 = we1 / max(fr * fr, 1e-30)
    bo2 = we2 / max(fr * fr, 1e-30)
    ca1 = we1 / max(re1, 1e-30)
    ca2 = we2 / max(re2, 1e-30)
    oh1 = (we1 ** 0.5) / max(re1, 1e-30)
    oh2 = (we2 ** 0.5) / max(re2, 1e-30)

    slip_over_r_l = slip_l / max(r, 1e-30)
    slip_over_r_g = slip_g / max(r, 1e-30)

    print(f"Config: {cfg_path}")
    print()
    print("Base setup:")
    print(f"  We1={_fmt(we1)} We2={_fmt(we2)}  Re1={_fmt(re1)} Re2={_fmt(re2)}  Fr={_fmt(fr)}")
    print(f"  contact_angle={_fmt(theta)} deg")
    print(f"  grid: Nx={nx} Ny={ny} dx={_fmt(dx)} dy={_fmt(dy)}")
    print(f"  droplet_radius={_fmt(r)} (R/Lx={_fmt(r/max(lx,1e-30))})")
    print(f"  slip_length_liquid={_fmt(slip_l)} slip_length_gas={_fmt(slip_g)}")
    print()
    print("Derived group proxies:")
    print(f"  Bo_liquid ~= We1/Fr^2 = {_fmt(bo1)}")
    print(f"  Bo_gas    ~= We2/Fr^2 = {_fmt(bo2)}")
    print(f"  Ca_liquid ~= We1/Re1  = {_fmt(ca1)}")
    print(f"  Ca_dense  ~= We2/Re2  = {_fmt(ca2)}")
    print(f"  Oh_liquid ~= sqrt(We1)/Re1 = {_fmt(oh1)}")
    print(f"  Oh_dense  ~= sqrt(We2)/Re2 = {_fmt(oh2)}")
    print(f"  slip/R liquid={_fmt(slip_over_r_l)} gas={_fmt(slip_over_r_g)}")
    print()
    print("Heuristic interpretation:")
    if bo1 < 1e-3:
        print("  - Bond proxy is very low: capillary should dominate gravity strongly.")
    elif bo1 < 1e-2:
        print("  - Bond proxy is low: capillary-dominant, but wall/slip dynamics matter.")
    else:
        print("  - Bond proxy is moderate/high: gravity can noticeably spread droplet.")
    if slip_over_r_l > 0.05:
        print("  - slip/R is large enough to promote lateral drift/spreading at contact line.")
    elif slip_over_r_l > 0.01:
        print("  - slip/R is moderate: can still affect static footprint noticeably.")
    else:
        print("  - slip/R is small: less likely to drive strong lateral spreading by itself.")
    print()

    print("We candidate sweep:")
    print("  We      Bo=We/Fr^2    Ca=We/Re2")
    for we in args.we_candidates:
        bo = we / max(fr * fr, 1e-30)
        ca = we / max(re2, 1e-30)
        print(f"  {_fmt(we):>7}  {_fmt(bo):>11}  {_fmt(ca):>10}")

    print()
    we_bo_1e3 = (fr * fr) * 1e-3
    we_bo_5e4 = (fr * fr) * 5e-4
    print("Suggested targets for 'collect/not spread' tests:")
    print(f"  - Bo<=1e-3 target: We <= {_fmt(we_bo_1e3)}")
    print(f"  - Bo<=5e-4 target: We <= {_fmt(we_bo_5e4)}")
    print("  - Keep contact angle fixed first; sweep We, then reduce slip_length_liquid if needed.")


if __name__ == "__main__":
    main()

