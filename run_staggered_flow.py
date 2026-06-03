"""
Run the standalone staggered-grid (MAC) flow debug simulation.

Usage:
  python3 run_staggered_flow.py
  python3 run_staggered_flow.py --config configs/staggered_poiseuille.json
"""

import sys
from pathlib import Path
import json
import argparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from simulation.staggered_flow import StaggeredFlowConfig, StaggeredFlowSimulation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    args = ap.parse_args()

    if args.config is None:
        cfg = StaggeredFlowConfig(
            Nx=32,
            Ny=32,
            Lx=1.0,
            Ly=1.0,
            dt=5e-4,
            nu=1e-2,
            steps=20,
            include_advection=True,
            inlet_profile="linear_to_half",
            u_inlet=1.0,
            top_bc="no_slip",
            bottom_bc="no_slip",
            outflow_right=True,
            save_plots=True,
            save_every=1,
        )
    else:
        cfg_path = Path(args.config)
        with open(cfg_path, "r") as f:
            d = json.load(f)
        cfg = StaggeredFlowConfig(**d)

    sim = StaggeredFlowSimulation(cfg)
    sim.run(log_every=1)


if __name__ == "__main__":
    main()

