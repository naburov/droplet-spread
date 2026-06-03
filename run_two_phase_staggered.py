"""
Run the two-phase staggered (MAC) flow debug simulation.

Usage:
  python3 run_two_phase_staggered.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from simulation.two_phase_staggered import TwoPhaseStaggeredConfig, TwoPhaseStaggeredSimulation


def main():
    cfg = TwoPhaseStaggeredConfig()
    sim = TwoPhaseStaggeredSimulation(cfg)
    sim.run(log_every=10)


if __name__ == "__main__":
    main()

