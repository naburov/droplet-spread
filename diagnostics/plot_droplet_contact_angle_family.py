#!/usr/bin/env python3
"""
Plot idealized droplet cross-sections for multiple contact angles.

The curves are spherical-cap profiles with equal 2D cap area, so the shapes are
directly comparable as "same droplet volume" in cross-section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _cap_profile(theta_deg: float, area_target: float, n: int = 400):
    """Return parametric x,y profile for a spherical cap on y=0.

    Uses a circular-arc parameterization that correctly handles obtuse contact
    angles (major arc for theta > 90 deg).
    """
    theta = np.deg2rad(float(theta_deg))
    if not (1e-6 < theta < np.pi - 1e-6):
        raise ValueError(f"Unsupported angle {theta_deg}; must be in (0, 180).")

    # 2D cap area (cross-section): A = R^2 * (theta - sin(theta)cos(theta))
    area_factor = theta - np.sin(theta) * np.cos(theta)
    radius = np.sqrt(area_target / max(area_factor, 1e-30))

    y_center = -radius * np.cos(theta)

    # Right contact-point radius angle (from circle center), then sweep through
    # liquid arc by 2*theta. This gives minor arc for theta<90 and major arc for theta>90.
    t0 = 0.5 * np.pi - theta
    t1 = t0 + 2.0 * theta
    t = np.linspace(t0, t1, n)

    x = radius * np.cos(t)
    y = y_center + radius * np.sin(t)
    # Guard tiny round-off near wall.
    y = np.maximum(y, 0.0)

    x_max = float(np.max(np.abs(x)))
    return x, y, x_max, float(np.max(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--angles",
        nargs="*",
        type=float,
        default=[40, 60, 90, 120, 150],
        help="Contact angles in degrees",
    )
    ap.add_argument(
        "--area",
        type=float,
        default=1.0,
        help="Target cap area (same for all curves)",
    )
    ap.add_argument(
        "--output",
        default="diagnostics/contact_angle_family_40_60_90_120_150.png",
        help="Output image path",
    )
    args = ap.parse_args()

    angles = [float(a) for a in args.angles]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    x_lim = 0.0
    y_lim = 0.0
    for color, ang in zip(cmap, angles):
        x, y, half_width, h = _cap_profile(ang, area_target=float(args.area))
        ax.plot(x, y, color=color, linewidth=2.2, label=f"{ang:.0f} deg")
        x_lim = max(x_lim, half_width)
        y_lim = max(y_lim, h)

    # Substrate
    ax.axhline(0.0, color="black", linewidth=1.4, linestyle="--", alpha=0.8, label="substrate")
    ax.set_xlim(-1.05 * x_lim, 1.05 * x_lim)
    ax.set_ylim(-0.02 * y_lim, 1.08 * y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Spherical-cap droplets at different contact angles (equal area)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

