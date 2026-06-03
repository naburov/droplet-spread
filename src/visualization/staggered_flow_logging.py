"""
Logging/visualization for the standalone staggered (MAC) flow debug simulation.

Outputs:
  - per-step frames: pressure, velocity magnitude, streamlines
  - telemetry plots: div_max, max speed/CFL, kinetic energy, pressure range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StaggeredFlowLoggerConfig:
    out_dir: str
    save_every: int = 1
    dpi: int = 150
    streamline_density: float = 1.5
    max_frames: int | None = None  # safety cap


def _as_numpy(a):
    return np.asarray(a) if not isinstance(a, np.ndarray) else a


def faces_to_cell_center(u_face, v_face):
    """
    Convert MAC face velocities to cell-centered (u,v).
      u_face: (Nx+1, Ny)
      v_face: (Nx, Ny+1)
    Returns:
      uc, vc: (Nx, Ny)
    """
    u_face = _as_numpy(u_face)
    v_face = _as_numpy(v_face)
    uc = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    vc = 0.5 * (v_face[:, 1:] + v_face[:, :-1])
    return uc, vc


def save_frame(
    cfg: StaggeredFlowLoggerConfig,
    *,
    step: int,
    u_face,
    v_face,
    p_cell,
    phi_cell=None,
    Lx: float,
    Ly: float,
    dx: float,
    dy: float,
    title_extra: str = "",
):
    """
    Save a single debug frame with:
      - pressure (cell-centered)
      - speed magnitude (cell-centered)
      - streamlines of (u,v) (cell-centered)
    """
    if cfg.max_frames is not None and step // cfg.save_every >= cfg.max_frames:
        return
    if (step % cfg.save_every) != 0:
        return

    out_dir = Path(cfg.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    uc, vc = faces_to_cell_center(u_face, v_face)
    # Curvilinear: no-slip at surface (eta=0, row j=0). Zero so streamlines stop at surface.
    uc[:, 0] = 0.0
    vc[:, 0] = 0.0
    p = _as_numpy(p_cell)
    speed = np.sqrt(uc**2 + vc**2)
    phi = None if phi_cell is None else _as_numpy(phi_cell)

    Nx, Ny = p.shape
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Pressure
    im1 = ax1.imshow(
        p.T,
        origin="lower",
        extent=[0.0, Lx, 0.0, Ly],
        cmap="coolwarm",
        aspect="auto",
    )
    if phi is not None:
        ax1.contour(x, y, phi.T, levels=[0.0], colors="k", linewidths=1.0)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("pressure p")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Speed magnitude
    im2 = ax2.imshow(
        speed.T,
        origin="lower",
        extent=[0.0, Lx, 0.0, Ly],
        cmap="viridis",
        aspect="auto",
    )
    if phi is not None:
        ax2.contour(x, y, phi.T, levels=[0.0], colors="k", linewidths=1.0)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("|u|")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    # Streamlines over speed background
    im3 = ax3.imshow(
        speed.T,
        origin="lower",
        extent=[0.0, Lx, 0.0, Ly],
        cmap="viridis",
        aspect="auto",
    )
    if phi is not None:
        ax3.contour(x, y, phi.T, levels=[0.0], colors="k", linewidths=1.0)
    ax3.streamplot(
        x,
        y,
        uc.T,
        vc.T,
        density=cfg.streamline_density,
        color="white",
        linewidth=0.8,
    )
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("streamlines")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    fig.suptitle(f"step {step:06d}{(' — ' + title_extra) if title_extra else ''}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fname = frames_dir / f"frame_{step:06d}.png"
    fig.savefig(fname, dpi=cfg.dpi)
    plt.close(fig)


def save_telemetry(
    cfg: StaggeredFlowLoggerConfig,
    *,
    history: list[dict[str, Any]],
    dx: float,
    dy: float,
    dt: float,
):
    """Save a compact telemetry plot for the run."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    div_max = np.array([h.get("div_max", np.nan) for h in history], dtype=float)
    div_mean = np.array([h.get("div_mean", np.nan) for h in history], dtype=float)
    u_max = np.array([h.get("u_max", np.nan) for h in history], dtype=float)
    v_max = np.array([h.get("v_max", np.nan) for h in history], dtype=float)
    p_min = np.array([h.get("p_min", np.nan) for h in history], dtype=float)
    p_max = np.array([h.get("p_max", np.nan) for h in history], dtype=float)

    # crude CFL based on max component speed
    umax = np.maximum(np.abs(u_max), np.abs(np.array([h.get("u_min", np.nan) for h in history], dtype=float)))
    vmax = np.maximum(np.abs(v_max), np.abs(np.array([h.get("v_min", np.nan) for h in history], dtype=float)))
    cfl = dt * (umax / max(dx, 1e-30) + vmax / max(dy, 1e-30))

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.plot(div_max, label="div_max")
    ax1.plot(div_mean, label="div_mean")
    ax1.set_yscale("log")
    ax1.set_title("divergence")
    ax1.set_xlabel("step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(umax, label="|u|_max (component)")
    ax2.plot(vmax, label="|v|_max (component)")
    ax2.set_title("max velocity components")
    ax2.set_xlabel("step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(cfl, label="CFL (rough)")
    ax3.set_title("CFL estimate")
    ax3.set_xlabel("step")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(p_min, label="p_min")
    ax4.plot(p_max, label="p_max")
    ax4.set_title("pressure range")
    ax4.set_xlabel("step")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "telemetry.png", dpi=cfg.dpi)
    plt.close(fig)

