"""In-step ghost-row diagnostics (captured when bottom ghost φ is built in CH step)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def compute_ghost_row_instep_diagnostic(bottom_ghost_phi, phi) -> dict[str, Any]:
    """Alternating-mode metrics for bottom ghost row vs φ[:,1] (solver path)."""
    ghost_delta = np.asarray(bottom_ghost_phi) - np.asarray(phi)[:, 1]
    phi0 = np.asarray(phi)[:, 0]
    phi1 = np.asarray(phi)[:, 1]
    idx = np.arange(ghost_delta.shape[0], dtype=np.int64)
    alt = (-1.0) ** idx.astype(np.float64)

    diag: dict[str, Any] = {
        "ghost_delta_max": float(np.max(np.abs(ghost_delta))),
        "ghost_delta_mean": float(np.mean(np.abs(ghost_delta))),
        "ghost_delta_alt": float(np.mean(ghost_delta * alt)),
        "phi0_alt": float(np.mean(phi0 * alt)),
        "phi1_alt": float(np.mean(phi1 * alt)),
    }

    mask = (np.abs(phi0) < 0.7) | (np.abs(phi1) < 0.7)
    if np.any(mask):
        diag["ghost_delta_contact_max"] = float(np.max(np.abs(ghost_delta[mask])))
        diag["ghost_delta_contact_alt"] = float(np.mean(ghost_delta[mask] * alt[mask]))
        diag["contact_cells"] = int(np.sum(mask))

    return diag


def record_bottom_ghost_instep(phase_solver, bottom_ghost_phi, phi) -> None:
    if not getattr(phase_solver, "record_ghost_row_instep", False):
        return
    phase_solver._last_ghost_row_instep = compute_ghost_row_instep_diagnostic(
        bottom_ghost_phi, phi
    )


def append_ghost_row_instep_csv(
    phase_solver,
    step: int,
    t: float,
    output_dir: str,
    *,
    filename: str = "ghost_row_instep_diagnostics.csv",
) -> None:
    pending = getattr(phase_solver, "_last_ghost_row_instep", None)
    if pending is None:
        return

    row = {"step": int(step), "t": float(t), **pending}
    path = Path(output_dir) / filename
    write_header = not path.is_file()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
