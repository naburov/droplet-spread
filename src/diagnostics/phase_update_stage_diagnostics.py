"""Within-step phase-update wall-row diagnostics (phi0 alternating mode)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def wall_alt_stats(name: str, phi, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    a = np.asarray(phi)
    phi0 = a[:, 0]
    phi1 = a[:, 1] if a.shape[1] > 1 else phi0
    idx = np.arange(a.shape[0], dtype=np.float64)
    alt = (-1.0) ** idx

    out: dict[str, Any] = {
        "stage": name,
        "phi0_alt": float(np.mean(phi0 * alt)),
        "phi1_alt": float(np.mean(phi1 * alt)),
        "phi0_abs_alt": float(abs(np.mean(phi0 * alt))),
        "phi1_abs_alt": float(abs(np.mean(phi1 * alt))),
        "phi0_tv": float(np.mean(np.abs(np.diff(phi0)))) if phi0.size > 1 else 0.0,
        "phi1_tv": float(np.mean(np.abs(np.diff(phi1)))) if phi1.size > 1 else 0.0,
        "phi0_min": float(np.min(phi0)),
        "phi0_max": float(np.max(phi0)),
        "phi1_min": float(np.min(phi1)),
        "phi1_max": float(np.max(phi1)),
    }
    if extra:
        out.update(extra)
    return out


def wall_alt_contact_stats(name: str, phi) -> dict[str, Any]:
    a = np.asarray(phi)
    phi0 = a[:, 0]
    phi1 = a[:, 1] if a.shape[1] > 1 else phi0
    idx = np.arange(a.shape[0], dtype=np.float64)
    alt = (-1.0) ** idx
    mask = (np.abs(phi0) < 0.7) | (np.abs(phi1) < 0.7)
    if not np.any(mask):
        return {}
    return {
        f"{name}_contact_cells": int(np.sum(mask)),
        f"{name}_phi0_contact_alt": float(np.mean(phi0[mask] * alt[mask])),
        f"{name}_phi1_contact_alt": float(np.mean(phi1[mask] * alt[mask])),
        f"{name}_phi0_contact_tv": float(np.mean(np.abs(np.diff(phi0[mask]))))
        if int(np.sum(mask)) > 1
        else 0.0,
    }


def diag_ghost_stats(name: str, bottom_ghost_phi, phi) -> dict[str, Any]:
    ghost_delta = np.asarray(bottom_ghost_phi) - np.asarray(phi)[:, 1]
    idx = np.arange(ghost_delta.shape[0], dtype=np.float64)
    alt = (-1.0) ** idx
    return {
        "stage": name,
        "ghost_delta_alt": float(np.mean(ghost_delta * alt)),
        "ghost_delta_max": float(np.max(np.abs(ghost_delta))),
        "ghost_delta_mean": float(np.mean(np.abs(ghost_delta))),
    }


def append_phase_stage_rows(
    phase_solver,
    step: int,
    t: float,
    output_dir: str,
    *,
    csv_name: str = "phase_update_stage_diagnostics.csv",
    jsonl_name: str = "phase_update_stage_diagnostics.jsonl",
) -> None:
    rows = getattr(phase_solver, "_phase_stage_rows", None)
    if not rows:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamped = [{"step": int(step), "t": float(t), **row} for row in rows]

    jsonl_path = out_dir / jsonl_name
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in stamped:
            handle.write(json.dumps(row) + "\n")

    csv_path = out_dir / csv_name
    fieldnames: list[str] = []
    for row in stamped:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(stamped)

    phase_solver._phase_stage_rows = []
