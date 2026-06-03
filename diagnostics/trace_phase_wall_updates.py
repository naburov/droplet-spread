#!/usr/bin/env python3
"""
Trace near-wall phase updates between consecutive checkpoints.

Focuses on rows y=0..y_max in the x-zone around the interface/contact lines:
  - per-step/per-row summary metrics for phi and dphi
  - top-|dphi| cells (local spikes) for forensic inspection
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--y-max", type=int, default=3)
    ap.add_argument("--phi-band", type=float, default=0.8)
    ap.add_argument("--contact-pad", type=int, default=6)
    ap.add_argument("--top-cells", type=int, default=12)
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument("--out-cells", type=Path, default=None)
    return ap.parse_args()


def _find_main_interval(phi_wall: np.ndarray) -> Tuple[int, int]:
    neg = phi_wall < 0.0
    intervals = []
    i = 0
    n = phi_wall.size
    while i < n:
        if not neg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and neg[j + 1]:
            j += 1
        intervals.append((i, j))
        i = j + 1
    if not intervals:
        return -1, -1
    mid = n // 2
    center = [iv for iv in intervals if iv[0] <= mid <= iv[1]]
    if center:
        return center[0]
    return max(intervals, key=lambda iv: iv[1] - iv[0])


def _zero_crossings(phi_wall: np.ndarray, dx: float) -> Tuple[float, float]:
    i0, i1 = _find_main_interval(phi_wall)
    if i0 < 0:
        return float("nan"), float("nan")

    if i0 > 0:
        a, b = phi_wall[i0 - 1], phi_wall[i0]
        t = abs(a) / (abs(a) + abs(b) + 1e-30)
        x_left = (i0 - 1 + t + 0.5) * dx
    else:
        x_left = (i0 + 0.5) * dx

    if i1 < phi_wall.size - 1:
        a, b = phi_wall[i1], phi_wall[i1 + 1]
        t = abs(a) / (abs(a) + abs(b) + 1e-30)
        x_right = (i1 + t + 0.5) * dx
    else:
        x_right = (i1 + 0.5) * dx

    return float(x_left), float(x_right)


def _safe_rough(row: np.ndarray) -> float:
    if row.size < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(row, n=2))))


def _iter_selected(checkpoints: Iterable[Path], step_min: int | None, step_max: int | None):
    for ck in checkpoints:
        step = int(ck.stem.split("_")[-1])
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        yield step, ck


def main() -> None:
    args = parse_args()
    exp = args.experiment_dir

    cfg = json.loads((exp / "simulation_parameters.json").read_text())
    nx = int(cfg["grid_params"]["Nx"])
    ny = int(cfg["grid_params"]["Ny"])
    dx = float(cfg["grid_params"]["Lx"]) / nx

    y_max = max(0, min(int(args.y_max), ny - 1))
    phi_band = float(args.phi_band)
    contact_pad = max(int(args.contact_pad), 0)
    top_cells = max(int(args.top_cells), 0)

    ck_paths = sorted((exp / "checkpoints").glob("checkpoint_*.npz"))
    selected = list(_iter_selected(ck_paths, args.step_min, args.step_max))
    if len(selected) < 2:
        raise SystemExit("Need at least two checkpoints in selected range")

    parity = ((-1.0) ** np.arange(nx)).astype(np.float64)
    summary_rows = []
    cell_rows = []

    eps = 1e-14
    for (s0, ck0), (s1, ck1) in zip(selected[:-1], selected[1:]):
        d0 = np.load(ck0, allow_pickle=True)
        d1 = np.load(ck1, allow_pickle=True)
        phi0 = np.asarray(d0["phi"], dtype=np.float64)
        phi1 = np.asarray(d1["phi"], dtype=np.float64)

        t0 = float(d0.get("t", np.nan))
        t1 = float(d1.get("t", np.nan))
        dt = t1 - t0 if np.isfinite(t0) and np.isfinite(t1) else np.nan

        iL, iR = _find_main_interval(phi0[:, 0])
        xL, xR = _zero_crossings(phi0[:, 0], dx)

        iface_x = np.any(np.abs(phi0[:, : y_max + 1]) < phi_band, axis=1)
        contact_x = np.zeros(nx, dtype=bool)
        if iL >= 0:
            l0 = max(0, iL - contact_pad)
            l1 = min(nx, iL + contact_pad + 1)
            r0 = max(0, iR - contact_pad)
            r1 = min(nx, iR + contact_pad + 1)
            contact_x[l0:l1] = True
            contact_x[r0:r1] = True
        xmask = iface_x | contact_x
        if not np.any(xmask):
            xmask[:] = True

        idx = np.flatnonzero(xmask)
        x0 = int(idx[0])
        x1 = int(idx[-1])

        dphi = phi1 - phi0
        for y in range(y_max + 1):
            row0 = phi0[:, y]
            drow = dphi[:, y]

            row0_sel = row0[xmask]
            drow_sel = drow[xmask]
            abs_d_mean = float(np.mean(np.abs(drow_sel)))
            abs_phi_mean = float(np.mean(np.abs(row0_sel)))

            checker_dphi = float(abs(np.mean(drow_sel * parity[xmask])) / (abs_d_mean + eps))
            checker_phi = float(abs(np.mean(row0_sel * parity[xmask])) / (abs_phi_mean + eps))

            seg0 = row0[x0 : x1 + 1]
            segd = drow[x0 : x1 + 1]
            sign_changes = int(np.sum(seg0[:-1] * seg0[1:] < 0.0)) if seg0.size > 1 else 0

            summary_rows.append(
                {
                    "step_prev": s0,
                    "step_next": s1,
                    "dt": dt,
                    "y": y,
                    "x_left": xL,
                    "x_right": xR,
                    "i_left": iL,
                    "i_right": iR,
                    "x_window_i0": x0,
                    "x_window_i1": x1,
                    "window_cells": int(idx.size),
                    "mean_phi": float(np.mean(row0_sel)),
                    "mean_abs_phi": abs_phi_mean,
                    "mean_dphi": float(np.mean(drow_sel)),
                    "mean_abs_dphi": abs_d_mean,
                    "max_abs_dphi": float(np.max(np.abs(drow_sel))),
                    "checker_phi": checker_phi,
                    "checker_dphi": checker_dphi,
                    "rough_phi": _safe_rough(seg0),
                    "rough_dphi": _safe_rough(segd),
                    "sign_changes_phi": sign_changes,
                }
            )

            if top_cells > 0:
                order = np.argsort(np.abs(drow_sel))
                top_local = order[-top_cells:]
                for j in top_local[::-1]:
                    i = int(idx[j])
                    cell_rows.append(
                        {
                            "step_prev": s0,
                            "step_next": s1,
                            "dt": dt,
                            "y": y,
                            "i": i,
                            "x": (i + 0.5) * dx,
                            "phi_prev": float(phi0[i, y]),
                            "phi_next": float(phi1[i, y]),
                            "dphi": float(dphi[i, y]),
                            "abs_dphi": float(abs(dphi[i, y])),
                            "in_interface_x": int(iface_x[i]),
                            "in_contact_pad": int(contact_x[i]),
                        }
                    )

    out_summary = args.out_summary or (exp / "phase_wall_update_trace_summary.csv")
    out_cells = args.out_cells or (exp / "phase_wall_update_trace_topcells.csv")

    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    if cell_rows:
        with out_cells.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
            w.writeheader()
            w.writerows(cell_rows)

    print(f"saved_summary {out_summary}")
    print(f"saved_cells {out_cells}")
    print(f"rows_summary {len(summary_rows)}")
    print(f"rows_cells {len(cell_rows)}")


if __name__ == "__main__":
    main()

