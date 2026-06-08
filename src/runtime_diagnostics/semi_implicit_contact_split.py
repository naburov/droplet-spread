"""Helpers for semi-implicit CH contact-delta experiments (temporary stabilization)."""

from __future__ import annotations

import numpy as np

VALID_SPLITS = frozenset(
    {
        "explicit_delta",
        "no_delta",
        "filtered_delta",
        "damped_delta",
        "implicit_wall_energy",
        "explicit_ghost",
    }
)


def normalize_split_mode(mode: str | None) -> str:
    if mode is None:
        return "explicit_delta"
    mode = str(mode)
    if mode not in VALID_SPLITS:
        raise ValueError(
            f"Unknown semi_implicit_contact_split={mode!r}. "
            f"Allowed: {sorted(VALID_SPLITS)}"
        )
    return mode


def lowpass_x_column(col: np.ndarray, passes: int = 1) -> np.ndarray:
    """1D x-direction low-pass with non-periodic edge averages."""
    a = np.asarray(col, dtype=np.float64).copy()
    for _ in range(max(int(passes), 0)):
        b = a.copy()
        if a.size >= 3:
            b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
        if a.size >= 2:
            b[0] = 0.5 * (a[0] + a[1])
            b[-1] = 0.5 * (a[-2] + a[-1])
        a = b
    return a


def lowpass_x_bottom_strip(field: np.ndarray, strip_rows: int = 3, passes: int = 1) -> np.ndarray:
    """Smooth contact_delta only near the bottom (first strip_rows in y)."""
    out = np.asarray(field, dtype=np.float64).copy()
    n_strip = min(int(strip_rows), out.shape[1])
    for j in range(n_strip):
        out[:, j] = lowpass_x_column(out[:, j], passes=passes)
    return out


def damp_bottom_rows(field: np.ndarray, beta: float, n_rows: int = 2) -> np.ndarray:
    out = np.asarray(field, dtype=np.float64).copy()
    beta = float(beta)
    n_rows = min(int(n_rows), out.shape[1])
    for j in range(n_rows):
        out[:, j] *= beta
    return out
