#!/usr/bin/env python3
"""
Совместная визуализация двух экспериментов: два графика слева и справа,
эволюция формы капли по ~5 моментам времени, контуры разными цветами.
Сохраняет рисунок в WeakLimit/figures/droplet_contours.pdf.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
_ckpt_path = os.path.join(PROJECT_ROOT, "src", "visualization", "checkpointing.py")
_spec = importlib.util.spec_from_file_location("checkpointing", _ckpt_path)
_ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ckpt)
load_checkpoint = _ckpt.load_checkpoint

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
# Distinct patterns for journal B&W (solid / dashed / dotted / dash-dot / long dash)
LINE_STYLES = [
    "solid",
    (0, (5, 2)),
    (0, (1, 1.5)),
    (0, (3, 1.5, 1, 1.5)),
    (0, (6, 2, 1.5, 2)),
]
XLIM = (0.3, 0.7)
YLIM = (0.0, 0.3)
N_CONTOURS = 5


def load_params(exp_dir):
    path = os.path.join(exp_dir, "simulation_parameters.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_grid(params):
    gp = params["grid_params"]
    Lx, Ly = gp["Lx"], gp["Ly"]
    Nx, Ny = gp["Nx"], gp["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    return x, y, Lx, Ly


def plot_one_panel(ax, exp_dir, title, steps=None, *, bw=False):
    params = load_params(exp_dir)
    x, y, Lx, Ly = get_grid(params)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Нет каталога чекпоинтов: {ckpt_dir}")
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".npz") and f.startswith("checkpoint_"))
    if len(files) < 2:
        raise ValueError(f"Мало чекпоинтов в {ckpt_dir}")

    def step_from_name(name):
        return int(name.replace("checkpoint_", "").replace(".npz", ""))

    steps_available = [step_from_name(f) for f in files]
    if steps is None:
        n = len(steps_available)
        indices = np.linspace(0, n - 1, N_CONTOURS, dtype=int)
        indices = np.unique(np.clip(indices, 0, n - 1))
        steps = [steps_available[i] for i in indices]

    X, Y = np.meshgrid(x, y)
    legend_handles = []
    for k, step in enumerate(steps):
        fname = f"checkpoint_{step:06d}.npz"
        path = os.path.join(ckpt_dir, fname)
        data = load_checkpoint(path)
        phi = np.asarray(data["phi"])
        Z = phi.T
        if bw:
            c = "black"
            ls = LINE_STYLES[k % len(LINE_STYLES)]
            ax.contour(
                X, Y, Z, levels=[0.0], colors=[c], linewidths=2.2, linestyles=[ls]
            )
            legend_handles.append(
                Line2D([0], [0], color=c, lw=2, linestyle=ls, label=f"step: {step}")
            )
        else:
            c = COLORS[k % len(COLORS)]
            ax.contour(X, Y, Z, levels=[0.0], colors=[c], linewidths=2)
            legend_handles.append(Line2D([0], [0], color=c, lw=2, label=f"step: {step}"))

    ax.set_xlim(XLIM[0], XLIM[1])
    ax.set_ylim(YLIM[0], YLIM[1])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=12)


def main():
    parser = argparse.ArgumentParser(description="Два эксперимента: эксперимент 1 и эксперимент 2.")
    parser.add_argument("--exp1", type=str, help="Каталог эксперимента 1")
    parser.add_argument("--exp2", type=str, help="Каталог эксперимента 2")
    parser.add_argument("--out", type=str, help="Путь к PDF")
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Чёрно-белый вариант (те же контуры, разные штрихи вместо цветов)",
    )
    args = parser.parse_args()
    if not args.exp1 or not args.exp2:
        parser.error("Нужны --exp1 и --exp2")
    out = args.out
    if not out:
        out = os.path.join(PROJECT_ROOT, "..", "WeakLimit", "figures", "droplet_contours.pdf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_one_panel(ax1, args.exp1, "Experiment 1", bw=args.bw)
    plot_one_panel(ax2, args.exp2, "Experiment 2", bw=args.bw)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
