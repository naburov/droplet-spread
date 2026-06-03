#!/usr/bin/env python3
"""
Строит один рисунок с несколькими контурами phi=0 в разные моменты времени.
Ось y по умолчанию [0, 0.5]. Чекпоинты и параметры берут из каталога эксперимента.
Запуск из корня проекта с PYTHONPATH=src.
"""

import argparse
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization.checkpointing import load_checkpoint


def load_params(exp_dir):
    path = os.path.join(exp_dir, "simulation_parameters.json")
    with open(path) as f:
        return json.load(f)


def get_grid(params):
    gp = params["grid_params"]
    Lx, Ly = gp["Lx"], gp["Ly"]
    Nx, Ny = gp["Nx"], gp["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    return x, y, Lx, Ly


# Максимум по y (высота капли ~0.3)
Y_MAX = 0.5

# Цвета контуров по умолчанию
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def plot_three_contours(exp_dir, steps=None, out_path=None, colors=None, first_third_only=True,
                       xlim=None, ylim=None):
    """
    Строит контуры phi=0 по чекпоинтам. exp_dir — каталог эксперимента (checkpoints + simulation_parameters.json).
    steps — номера шагов; если None, берутся из первой трети. xlim, ylim — пределы осей (xmin, xmax), (ymin, ymax).
    """
    params = load_params(exp_dir)
    x, y, Lx, Ly = get_grid(params)
    dt = params["time_params"]["dt"]
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")

    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".npz") and f.startswith("checkpoint_"))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 checkpoints, found {len(files)} in {ckpt_dir}")

    def step_from_name(name):
        return int(name.replace("checkpoint_", "").replace(".npz", ""))

    steps_available = [step_from_name(f) for f in files]
    if steps is None:
        n = len(steps_available)
        if first_third_only:
            third = max(1, n // 3)
            pool = steps_available[:third]
        else:
            pool = steps_available
        # 3–5 шагов из первой трети, равномерно
        n_contours = min(5, max(3, len(pool)))
        indices = np.unique(np.linspace(0, len(pool) - 1, n_contours, dtype=int))
        steps = [pool[i] for i in indices]
    else:
        for s in steps:
            if s not in steps_available:
                raise ValueError(f"Step {s} not in checkpoints; available: {steps_available[:5]}...")

    if colors is None:
        colors = DEFAULT_COLORS

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    legend_handles = []
    n_steps = len(steps)
    for k, step in enumerate(steps):
        fname = f"checkpoint_{step:06d}.npz"
        path = os.path.join(ckpt_dir, fname)
        data = load_checkpoint(path)
        phi = np.asarray(data["phi"])
        # phi (Nx, Ny); для contour нужен Z в форме (Ny, Nx)
        Z = phi.T
        t = step * dt
        is_first_or_last = k == 0 or k == n_steps - 1
        linestyle = "-" if is_first_or_last else "--"
        ax.contour(X, Y, Z, levels=[0.0], colors=[colors[k % len(colors)]], linewidths=2, linestyles=linestyle)
        legend_handles.append(Line2D([0], [0], color=colors[k % len(colors)], lw=2, ls=linestyle, label=f"t = {t:.4f}"))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(0, Lx)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {out_path}")
    else:
        out_path = os.path.join(exp_dir, "droplet_slide_three_contours.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {out_path}")
    plt.close(fig)
    return out_path


def main():
    default_exp = os.path.join(os.path.dirname(PROJECT_ROOT), "good_exps", "droplet_slide")
    parser = argparse.ArgumentParser(description="Контуры φ=0 по чекпоинтам, y ∈ [0, 0.5] по умолчанию.")
    parser.add_argument("--exp", type=str, default=default_exp, help="Каталог эксперимента")
    parser.add_argument("--steps", type=int, nargs="+", metavar="S", help="Номера шагов чекпоинтов")
    parser.add_argument("--out", type=str, help="Путь к сохраняемому рисунку")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="Пределы по x")
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="Пределы по y")
    parser.add_argument("--single-color", action="store_true", help="Один цвет для всех контуров")
    args = parser.parse_args()

    colors = ["#1f77b4"] if args.single_color else None
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None
    plot_three_contours(args.exp, steps=args.steps, out_path=args.out, colors=colors, xlim=xlim, ylim=ylim)


if __name__ == "__main__":
    main()
