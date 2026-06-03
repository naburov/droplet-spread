"""Matplotlib joint plots. Same layout/API as PyVista; extent [0,Lx,0,Ly]."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from visualization.plot_layout import get_joint_plot_layout, prepare_joint_plot_data
from visualization.plot_layout import (
    KIND_PHASE, KIND_SURFACE_TENSION, KIND_VELOCITY_STREAMLINES, KIND_VELOCITY_MAGNITUDE,
    KIND_PRESSURE, KIND_DENSITY, KIND_ICE_PHASE, KIND_TEMPERATURE, KIND_COMBINED_PHASE, KIND_TEMPERATURE_INTERFACES,
)

# Per-kind: (data key for imshow, vmin key or value, vmax key or value, contour_phi, contour_psi, contour_rho_mid, streamplot, legend)
_KIND_OPTS = {
    KIND_PHASE: ("phi", -1, 1, True, True, False, False, "phase"),
    KIND_SURFACE_TENSION: ("ST_magnitude", 0, "st_vmax", True, True, False, False, None),
    KIND_VELOCITY_STREAMLINES: ("U_magnitude_masked", None, None, True, True, False, True, None),
    KIND_VELOCITY_MAGNITUDE: ("U_magnitude_masked", None, None, True, True, False, False, None),
    KIND_PRESSURE: ("P_vis", None, None, True, True, False, False, None),  # we plot -P_vis
    KIND_DENSITY: ("rho", None, None, True, True, True, False, None),  # contour_rho_mid when not ice
    KIND_ICE_PHASE: ("psi", -1, 1, True, True, False, False, "ice"),
    KIND_TEMPERATURE: ("T", "T_min", "T_max", True, True, False, False, None),
    KIND_COMBINED_PHASE: ("combined", -1, 1, True, True, False, False, "combined"),
    KIND_TEMPERATURE_INTERFACES: ("T", "T_min", "T_max", True, True, False, False, None),
}


def _draw_subplot(ax, data, spec):
    kind = spec["kind"]
    opts = _KIND_OPTS[kind]
    key, vmin_spec, vmax_spec, c_phi, c_psi, c_rho_mid, do_stream, legend_type = opts
    extent, X, Eta = data["extent"], data["X"], data["Eta"]

    Z = data[key]
    if kind == KIND_PRESSURE:
        Z = -Z
    vmin = data[vmin_spec] if isinstance(vmin_spec, str) else vmin_spec
    vmax = data[vmax_spec] if isinstance(vmax_spec, str) else vmax_spec
    kw = dict(origin="lower", extent=extent, cmap=spec["cmap"], aspect="auto")
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(Z.T, **kw)

    if c_phi and not (kind == KIND_DENSITY and not data["include_ice_water"]):
        ax.contour(X, Eta, data["phi"].T, levels=[0], colors="r", linewidths=2, linestyles="-" if kind != KIND_ICE_PHASE else ":")
    if c_psi and data["include_ice_water"] and data["psi"] is not None:
        ax.contour(X, Eta, data["psi"].T, levels=[0], colors="b", linewidths=2, linestyles="--")
    if c_rho_mid and not (data["include_ice_water"] and data["psi"] is not None):
        ax.contour(X, Eta, data["rho"].T, levels=[(data["rho1"] + data["rho2"]) / 2.0], colors="k", linewidths=2)

    if do_stream:
        ax.streamplot(data["x"], data["eta"], data["U_masked"][..., 0].T, data["U_masked"][..., 1].T, density=1.5, color="white")

    if legend_type == "phase":
        ax.legend(handles=[Line2D([0], [0], color="r", lw=2, label="Water-Air"), Line2D([0], [0], color="b", lw=2, linestyle="--", label="Water-Ice")], loc="upper right", fontsize=8)
    elif legend_type == "ice":
        legs = [Line2D([0], [0], color="b", lw=2, label="Water-Ice")]
        if np.any(data["phi"] < 0):
            legs.append(Line2D([0], [0], color="r", lw=2, linestyle=":", label="Water-Air"))
        ax.legend(handles=legs, loc="upper right", fontsize=8)
    elif legend_type == "combined":
        ax.legend(handles=[Line2D([0], [0], color="r", lw=2, label="Water-Air"), Line2D([0], [0], color="b", lw=2, linestyle="--", label="Water-Ice")], loc="upper right", fontsize=8)

    plt.colorbar(im, ax=ax, label=spec["cbar_label"])
    ax.set_title(spec["title"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def create_joint_plot(phi, U, P, surface_tension, dt, step, dx, dy, mass, rho1, rho2, save_path=None,
                      psi=None, T=None, include_ice_water=False, Lx=1.0, Ly=1.0, geometry=None):
    """Joint plot: same layout and API as create_joint_plot_pyvista_full."""
    data = prepare_joint_plot_data(phi, U, P, surface_tension, rho1, rho2, Lx, Ly, psi=psi, T=T, include_ice_water=include_ice_water)
    nrows, ncols, specs = get_joint_plot_layout(include_ice_water)
    fig = plt.figure(figsize=(24, 16) if include_ice_water else (18, 12))
    try:
        gs = GridSpec(nrows, ncols, figure=fig)
        for spec in specs:
            ax = fig.add_subplot(gs[spec["row"], spec["col"]])
            _draw_subplot(ax, data, spec)
        fig.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.figtext(0.5, 0.02, f"Time Step: {step} | Simulation Time: {step * dt:.5f}", ha="center", fontsize=10)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)
