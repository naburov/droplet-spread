"""
PyVista 2D field rendering. Grid warped (x, eta) -> (x, y) with y = eta + f(x) so VTK interpolates in physical (x,y).
view_xy() = x right, y up. indexing='xy'.
"""

import gc
import os
import sys
import traceback
import numpy as np

# Set to skip streamline computation (avoids VTK segfault in some envs). E.g. PYVISTA_SKIP_STREAMLINES=1
SKIP_STREAMLINES = os.environ.get("PYVISTA_SKIP_STREAMLINES", "").strip().lower() in ("1", "true", "yes")

# Log VTK call stack before each plot (to diagnose segfaults/bus errors). E.g. PYVISTA_LOG_VTK_STACK=1
_LOG_VTK_STACK = os.environ.get("PYVISTA_LOG_VTK_STACK", "").strip().lower() in ("1", "true", "yes")


def _log_vtk_stack(label: str, operation: str = ""):
    """Log current call stack before a VTK operation. When a crash occurs, the last log shows what was running."""
    if not _LOG_VTK_STACK:
        return
    parts = [f"\n[VTK stack] {label}"]
    if operation:
        parts.append(f" about to: {operation}")
    parts.append("\n")
    stack = "".join(traceback.format_stack())
    msg = "".join(parts) + stack
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception as e:
        print(e)


def _log_vtk_error(label: str, operation: str, exc: BaseException):
    """Log an exception that occurred during a VTK operation (so we can see why the error happened)."""
    try:
        sys.stderr.write(f"\n[VTK error] {label} during: {operation}\n")
        sys.stderr.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        sys.stderr.flush()
    except Exception:
        pass

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


# Teardown: ren_win.Finalize() and deep_clean() can segfault in some VTK/PyVista builds.
# We skip Finalize by default. Set PYVISTA_DO_FINALIZE=1 to call it; PYVISTA_SKIP_DEEP_CLEAN=1 to skip deep_clean.
_SKIP_FINALIZE = os.environ.get("PYVISTA_DO_FINALIZE", "").strip().lower() not in ("1", "true", "yes")
_SKIP_DEEP_CLEAN = os.environ.get("PYVISTA_SKIP_DEEP_CLEAN", "").strip().lower() in ("1", "true", "yes")


def _close_plotter(plotter):
    """Close plotter and release memory. Reduces leak when creating many plotters in a loop."""
    if plotter is None:
        return
    _log_vtk_stack("_close_plotter", "plotter.close / deep_clean")
    if not _SKIP_FINALIZE:
        try:
            if hasattr(plotter, "ren_win") and plotter.ren_win is not None:
                plotter.ren_win.Finalize()
        except Exception:
            pass
    try:
        plotter.close()
    except Exception:
        pass
    if not _SKIP_DEEP_CLEAN:
        try:
            if hasattr(plotter, "deep_clean"):
                plotter.deep_clean()
        except Exception:
            pass
    gc.collect()

DEFAULT_RENDER_SIZE = (800, 640)

# kind -> (data_key, vmin, vmax); vmin/vmax can be None or spec key like "st_vmax"
_FIELD_FOR_KIND = {
    "phase": ("phi", -1, 1),
    "surface_tension": ("ST_magnitude", 0, "st_vmax"),
    "velocity_streamlines": ("U_magnitude_masked", None, None),
    "velocity_magnitude": ("U_magnitude_masked", None, None),
    "pressure": ("P_vis", None, None),
    "density": ("rho", None, None),
    "ice_phase": ("psi", -1, 1),
    "temperature": ("T", "T_min", "T_max"),
    "temperature_interfaces": ("T", "T_min", "T_max"),
    "combined_phase": ("combined", -1, 1),
}


def _strip_alpha(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def cell_centered_to_point_2d(Z_cc):
    """(Nx, Ny) cell-centered -> (Nx+1, Ny+1) vertex. Interior = avg of 4 cells; edges/corners replicated."""
    Nx, Ny = Z_cc.shape
    Z_pt = np.zeros((Nx + 1, Ny + 1), dtype=Z_cc.dtype)
    Z_pt[1:-1, 1:-1] = (Z_cc[:-1, :-1] + Z_cc[1:, :-1] + Z_cc[:-1, 1:] + Z_cc[1:, 1:]) / 4.0
    Z_pt[0, 1:-1] = (Z_cc[0, :-1] + Z_cc[0, 1:]) / 2.0
    Z_pt[-1, 1:-1] = (Z_cc[-1, :-1] + Z_cc[-1, 1:]) / 2.0
    Z_pt[1:-1, 0] = (Z_cc[:-1, 0] + Z_cc[1:, 0]) / 2.0
    Z_pt[1:-1, -1] = (Z_cc[:-1, -1] + Z_cc[1:, -1]) / 2.0
    Z_pt[0, 0], Z_pt[-1, 0], Z_pt[0, -1], Z_pt[-1, -1] = Z_cc[0, 0], Z_cc[-1, 0], Z_cc[0, -1], Z_cc[-1, -1]
    return Z_pt


def build_structured_grid_2d(X_verts, Y_verts):
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista is not installed")
    return pv.StructuredGrid(np.asarray(X_verts, order="C"), np.asarray(Y_verts, order="C"), np.zeros_like(X_verts))

def _bottom_y(x, geometry):
    """Physical y of bottom at x. Shape matches x. f(x)=0 if no geometry."""
    if geometry is not None and getattr(geometry, "has_geometry", False):
        return np.asarray(geometry.f(x))
    return np.zeros_like(x)


def _bottom_y_prime(x, geometry):
    """f'(x) for velocity transform u_y = u_eta + f'(x)*u_x. Shape matches x."""
    if geometry is not None and getattr(geometry, "has_geometry", False):
        return np.asarray(geometry.f_1(x))
    return np.zeros_like(x)

def mesh_geometry_for_plot(Nx, Ny, Lx, Ly, dx, dy, geometry=None, use_terrain_coords=False):
    """(X_verts, Y_verts, extent, X_cc_plot, Y_cc_plot). indexing='xy'.

    Warps (x, eta) -> (x, y) with y = eta + f(x) so the VTK grid lives in physical (x,y)
    and VTK handles smooth interpolation. f(x) = geometry.f(x) when geometry, else 0.
    """
    x_verts = np.linspace(0, Lx, Nx + 1)
    eta_verts = np.linspace(0, Ly, Ny + 1)
    X_verts, Eta_verts = np.meshgrid(x_verts, eta_verts, indexing="xy")
    x_cc = (np.arange(Nx) + 0.5) * dx
    eta_cc = (np.arange(Ny) + 0.5) * dy
    X_cc, Eta_cc = np.meshgrid(x_cc, eta_cc, indexing="xy")
    # Smooth warp: physical y = eta + f(x); same point (x, eta) gets value from (x, eta)

    f_verts = _bottom_y(x_verts, geometry)
    f_cc = _bottom_y(x_cc, geometry)
    Y_verts = Eta_verts + f_verts[np.newaxis, :]
    Y_cc_plot = Eta_cc + f_cc[np.newaxis, :]
    y_min, y_max = float(Y_verts.min()), float(Y_verts.max())
    extent = (0.0, Lx, y_min, y_max)
    return X_verts, Y_verts, extent, X_cc, Y_cc_plot


def reference_line_meshes_for_plot(extent, Lx, Ly, use_terrain_coords, geometry=None):
    """(line_x0, line_eta0) in physical (x,y): x=0 vertical (red), bottom y=f(x) (blue)."""
    if not PYVISTA_AVAILABLE:
        return None, None
    y0, y1 = extent[2], extent[3]
    line_x0 = pv.Line((0.0, float(y0), 0.0), (0.0, float(y1), 0.0))
    x_line = np.linspace(0, Lx, 80)
    y_bottom = _bottom_y(x_line, geometry)
    pts = np.column_stack([x_line, y_bottom, np.zeros(80)])
    line_eta0 = pv.Spline(pts, n_points=80)
    return line_x0, line_eta0


def _get_field_for_kind(data, spec):
    """(Z_cc, vmin, vmax) for PyVista subplot."""
    kind = spec["kind"]
    key, vmin_spec, vmax_spec = _FIELD_FOR_KIND[kind]
    phi = data["phi"]
    Z = data.get(key)
    if Z is None:
        Z = np.zeros_like(phi)
    elif kind == "pressure":
        Z = -Z
    vmin = data.get(vmin_spec, vmin_spec) if isinstance(vmin_spec, str) else vmin_spec
    vmax = data.get(vmax_spec, vmax_spec) if isinstance(vmax_spec, str) else vmax_spec
    return Z, vmin, vmax


# Default and velocity subplot font sizes (title_font_size, label_font_size for scalar bar; text for title/step)
_SCALAR_BAR_FONT = (10, 8)
_SCALAR_BAR_FONT_VEL = (8, 6)
_TITLE_FONT_SIZE = 10
_STEP_FONT_SIZE = 10


def _add_scalar_subplot(plotter, row, col, X_verts, Y_verts, Z_cc, title, cbar_label, cmap, vmin, vmax, extent,
                        surface_line_mesh=None, line_x0_mesh=None, line_eta0_mesh=None, scalar_bar_font=None,
                        solid_region_mesh=None, phi_cc=None):
    if not PYVISTA_AVAILABLE or Z_cc.size == 0:
        return
    Z_pt = cell_centered_to_point_2d(Z_cc)
    grid = build_structured_grid_2d(X_verts, Y_verts)
    grid["scalar"] = np.asarray(Z_pt.ravel(order="C"), dtype=np.float64)
    vmin = float(np.nanmin(Z_cc)) if vmin is None else vmin
    vmax = float(np.nanmax(Z_cc)) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1.0
    plotter.subplot(row, col)
    if solid_region_mesh is not None:
        plotter.add_mesh(solid_region_mesh, color="gray", show_edges=False)
    plotter.add_mesh(grid, scalars="scalar", cmap=cmap, clim=[vmin, vmax], show_edges=False, interpolate_before_map=True, show_scalar_bar=False)
    # Phase interface contour phi == 0 when we have two phases
    if phi_cc is not None:
        phi_min, phi_max = float(np.nanmin(phi_cc)), float(np.nanmax(phi_cc))
        if phi_min < 0 < phi_max:
            phi_pt = cell_centered_to_point_2d(np.asarray(phi_cc, dtype=np.float64))
            grid["phi"] = np.asarray(phi_pt.ravel(order="C"), dtype=np.float64)
            try:
                contour_phi0 = grid.contour(scalars="phi", isosurfaces=[0.0])
                if contour_phi0.n_cells > 0:
                    plotter.add_mesh(contour_phi0, color="black", line_width=2.0)
            except Exception:
                pass
    if surface_line_mesh is not None:
        plotter.add_mesh(surface_line_mesh, color="grey", line_width=2)
    if line_x0_mesh is not None:
        plotter.add_mesh(line_x0_mesh, color="red", line_width=2.5)
    if line_eta0_mesh is not None:
        plotter.add_mesh(line_eta0_mesh, color="blue", line_width=2.5)
    tfs, lfs = scalar_bar_font or _SCALAR_BAR_FONT
    plotter.add_scalar_bar(title=cbar_label, vertical=True, n_labels=4, title_font_size=tfs, label_font_size=lfs, font_family="arial", width=0.06, height=0.5, position_x=0.92, position_y=0.2)
    if title:
        plotter.add_text(title, font_size=_TITLE_FONT_SIZE, position="upper_edge")
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.set_background("white")


def _add_streamlines(plotter, row, col, X_verts, Y_verts, U_cc, extent, geometry=None):
    if not PYVISTA_AVAILABLE or U_cc.size == 0 or SKIP_STREAMLINES:
        return
    Ux = cell_centered_to_point_2d(U_cc[:, :, 0])  # (Nx+1, Ny+1) u_x
    Uy_eta = cell_centered_to_point_2d(U_cc[:, :, 1])  # u_eta
    # Curvilinear: no-slip at surface (eta=0). Zero velocity at bottom vertex row so streamlines stop at surface.
    Ux[:, 0] = 0.0
    Uy_eta[:, 0] = 0.0
    # Top: do_nothing in solver extrapolates U; zero top vertex row so streamlines don't cross/emanate from top.
    Ux[:, -1] = 0.0
    Uy_eta[:, -1] = 0.0
    # Physical y-component: u_y = u_eta + f'(x)*u_x so streamlines match (x,y) coords
    x_verts = np.asarray(X_verts[0, :])
    fp = _bottom_y_prime(x_verts, geometry)
    f_prime = np.broadcast_to(np.asarray(fp), x_verts.shape)
    Uy = Uy_eta + f_prime[:, np.newaxis] * Ux
    grid = build_structured_grid_2d(X_verts, Y_verts)
    # Contiguous copy so VTK never holds a pointer to temporary numpy buffers (avoids segfault).
    vec = np.column_stack([Ux.T.ravel(order="F"), Uy.T.ravel(order="F"), np.zeros(Ux.size)])
    vec = np.asarray(vec, dtype=np.float64, order="C").copy()
    np.nan_to_num(vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    grid["vec"] = vec
    x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
    sep = max(0.02, min(x1 - x0, y1 - y0) * 0.08)
    start = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0)
    plotter.subplot(row, col)
    _log_vtk_stack("_add_streamlines", "grid.streamlines_evenly_spaced_2D")
    try:
        sl = grid.streamlines_evenly_spaced_2D(
            vectors="vec",
            start_position=start,
            step_length=sep * 0.5,
            step_unit="l",
            separating_distance=0.03,
            max_steps=500,
            compute_vorticity=False,
        )
        if sl.n_cells > 0:
            plotter.add_mesh(sl, color="w", line_width=1.5)
    except Exception as e:
        _log_vtk_error("_add_streamlines", "streamlines_evenly_spaced_2D", e)
        import warnings
        warnings.warn(f"Streamlines failed: {e}", RuntimeWarning)


def _surface_line_mesh(extent, Lx, geometry, use_terrain_coords):
    if not PYVISTA_AVAILABLE:
        return None
    x_line = np.linspace(extent[0], extent[1], 80)
    y_line = _bottom_y(x_line, geometry)
    return pv.Spline(np.column_stack([x_line, y_line, np.zeros(80)]), n_points=80)


def _solid_region_mesh(extent, Lx, geometry):
    """Gray filled region below eta=0 (below y=f(x)). Added first so it sits behind the fluid."""
    if not PYVISTA_AVAILABLE:
        return None
    y0, y1 = extent[2], extent[3]
    y_low = y0 - 0.15 * (y1 - y0) if y1 > y0 else y0 - 0.01
    x_line = np.linspace(0, Lx, 60)
    y_bottom = _bottom_y(x_line, geometry)
    # Polygon: (0,y_low) -> (Lx,y_low) -> (Lx, f(Lx)) -> ... (curve) -> (0, f(0)) -> (0,y_low)
    pts = np.array(
        [[0.0, float(y_low), 0.0], [float(Lx), float(y_low), 0.0]]
        + [[float(x_line[i]), float(y_bottom[i]), 0.0] for i in range(len(x_line) - 1, -1, -1)]
    )
    n = len(pts)
    faces = np.array([n] + list(range(n)), dtype=np.int64)
    poly = pv.PolyData(pts, faces)
    return poly


def _save_img(save_path, img):
    if not save_path:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.imsave(save_path, img)
    finally:
        plt.close("all")


def render_field_to_image(X_verts, Y_verts, Z_cc, vmin=None, vmax=None, cmap="viridis", size=DEFAULT_RENDER_SIZE, show_edges=False):
    if not PYVISTA_AVAILABLE:
        return None, vmin, vmax
    _log_vtk_stack("render_field_to_image", "add_mesh + screenshot")
    Z_pt = cell_centered_to_point_2d(Z_cc)
    grid = build_structured_grid_2d(X_verts, Y_verts)
    grid["scalar"] = np.asarray(Z_pt.ravel(order="C"), dtype=np.float64)
    vmin = float(np.nanmin(Z_cc)) if vmin is None else vmin
    vmax = float(np.nanmax(Z_cc)) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1.0
    plotter = pv.Plotter(off_screen=True, window_size=size)
    try:
        plotter.add_mesh(grid, scalars="scalar", cmap=cmap, clim=[vmin, vmax], show_edges=show_edges, interpolate_before_map=True, show_scalar_bar=False)
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.set_background("white")
        _log_vtk_stack("render_field_to_image", "screenshot")
        img = _strip_alpha(plotter.screenshot(return_img=True, transparent_background=False))
        return img, vmin, vmax
    finally:
        _close_plotter(plotter)
        plotter = None


def render_single_field_to_file(X_verts, Y_verts, Z_cc, extent, title, scalar_bar_title, save_path, cmap="viridis", vmin=None, vmax=None, surface_line_mesh=None, line_x0_mesh=None, line_eta0_mesh=None, window_size=(1000, 840), phi_cc=None):
    if not PYVISTA_AVAILABLE:
        return
    _log_vtk_stack("render_single_field_to_file", "subplot + screenshot")
    plotter = pv.Plotter(shape=(1, 1), off_screen=True, window_size=window_size)
    try:
        plotter.set_background("white")
        _add_scalar_subplot(plotter, 0, 0, X_verts, Y_verts, Z_cc, title, scalar_bar_title, cmap, vmin, vmax, extent, surface_line_mesh, line_x0_mesh, line_eta0_mesh, phi_cc=phi_cc)
        _log_vtk_stack("render_single_field_to_file", "screenshot")
        img = _strip_alpha(plotter.screenshot(return_img=True, transparent_background=False))
        _save_img(save_path, img)
    finally:
        _close_plotter(plotter)
        plotter = None


def render_velocity_with_streamlines_to_file(X_verts, Y_verts, U_magnitude_cc, U, extent, surface_line_mesh, save_path, line_x0_mesh=None, line_eta0_mesh=None, window_size=(1000, 840), geometry=None, phi_cc=None):
    if not PYVISTA_AVAILABLE:
        return
    _log_vtk_stack("render_velocity_with_streamlines_to_file", "subplot + streamlines + screenshot")
    plotter = pv.Plotter(shape=(1, 1), off_screen=True, window_size=window_size)
    try:
        plotter.set_background("white")
        _add_scalar_subplot(plotter, 0, 0, X_verts, Y_verts, U_magnitude_cc, "Velocity (streamlines)", "Speed", "viridis", None, None, extent, surface_line_mesh, line_x0_mesh, line_eta0_mesh, scalar_bar_font=(4, 3), phi_cc=phi_cc)
        _add_streamlines(plotter, 0, 0, X_verts, Y_verts, U, extent, geometry=geometry)
        _log_vtk_stack("render_velocity_with_streamlines_to_file", "screenshot")
        img = _strip_alpha(plotter.screenshot(return_img=True, transparent_background=False))
        _save_img(save_path, img)
    finally:
        _close_plotter(plotter)
        plotter = None


def create_joint_plot_pyvista_full(phi, U, P, surface_tension, dt, step, dx, dy, mass, rho1, rho2, save_path=None, psi=None, T=None, include_ice_water=False, Lx=1.0, Ly=1.0, geometry=None, window_size=(2400, 1600), use_terrain_coords=False):
    """Joint plot in PyVista. Same layout and API as matplotlib create_joint_plot."""
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista is required for create_joint_plot_pyvista_full")
    _log_vtk_stack("create_joint_plot_pyvista_full", f"step={step} build+plot+screenshot")
    from visualization.plot_layout import get_joint_plot_layout, prepare_joint_plot_data

    data = prepare_joint_plot_data(phi, U, P, surface_tension, rho1, rho2, Lx, Ly, psi=psi, T=T, include_ice_water=include_ice_water)
    nrows, ncols, specs = get_joint_plot_layout(include_ice_water)
    Nx, Ny = data["Nx"], data["Ny"]
    X_verts, Y_verts, extent, _, _ = mesh_geometry_for_plot(Nx, Ny, Lx, Ly, dx, dy, geometry, use_terrain_coords=use_terrain_coords)
    surface_mesh = _surface_line_mesh(extent, Lx, geometry, use_terrain_coords)
    solid_region = _solid_region_mesh(extent, Lx, geometry)
    line_x0, line_eta0 = reference_line_meshes_for_plot(extent, Lx, Ly, use_terrain_coords, geometry)

    plotter = pv.Plotter(shape=(nrows, ncols), off_screen=True, window_size=window_size)
    try:
        try:
            plotter.set_background("white")
            phi_cc = data.get("phi")
            for spec in specs:
                r, c = spec["row"], spec["col"]
                Z_cc, vmin, vmax = _get_field_for_kind(data, spec)
                font = _SCALAR_BAR_FONT_VEL if spec["kind"] in ("velocity_streamlines", "velocity_magnitude") else None
                _add_scalar_subplot(plotter, r, c, X_verts, Y_verts, Z_cc, spec["title"], spec["cbar_label"], spec["cmap"], vmin, vmax, extent, surface_mesh, line_x0, line_eta0, font, solid_region_mesh=solid_region, phi_cc=phi_cc)
                if spec["kind"] == "velocity_streamlines":
                    _add_streamlines(plotter, r, c, X_verts, Y_verts, data["U_masked"], extent, geometry=geometry)
            plotter.subplot(nrows - 1, ncols - 1)
            plotter.add_text(f"Step: {step} | t = {step * dt:.5f}", font_size=_STEP_FONT_SIZE, position="lower_edge")
            _log_vtk_stack("create_joint_plot_pyvista_full", f"step={step} screenshot")
            img = _strip_alpha(plotter.screenshot(return_img=True, transparent_background=False))
            _save_img(save_path, img)
            return img
        except BaseException as e:
            _log_vtk_error("create_joint_plot_pyvista_full", f"step={step} plot/screenshot", e)
            raise
    finally:
        _close_plotter(plotter)
        plotter = None
