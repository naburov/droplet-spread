"""
Collocated (cell-centred) velocity boundary conditions.

This is a cleaned-up version of the original monolithic implementation:
  - No inlet "crutches" (no artificial extension of profiles into the interior,
    no reverse-flow clamping near the inlet).
  - Geometry-aware no-slip logic is preserved.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import jax.numpy as jnp

from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


def _apply_slip_at_wall(u_profile, bottom_bc: str, slip_length: float, dy: float):
    """When bottom is slip, set wall (index 0) to slip value so inlet doesn't pin contact line."""
    u_profile = np.asarray(u_profile, dtype=np.float64)
    if u_profile.size < 2:
        return u_profile
    if bottom_bc == "slip_symmetry":
        u_profile[0] = u_profile[1]
    elif bottom_bc == "navier_slip" and slip_length > 0:
        u_profile[0] = u_profile[1] * slip_length / (slip_length + dy)
    return u_profile


@lru_cache(maxsize=64)
def _solve_slip_blasius_profile(slip_parameter: float, eta_max: float, n_points: int):
    """Solve f''' + 0.5 f f'' = 0 with f'(0) = K f''(0), f'(inf) = 1."""
    from scipy.integrate import solve_bvp

    slip_parameter = max(float(slip_parameter), 0.0)
    eta = np.linspace(0.0, float(eta_max), int(n_points), dtype=np.float64)
    exp_eta = np.exp(-eta)
    guess = np.vstack((
        eta + exp_eta - 1.0,
        1.0 - exp_eta,
        exp_eta,
    ))

    def ode(_, y):
        return np.vstack((y[1], y[2], -0.5 * y[0] * y[2]))

    def bc(ya, yb):
        return np.array((
            ya[0],
            ya[1] - slip_parameter * ya[2],
            yb[1] - 1.0,
        ))

    sol = solve_bvp(ode, bc, eta, guess, tol=1e-6, max_nodes=20000)
    if not sol.success:
        raise RuntimeError(f"Slip-Blasius profile solve failed: {sol.message}")
    values = sol.sol(eta)
    return eta, values[0], values[1], values[2]


def _evaluate_slip_blasius(eta_values, slip_parameter: float, eta_max: float = 12.0):
    eta_values = np.asarray(eta_values, dtype=np.float64)
    eta_grid, f_grid, fp_grid, fpp_grid = _solve_slip_blasius_profile(
        round(float(slip_parameter), 10),
        round(float(max(eta_max, float(np.max(eta_values, initial=0.0)) + 1.0)), 6),
        512,
    )
    fp = np.interp(eta_values, eta_grid, fp_grid, left=fp_grid[0], right=1.0)
    f = np.interp(eta_values, eta_grid, f_grid, left=f_grid[0], right=f_grid[-1])
    fpp = np.interp(eta_values, eta_grid, fpp_grid, left=fpp_grid[0], right=0.0)
    return f, fp, fpp


class CollocatedVelocityBoundaryConditions(BaseBoundaryCondition):
    """Velocity BC with no_slip, do_nothing, slip_symmetry support (collocated grid)."""

    BC_ALIASES = {
        **BaseBoundaryCondition.BC_ALIASES,
        "no_slip": BCType.DIRICHLET,
        "do_nothing": BCType.NEUMANN,
        "slip_symmetry": BCType.SPECIAL,
        "navier_slip": BCType.SPECIAL,
    }

    def __init__(self, config=None):
        super().__init__(config, "velocity")
        if not self.config.get("boundary_conditions", {}).get("velocity"):
            self.bc_raw = {
                "top": "do_nothing",
                "bottom": "no_slip",
                "left": "do_nothing",
                "right": "do_nothing",
            }
            self.bc_types = {b: self._resolve(t) for b, t in self.bc_raw.items()}

        bc_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})
        self.use_geometry = bc_cfg.get("use_geometry", False)
        self.slip_parameter = bc_cfg.get("slip_parameter", 1.0)
        # Navier slip: u_tang|wall = λ ∂u_tang/∂n => u_wall = u_interior * λ/(λ+dn). Same units as dx, dy.
        self.slip_length = float(bc_cfg.get("slip_length", 0.01))

    def apply(self, U, dx: float, dy: float, **kwargs):
        """Apply velocity BCs. U shape: (Nx, Ny, 2). Surface at eta=0 (j=0)."""
        geometry = kwargs.get("geometry")

        for b in ["bottom", "top", "left", "right"]:
            U = self._apply_boundary(U, b, dx, dy)

        if geometry is not None and getattr(geometry, "has_geometry", False) and self.bc_raw.get("top") in ("do_nothing", "slip_symmetry", "navier_slip"):
            Ny = U.shape[1]
            f_1_top = geometry.f_1_grid[:, Ny - 1]
            U = U.at[:, -1, 1].set(U[:, -1, 0] * f_1_top)

        # Corners: set each to a value consistent with BOTH boundaries that meet there.
        U = self._apply_corner_bcs(U, dx, dy)

        return U

    def _apply_boundary(self, U, boundary: str, dx: float, dy: float):
        raw = self.bc_raw[boundary]
        if raw == "no_slip":
            return self._no_slip(U, boundary)
        if raw in ("do_nothing", "neumann"):
            return self._do_nothing(U, boundary)
        if raw == "slip_symmetry":
            return self._slip_symmetry(U, boundary)
        if raw == "navier_slip":
            return self._navier_slip(U, boundary, dx, dy)
        if raw == "dirichlet":
            return self._dirichlet_vel(U, boundary, dx, dy)
        return U

    def _apply_corner_bcs(self, U, dx: float, dy: float):
        """Set each corner so both meeting boundaries are satisfied."""
        Nx, Ny = U.shape[:2]
        top_bc = self.bc_raw.get("top")
        bottom_bc = self.bc_raw.get("bottom")
        left_bc = self.bc_raw.get("left")
        right_bc = self.bc_raw.get("right")

        # Bottom-left (0, 0): no_slip or left Dirichlet already applied; row j=0 forced to 0 above if bottom=no_slip
        if bottom_bc == "no_slip" or left_bc == "no_slip":
            U = U.at[0, 0, :].set(0.0)

        # Bottom-right (Nx-1, 0)
        if bottom_bc == "no_slip" or right_bc == "no_slip":
            U = U.at[-1, 0, :].set(0.0)

        # Top-left (0, Ny-1): top slip wants v=0; left sets u
        if top_bc in ("slip_symmetry", "navier_slip"):
            U = U.at[0, -1, 1].set(0.0)
        elif top_bc == "no_slip":
            U = U.at[0, -1, :].set(0.0)

        # Top-right (Nx-1, Ny-1): top slip wants v=0; right do_nothing sets from interior
        if top_bc in ("slip_symmetry", "navier_slip"):
            U = U.at[-1, -1, 1].set(0.0)
        elif top_bc == "no_slip":
            U = U.at[-1, -1, :].set(0.0)

        return U

    def _no_slip(self, U, b):
        if b == "top":
            return U.at[:, -1, :].set(0.0)
        if b == "bottom":
            return U.at[:, 0, :].set(0.0)
        if b == "left":
            return U.at[0, :, :].set(0.0)
        if b == "right":
            return U.at[-1, :, :].set(0.0)
        return U

    def _do_nothing(self, U, b):
        if b == "top":
            return U.at[:, -1, :].set(U[:, -2, :])
        if b == "bottom":
            return U.at[:, 0, :].set(U[:, 1, :])
        if b == "left":
            return U.at[0, :, :].set(U[1, :, :])
        if b == "right":
            return U.at[-1, :, :].set(U[-2, :, :])
        return U

    def _slip_symmetry(self, U, b):
        """Zero normal velocity, zero gradient for tangential (free slip, λ→∞).
        Compatible with dirichlet/do_nothing at adjacent boundaries: corner keeps normal=0 from slip, tangential from the other boundary; corners are only zeroed when at least one side is no_slip."""
        if b == "top":
            U = U.at[:, -1, 1].set(0.0)
            U = U.at[:, -1, 0].set(U[:, -2, 0])
        elif b == "bottom":
            U = U.at[:, 0, 1].set(0.0)
            U = U.at[:, 0, 0].set(U[:, 1, 0])
        elif b == "left":
            U = U.at[0, :, 0].set(0.0)
            U = U.at[0, :, 1].set(U[1, :, 1])
        elif b == "right":
            U = U.at[-1, :, 0].set(0.0)
            U = U.at[-1, :, 1].set(U[-2, :, 1])
        return U

    def _navier_slip(self, U, b, dx: float, dy: float):
        """Navier slip: u·n = 0 and u_tang|wall = λ ∂u_tang/∂n => u_wall = u_interior * λ/(λ+dn).
        λ = self.slip_length (same units as dx, dy). λ=0 => no_slip; λ→∞ => free slip."""
        lam = self.slip_length
        if lam <= 0.0:
            return self._no_slip(U, b)
        if b == "top":
            U = U.at[:, -1, 1].set(0.0)
            dn = dy
            coef = lam / (lam + dn)
            U = U.at[:, -1, 0].set(U[:, -2, 0] * coef)
        elif b == "bottom":
            U = U.at[:, 0, 1].set(0.0)
            dn = dy
            coef = lam / (lam + dn)
            U = U.at[:, 0, 0].set(U[:, 1, 0] * coef)
        elif b == "left":
            U = U.at[0, :, 0].set(0.0)
            dn = dx
            coef = lam / (lam + dn)
            U = U.at[0, :, 1].set(U[1, :, 1] * coef)
        elif b == "right":
            U = U.at[-1, :, 0].set(0.0)
            dn = dx
            coef = lam / (lam + dn)
            U = U.at[-1, :, 1].set(U[-2, :, 1] * coef)
        return U

    def _dirichlet_vel(self, U, b, dx: float, dy: float):
        """Apply Dirichlet velocity BC with optional y-dependent profile."""
        val = self.dirichlet_values.get(b, {"u": 0.0, "v": 0.0})
        if isinstance(val, (list, tuple)):
            u_val, v_val = float(val[0]), float(val[1])
        else:
            u_val, v_val = float(val.get("u", 0.0)), float(val.get("v", 0.0))

        bc_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})
        profile_cfg = bc_cfg.get("dirichlet_profiles", {}).get(b, {})
        profile_type = profile_cfg.get("type", None)

        # For left/right boundaries, check if we need y-dependent profile
        if b in ["left", "right"] and profile_type == "linear":
            return self._dirichlet_vel_profile(
                U, b, u_val, v_val, dx, dy, profile_cfg
            )
        if b in ["left", "right"] and profile_type == "smooth":
            return self._dirichlet_vel_smooth(U, b, u_val, v_val, dy, profile_cfg)
        if b in ["left", "right"] and profile_type == "boundary_layer":
            return self._dirichlet_vel_boundary_layer(
                U, b, u_val, v_val, dx, dy, profile_cfg
            )

        # Default: constant value
        if b == "top":
            return U.at[:, -1, 0].set(u_val).at[:, -1, 1].set(v_val)
        if b == "bottom":
            return U.at[:, 0, 0].set(u_val).at[:, 0, 1].set(v_val)
        if b == "left":
            return U.at[0, :, 0].set(u_val).at[0, :, 1].set(v_val)
        if b == "right":
            return U.at[-1, :, 0].set(u_val).at[-1, :, 1].set(v_val)
        return U

    def _dirichlet_vel_profile(
        self, U, b, u_target: float, v_val: float, dx: float, dy: float, profile_cfg: dict
    ):
        """Apply simple linear-in-y inlet/outlet profile (no crutches)."""
        Nx, Ny = U.shape[:2]
        Ly = Ny * dy

        y_coords = jnp.arange(Ny, dtype=jnp.float32) * dy
        y_half = Ly / 2.0
        u_profile = jnp.where(
            y_coords <= y_half,
            (2.0 * u_target / Ly) * y_coords,
            u_target,
        )
        # Slip at bottom: wall gets slip value so contact line can move; else no-slip
        if self.bc_raw.get("bottom") == "slip_symmetry":
            u_profile = u_profile.at[0].set(u_profile[1])
        elif self.bc_raw.get("bottom") == "navier_slip" and getattr(self, "slip_length", 0.0) > 0:
            coef = self.slip_length / (self.slip_length + dy)
            u_profile = u_profile.at[0].set(u_profile[1] * coef)
        else:
            u_profile = u_profile.at[0].set(0.0)

        v_profile = jnp.full(Ny, v_val)
        v_profile = v_profile.at[0].set(0.0)

        if b == "left":
            U = U.at[0, :, 0].set(u_profile)
            U = U.at[0, :, 1].set(v_profile)
        elif b == "right":
            U = U.at[-1, :, 0].set(u_profile)
            U = U.at[-1, :, 1].set(v_profile)
        return U

    def get_inlet_profile(self, Ny: int, dy: float):
        """Get inlet velocity profile at all y positions.
        When bottom is slip_symmetry or navier_slip, wall (j=0) gets slip value so left contact line can move."""
        bc_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})
        if bc_cfg.get("left") != "dirichlet":
            return None

        dirichlet_vals = bc_cfg.get("dirichlet_values", {}).get("left", {})
        if isinstance(dirichlet_vals, dict):
            u_target = dirichlet_vals.get("u", 0.0)
        else:
            u_target = (
                float(dirichlet_vals[0])
                if isinstance(dirichlet_vals, (list, tuple))
                else 0.0
            )

        profile_cfg = bc_cfg.get("dirichlet_profiles", {}).get("left", {})
        profile_type = profile_cfg.get("type", None)
        bottom_bc = bc_cfg.get("bottom", "no_slip")
        slip_length = float(bc_cfg.get("slip_length", 0.01))

        Ly = Ny * dy
        y_coords = np.arange(Ny) * dy

        if profile_type is None:
            u_profile = np.full(Ny, u_target)
            u_profile[0] = 0.0
            u_profile = _apply_slip_at_wall(u_profile, bottom_bc, slip_length, dy)
            return u_profile

        if profile_type == "linear":
            y_half = Ly / 2.0
            u_profile = np.where(
                y_coords <= y_half,
                (2.0 * u_target / Ly) * y_coords,
                u_target,
            )
            u_profile[0] = 0.0
            u_profile = _apply_slip_at_wall(u_profile, bottom_bc, slip_length, dy)
            return u_profile

        if profile_type == "boundary_layer":
            u_profile, _ = self._boundary_layer_profiles_np(
                Ny, dy, u_target, profile_cfg, y_coords=y_coords
            )
            return u_profile

        # Default: constant
        u_profile = np.full(Ny, u_target)
        u_profile[0] = 0.0
        u_profile = _apply_slip_at_wall(u_profile, bottom_bc, slip_length, dy)
        return u_profile

    def get_inlet_normal_profile(self, n_points: int, dy: float):
        """Get optional wall-normal inlet profile on y-face locations for MAC grids."""
        bc_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})
        if bc_cfg.get("left") != "dirichlet":
            return None

        profile_cfg = bc_cfg.get("dirichlet_profiles", {}).get("left", {})
        if not bool(profile_cfg.get("include_normal_velocity", False)):
            return None

        dirichlet_vals = bc_cfg.get("dirichlet_values", {}).get("left", {})
        u_target = dirichlet_vals.get("u", 0.0) if isinstance(dirichlet_vals, dict) else 0.0
        y_coords = np.arange(n_points, dtype=np.float64) * dy
        _, v_profile = self._boundary_layer_profiles_np(
            n_points, dy, float(u_target), profile_cfg, y_coords=y_coords
        )
        return v_profile

    def _boundary_layer_profiles_np(self, Ny: int, dy: float, u_target: float, profile_cfg: dict, y_coords=None):
        bc_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})
        physical_params = self.config.get("physical_params", {})
        Re1 = physical_params.get("Re1", 100.0)
        Re2 = physical_params.get("Re2", 1000.0)
        Re_to_use = profile_cfg.get("reynolds_number", "Re2")
        Re = float(Re1 if Re_to_use == "Re1" else Re2)

        Ly = Ny * dy
        char_length = profile_cfg.get("characteristic_length", "Ly")
        L_char = Ly if char_length == "Ly" else 1.0
        bl_exponent = float(profile_cfg.get("bl_exponent", 0.5))
        bl_thickness = np.clip(L_char / (Re ** bl_exponent), dy, Ly)
        transition_scale = float(profile_cfg.get("transition_scale", 1.0))
        effective_thickness = bl_thickness * transition_scale
        if y_coords is None:
            y_coords = np.arange(Ny, dtype=np.float64) * dy
        eta = np.asarray(y_coords, dtype=np.float64) / effective_thickness

        profile_subtype = profile_cfg.get("subtype", "blasius")
        if profile_subtype == "slip_blasius":
            bottom_bc = bc_cfg.get("bottom", "no_slip")
            slip_length = float(bc_cfg.get("slip_length", 0.01))
            default_slip = slip_length / effective_thickness if bottom_bc == "navier_slip" else 0.0
            slip_parameter = float(profile_cfg.get("slip_parameter", default_slip))
            eta_max = float(profile_cfg.get("eta_max", 12.0))
            f_profile, fp_profile, _ = _evaluate_slip_blasius(eta, slip_parameter, eta_max=eta_max)
            u_profile = float(u_target) * fp_profile
            x_ref = float(profile_cfg.get("x_ref", L_char))
            if bool(profile_cfg.get("include_normal_velocity", False)) and x_ref > 0.0:
                normal_scale = 0.5 * effective_thickness / x_ref
                v_profile = float(u_target) * normal_scale * (eta * fp_profile - f_profile)
            else:
                v_profile = np.zeros_like(u_profile)
        elif profile_subtype == "power_law":
            power = profile_cfg.get("power", 1.0 / 7.0)
            u_profile = np.where(eta < 1.0, float(u_target) * (eta ** power), float(u_target))
            v_profile = np.zeros_like(u_profile)
        elif profile_subtype == "exponential":
            k = profile_cfg.get("exponential_factor", 2.5)
            u_profile = float(u_target) * (1.0 - np.exp(-k * eta))
            v_profile = np.zeros_like(u_profile)
        elif profile_subtype == "blasius":
            u_profile = np.where(eta < 3.0, float(u_target) * np.tanh(2.5 * eta), float(u_target))
            v_profile = np.zeros_like(u_profile)
        else:
            u_profile = float(u_target) * np.ones(Ny)
            v_profile = np.zeros_like(u_profile)

        u_profile = np.asarray(u_profile, dtype=np.float64)
        v_profile = np.asarray(v_profile, dtype=np.float64)
        if profile_subtype != "slip_blasius":
            u_profile[0] = 0.0
            bottom_bc = bc_cfg.get("bottom", "no_slip")
            slip_length = float(bc_cfg.get("slip_length", 0.01))
            u_profile = _apply_slip_at_wall(u_profile, bottom_bc, slip_length, dy)
        v_profile[0] = 0.0
        return u_profile, v_profile

    def _dirichlet_vel_boundary_layer(
        self, U, b, u_target: float, v_val: float, dx: float, dy: float, profile_cfg: dict
    ):
        """Apply boundary-layer-style profile on left/right boundaries."""
        Nx, Ny = U.shape[:2]
        Ly = Ny * dy
        Lx = Nx * dx

        physical_params = self.config.get("physical_params", {})
        Re1 = physical_params.get("Re1", 100.0)
        Re2 = physical_params.get("Re2", 1000.0)

        Re_to_use = profile_cfg.get("reynolds_number", "Re2")
        Re = Re1 if Re_to_use == "Re1" else Re2

        y_coords = np.arange(Ny, dtype=np.float64) * dy
        u_profile_np, v_profile_np = self._boundary_layer_profiles_np(
            Ny, dy, float(u_target), profile_cfg, y_coords=y_coords
        )
        if not bool(profile_cfg.get("include_normal_velocity", False)):
            v_profile_np = np.full(Ny, float(v_val), dtype=np.float64)
            v_profile_np[0] = 0.0
        u_profile = jnp.asarray(u_profile_np)
        v_profile = jnp.asarray(v_profile_np)

        if b == "left":
            U = U.at[0, :, 0].set(u_profile)
            U = U.at[0, :, 1].set(v_profile)
        elif b == "right":
            U = U.at[-1, :, 0].set(u_profile)
            U = U.at[-1, :, 1].set(v_profile)

        return U
