"""
Staggered (MAC) velocity boundary conditions.

Apply BCs directly to face velocities (u_face, v_face) so divergence and PPE
RHS are consistent with the enforced boundary values.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .collocated import CollocatedVelocityBoundaryConditions


class StaggeredVelocityBoundaryConditions(CollocatedVelocityBoundaryConditions):
    """Staggered BCs: apply to (u_face, v_face) for correct use in PPE."""

    def apply_to_faces(self, u_face, v_face, dx: float, dy: float, **kwargs):
        """
        Apply velocity BCs directly to MAC face arrays.

        u_face: (Nx+1, Ny), v_face: (Nx, Ny+1).
        Returns (u_face, v_face) with BCs applied.
        With geometry (curvilinear top y = Ly + f(x)), top BCs use normal n = (-f', 1):
        zero normal velocity => v = u*f' at top.
        """
        u_face = jnp.array(u_face) if not isinstance(u_face, jnp.ndarray) else u_face
        v_face = jnp.array(v_face) if not isinstance(v_face, jnp.ndarray) else v_face
        Nxu, Ny = u_face.shape
        Nx = Nxu - 1
        Nyv = v_face.shape[1]
        assert Nyv == Ny + 1
        geometry = kwargs.get("geometry")
        phi = kwargs.get("phi")
        has_top_curv = (
            geometry is not None
            and getattr(geometry, "has_geometry", False)
            and hasattr(geometry, "f_1_grid")
        )
        has_bottom_curv = has_top_curv
        f_1_top = geometry.f_1_grid[:, Ny - 1] if has_top_curv else None
        f_1_bottom = geometry.f_1_grid[:, 0] if has_bottom_curv else None
        vel_cfg = self.config.get("boundary_conditions", {}).get("velocity", {})

        # Bottom (j=0): no_slip => u=v=0; slip_symmetry => v=0, u extrapolated;
        # navier_slip => v=0, u = u_int*λ/(λ+dy); do_nothing/neumann => extrapolate both.
        lam = getattr(self, "slip_length", 0.01)
        if self.bc_raw.get("bottom") == "no_slip":
            v_face = v_face.at[:, 0].set(0.0)
            u_face = u_face.at[:, 0].set(0.0)
        elif self.bc_raw.get("bottom") == "dirichlet":
            dv = self.dirichlet_values.get("bottom", {"u": 0.0, "v": 0.0})
            u_val = dv.get("u", 0.0) if isinstance(dv, dict) else 0.0
            v_val = dv.get("v", 0.0) if isinstance(dv, dict) else 0.0
            v_face = v_face.at[:, 0].set(float(v_val))
            u_face = u_face.at[:, 0].set(float(u_val))
        elif self.bc_raw.get("bottom") == "slip_symmetry":
            u_face = u_face.at[:, 0].set(u_face[:, 1])
            if f_1_bottom is not None:
                u_avg = 0.5 * (u_face[:-1, 0] + u_face[1:, 0])
                v_face = v_face.at[:, 0].set(u_avg * f_1_bottom)
            else:
                v_face = v_face.at[:, 0].set(0.0)
        elif self.bc_raw.get("bottom") == "navier_slip":
            phase_aware = bool(vel_cfg.get("navier_slip_phase_aware", False))
            if phase_aware and phi is not None:
                phi_np = np.asarray(phi)
                if phi_np.ndim == 2 and phi_np.shape == (Nx, Ny):
                    # Liquid fraction from phase field at bottom cells:
                    # phi=-1 -> liquid (w=1), phi=+1 -> gas (w=0).
                    w_liq_cell = np.clip((1.0 - phi_np[:, 0]) * 0.5, 0.0, 1.0)
                    w_liq_face = np.empty((Nx + 1,), dtype=np.float64)
                    w_liq_face[0] = w_liq_cell[0]
                    w_liq_face[Nx] = w_liq_cell[-1]
                    if Nx > 1:
                        w_liq_face[1:Nx] = 0.5 * (w_liq_cell[1:] + w_liq_cell[:-1])

                    lam_liq = float(vel_cfg.get("slip_length_liquid", lam))
                    lam_gas = float(vel_cfg.get("slip_length_gas", 0.0))
                    lam_eff = lam_gas + w_liq_face * (lam_liq - lam_gas)
                    coef_face = lam_eff / (lam_eff + dy)
                    coef_face = jnp.array(coef_face, dtype=u_face.dtype)
                    u_face = u_face.at[:, 0].set(u_face[:, 1] * coef_face)
                elif lam > 0.0:
                    coef = lam / (lam + dy)
                    u_face = u_face.at[:, 0].set(u_face[:, 1] * coef)
                else:
                    u_face = u_face.at[:, 0].set(0.0)
            elif lam > 0.0:
                coef = lam / (lam + dy)
                u_face = u_face.at[:, 0].set(u_face[:, 1] * coef)
            else:
                u_face = u_face.at[:, 0].set(0.0)
            if f_1_bottom is not None:
                u_avg = 0.5 * (u_face[:-1, 0] + u_face[1:, 0])
                v_face = v_face.at[:, 0].set(u_avg * f_1_bottom)
            else:
                v_face = v_face.at[:, 0].set(0.0)
        elif self.bc_raw.get("bottom") in ("do_nothing", "neumann"):
            v_face = v_face.at[:, 0].set(v_face[:, 1])
            u_face = u_face.at[:, 0].set(u_face[:, 1])

        # Top (j=Ny for v; last row of u is j=Ny-1). Curvilinear: top is y = Ly + f(x), normal n = (-f', 1) => v = u*f'
        if self.bc_raw.get("top") == "no_slip":
            v_face = v_face.at[:, Ny].set(0.0)
            u_face = u_face.at[:, Ny - 1].set(0.0)
        elif self.bc_raw.get("top") == "dirichlet":
            dv = self.dirichlet_values.get("top", {"u": 0.0, "v": 0.0})
            u_val = dv.get("u", 0.0) if isinstance(dv, dict) else 0.0
            v_val = dv.get("v", 0.0) if isinstance(dv, dict) else 0.0
            v_face = v_face.at[:, Ny].set(float(v_val))
            u_face = u_face.at[:, Ny - 1].set(float(u_val))
        elif self.bc_raw.get("top") in ("do_nothing", "neumann"):
            if f_1_top is not None:
                u_avg = 0.5 * (u_face[:-1, Ny - 1] + u_face[1:, Ny - 1])
                v_face = v_face.at[:, Ny].set(u_avg * f_1_top)
            else:
                v_face = v_face.at[:, Ny].set(v_face[:, Ny - 1])
            u_face = u_face.at[:, Ny - 1].set(u_face[:, Ny - 2])
        elif self.bc_raw.get("top") == "slip_symmetry":
            if f_1_top is not None:
                u_avg = 0.5 * (u_face[:-1, Ny - 1] + u_face[1:, Ny - 1])
                v_face = v_face.at[:, Ny].set(u_avg * f_1_top)
            else:
                v_face = v_face.at[:, Ny].set(0.0)
            u_face = u_face.at[:, Ny - 1].set(u_face[:, Ny - 2])
        elif self.bc_raw.get("top") == "navier_slip":
            if f_1_top is not None:
                u_avg = 0.5 * (u_face[:-1, Ny - 1] + u_face[1:, Ny - 1])
                v_face = v_face.at[:, Ny].set(u_avg * f_1_top)
            else:
                v_face = v_face.at[:, Ny].set(0.0)
            if lam > 0.0:
                coef = lam / (lam + dy)
                u_face = u_face.at[:, Ny - 1].set(u_face[:, Ny - 2] * coef)
            else:
                u_face = u_face.at[:, Ny - 1].set(0.0)

        # Left (i=0): no_slip => u=v=0; slip_symmetry => u=0, v extrapolated; navier_slip => u=0, v = v_int*λ/(λ+dx); do_nothing => extrapolate; dirichlet => profile
        if self.bc_raw.get("left") == "no_slip":
            u_face = u_face.at[0, :].set(0.0)
            v_face = v_face.at[0, :].set(0.0)
        elif self.bc_raw.get("left") == "slip_symmetry":
            u_face = u_face.at[0, :].set(0.0)
            v_face = v_face.at[0, :].set(v_face[1, :])
        elif self.bc_raw.get("left") == "navier_slip":
            u_face = u_face.at[0, :].set(0.0)
            if lam > 0.0:
                coef = lam / (lam + dx)
                v_face = v_face.at[0, :].set(v_face[1, :] * coef)
            else:
                v_face = v_face.at[0, :].set(0.0)
        elif self.bc_raw.get("left") in ("do_nothing", "neumann"):
            u_face = u_face.at[0, :].set(u_face[1, :])
            v_face = v_face.at[0, :].set(v_face[1, :])
        elif self.bc_raw.get("left") == "dirichlet":
            u_profile = self.get_inlet_profile(Ny, dy)
            if u_profile is not None:
                u_profile = jnp.array(np.asarray(u_profile))
                u_face = u_face.at[0, :].set(u_profile)
            v_profile = None
            if hasattr(self, "get_inlet_normal_profile"):
                v_profile = self.get_inlet_normal_profile(Ny + 1, dy)
            if v_profile is not None:
                v_profile = jnp.array(np.asarray(v_profile))
                v_face = v_face.at[0, :].set(v_profile)
            else:
                v_face = v_face.at[0, :].set(0.0)

        # Right (i=Nx for u; rightmost v column is i=Nx-1)
        if self.bc_raw.get("right") == "no_slip":
            u_face = u_face.at[Nx, :].set(0.0)
            v_face = v_face.at[Nx - 1, :].set(0.0)
        elif self.bc_raw.get("right") == "slip_symmetry":
            u_face = u_face.at[Nx, :].set(0.0)
            v_face = v_face.at[Nx - 1, :].set(v_face[Nx - 2, :])
        elif self.bc_raw.get("right") == "navier_slip":
            u_face = u_face.at[Nx, :].set(0.0)
            if lam > 0.0:
                coef = lam / (lam + dx)
                v_face = v_face.at[Nx - 1, :].set(v_face[Nx - 2, :] * coef)
            else:
                v_face = v_face.at[Nx - 1, :].set(0.0)
        elif self.bc_raw.get("right") in ("do_nothing", "neumann"):
            u_face = u_face.at[Nx, :].set(u_face[Nx - 1, :])
            v_face = v_face.at[Nx - 1, :].set(v_face[Nx - 2, :])

        # Reconcile right-side corners after right boundary overwrite.
        # Bottom/top normal-velocity constraints must win over right do_nothing,
        # otherwise the bottom-right / top-right cells retain large divergence spikes.
        if self.bc_raw.get("right") in ("do_nothing", "neumann"):
            if self.bc_raw.get("bottom") in ("no_slip", "slip_symmetry", "navier_slip"):
                if f_1_bottom is not None:
                    v_face = v_face.at[Nx - 1, 0].set(
                        0.5 * (u_face[Nx - 1, 0] + u_face[Nx, 0]) * f_1_bottom[Nx - 1]
                    )
                else:
                    v_face = v_face.at[Nx - 1, 0].set(0.0)
            elif self.bc_raw.get("bottom") == "dirichlet":
                dv = self.dirichlet_values.get("bottom", {"u": 0.0, "v": 0.0})
                v_val = dv.get("v", 0.0) if isinstance(dv, dict) else 0.0
                v_face = v_face.at[Nx - 1, 0].set(float(v_val))

            if self.bc_raw.get("top") in ("no_slip", "slip_symmetry", "navier_slip"):
                if f_1_top is not None:
                    v_face = v_face.at[Nx - 1, Ny].set(
                        0.5 * (u_face[Nx - 1, Ny - 1] + u_face[Nx, Ny - 1]) * f_1_top[Nx - 1]
                    )
                else:
                    v_face = v_face.at[Nx - 1, Ny].set(0.0)
            elif self.bc_raw.get("top") == "dirichlet":
                dv = self.dirichlet_values.get("top", {"u": 0.0, "v": 0.0})
                v_val = dv.get("v", 0.0) if isinstance(dv, dict) else 0.0
                v_face = v_face.at[Nx - 1, Ny].set(float(v_val))

            # Make the right boundary flux continuity-compatible for the last cell column:
            # div[i=Nx-1,j] = (u[Nx,j] - u[Nx-1,j]) / dx + (v[Nx-1,j+1] - v[Nx-1,j]) / dy.
            # Solve this for u[Nx,j] so the outlet strip does not retain a large divergence
            # bias from the copied v-column.
            outlet_u = u_face[Nx - 1, :] - (dx / dy) * (v_face[Nx - 1, 1:] - v_face[Nx - 1, :-1])
            u_face = u_face.at[Nx, :].set(outlet_u)

        return u_face, v_face
