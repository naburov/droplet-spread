"""
Simplified Pressure Projection Equation (PPE) solver.

Global PPE only - no local refinement.
"""

import numpy as np
import jax.numpy as jnp
import scipy.sparse
import scipy.sparse.linalg
from .ppe_utils import (
    check_divergence,
    compute_divergence_for_solve,
    solve_pressure_correction,
    apply_pressure_correction_to_velocity
)
from .ppe_diagnostics import dump_ppe_diagnostics
from numerics.staggered_utils import to_staggered, to_collocated
from numerics.finite_differences import jax_divergence, jax_gradient
from numerics.staggered_mac import divergence as mac_divergence, grad_p_to_faces
from physics.properties import jax_calculate_density


def _has_terrain(geometry):
    return bool(geometry is not None and getattr(geometry, "has_geometry", False))


def _mac_divergence_geometry(u_face, v_face, dx, dy, geometry):
    """MAC divergence for flat or terrain-following coordinates.

    For terrain y = eta + h(x), physical divergence is
        u_x - h' u_eta + v_eta.
    Face velocities store physical components on computational MAC faces.
    """
    div = mac_divergence(u_face, v_face, dx, dy)
    if not _has_terrain(geometry):
        return div

    u_center = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    u_eta = jnp.zeros_like(u_center)
    u_eta = u_eta.at[:, 1:-1].set((u_center[:, 2:] - u_center[:, :-2]) / (2.0 * dy))
    u_eta = u_eta.at[:, 0].set((u_center[:, 1] - u_center[:, 0]) / dy)
    u_eta = u_eta.at[:, -1].set((u_center[:, -1] - u_center[:, -2]) / dy)
    return div - geometry.f_1_grid * u_eta


def _terrain_grad_p_to_faces(p, dx, dy, geometry):
    """Physical pressure gradient on MAC faces for terrain coordinates."""
    dpdx_face, dpeta_face = grad_p_to_faces(p, dx, dy)
    if not _has_terrain(geometry):
        return dpdx_face, dpeta_face

    f_1 = geometry.f_1_grid
    p_eta_cc = jnp.zeros_like(p)
    p_eta_cc = p_eta_cc.at[:, 1:-1].set((p[:, 2:] - p[:, :-2]) / (2.0 * dy))
    p_eta_cc = p_eta_cc.at[:, 0].set((p[:, 1] - p[:, 0]) / dy)
    p_eta_cc = p_eta_cc.at[:, -1].set((p[:, -1] - p[:, -2]) / dy)

    Nx, Ny = p.shape
    p_eta_u = jnp.zeros((Nx + 1, Ny), dtype=p.dtype)
    p_eta_u = p_eta_u.at[1:Nx, :].set(0.5 * (p_eta_cc[1:, :] + p_eta_cc[:-1, :]))
    p_eta_u = p_eta_u.at[0, :].set(p_eta_cc[0, :])
    p_eta_u = p_eta_u.at[Nx, :].set(p_eta_cc[-1, :])

    f_1_u = jnp.zeros((Nx + 1, Ny), dtype=p.dtype)
    f_1_u = f_1_u.at[1:Nx, :].set(0.5 * (f_1[1:, :] + f_1[:-1, :]))
    f_1_u = f_1_u.at[0, :].set(f_1[0, :])
    f_1_u = f_1_u.at[Nx, :].set(f_1[-1, :])

    grad_x_u = dpdx_face - f_1_u * p_eta_u
    grad_y_v = dpeta_face
    return grad_x_u, grad_y_v


def ppe_solve(U, dx, dy, dt, geometry, correction_solver=None,
              velocity_bc_manager=None, ppe_bcs=None,
              psi=None,
              div_threshold=0.05, max_div_threshold=0.05, mean_div_threshold=0.1,
              max_iterations=1000, debug_output_dir=None, debug_step=None,
              under_relaxation=1.0):
    """Simplified PPE solver for incompressible flow. Curvilinear (x, eta): fluid-only grid."""
    divergence, max_div, mean_div, max_div_interior = check_divergence(U, dx, dy, geometry)
    
    info = {
        'applied': False,
        'iterations': 0,
        'div_before_max': float(max_div),
        'div_before_mean': float(mean_div),
        'last_max_div': float(max_div),
    }
    
    # Check if PPE is needed (use interior max so boundary spikes don't force PPE)
    if max_div_interior <= max_div_threshold and mean_div <= mean_div_threshold:
        info['div_after_max'] = float(max_div)
        info['div_after_mean'] = float(mean_div)
        return U, info
    
    # Safety check - if divergence is catastrophically high, skip PPE
    if max_div > 1e6 or mean_div > 1e4:
        print(f"⚠️  WARNING: Extremely high divergence detected (max={max_div:.2e}, mean={mean_div:.2e})")
        print(f"   Skipping PPE correction to prevent exponential divergence.")
        info['div_after_max'] = float(max_div)
        info['div_after_mean'] = float(mean_div)
        info['applied'] = False
        info['skipped_due_to_high_div'] = True
        return U, info
    
    info['applied'] = True
    
    # Store initial velocity for diagnostics (BEFORE PPE cycle)
    U_before_ppe = U.copy() if isinstance(U, np.ndarray) else np.array(U)
    
    # PPE iteration loop
    for iteration in range(max_iterations):
        divergence, max_div, mean_div, max_div_interior = check_divergence(U, dx, dy, geometry)
        # Safety check
        if max_div > 1e6 or mean_div > 1e4:
            print(f"⚠️  PPE iteration {iteration}: Divergence exploded (max={max_div:.2e}, mean={mean_div:.2e})")
            info['iterations'] = iteration
            info['div_after_max'] = float(max_div)
            info['div_after_mean'] = float(mean_div)
            info['diverged'] = True
            # Dump diagnostics even if PPE diverged
            if debug_output_dir is not None and debug_step is not None:
                U_after_ppe = U.copy() if isinstance(U, np.ndarray) else np.array(U)
                try:
                    dump_ppe_diagnostics(
                        U_before_ppe, U_after_ppe, dx, dy,
                        debug_step, -1, debug_output_dir  # -1 indicates final (even if diverged)
                    )
                except Exception as e:
                    print(f"⚠️  Failed to dump PPE diagnostics: {e}")
            return U, info
        
        # Store u* before correction (for compatibility BC)
        U_star = U.copy() if isinstance(U, np.ndarray) else np.array(U)
        
        div_for_solve = compute_divergence_for_solve(
            U, dx, dy, dt, divergence, psi, geometry, ppe_bcs
        )
        p_correction = solve_pressure_correction(
            div_for_solve, correction_solver, geometry, dy, psi,
            U_star, velocity_bc_manager, dx, dt
        )
        U = apply_pressure_correction_to_velocity(
            U, p_correction, dt, dx, dy, geometry, psi,
            under_relaxation=under_relaxation
        )
        U = _apply_velocity_bcs(U, dx, dy, velocity_bc_manager, psi, geometry)
        
        # Progress output every 10 steps (and first 5)
        if iteration % 10 == 0 or iteration < 5:
            import sys
            sys.stdout.write(f"\rPPE: iter {iteration}, max_div: {max_div:.6f}, mean_div: {mean_div:.6f}")
            sys.stdout.flush()
        
        # Check if divergence is increasing (solver instability)
        if iteration > 0:
            prev_max_div = info.get('last_max_div', max_div)
            if max_div > prev_max_div * 1.5 and max_div > 10.0:
                print(f"\n⚠️  PPE divergence increasing: {prev_max_div:.2e} -> {max_div:.2e}")
                info['iterations'] = iteration
                info['div_after_max'] = float(max_div)
                info['div_after_mean'] = float(mean_div)
                info['diverged'] = True
                return U, info
        
        info['last_max_div'] = float(max_div)
        
        # Check convergence (use interior max so boundary spikes don't block)
        if max_div_interior <= max_div_threshold and mean_div <= mean_div_threshold:
            break
    
    import sys
    sys.stdout.write(f"\nPPE: {iteration + 1} iterations\n")
    
    # Final diagnostics: BEFORE PPE cycle vs AFTER PPE cycle (final)
    if debug_output_dir is not None and debug_step is not None:
        U_after_ppe = U.copy() if isinstance(U, np.ndarray) else np.array(U)
        try:
            dump_ppe_diagnostics(
                U_before_ppe, U_after_ppe, dx, dy,
                debug_step, -1, debug_output_dir  # -1 indicates final (after complete cycle)
            )
        except Exception as e:
            print(f"⚠️  Failed to dump final PPE diagnostics: {e}")
    
    info['iterations'] = iteration + 1
    info['div_after_max'] = float(max_div)
    info['div_after_mean'] = float(mean_div)
    
    return U, info


# ---------- Staggered (MAC) PPE correction (incremental integration) ----------

def ppe_solve_staggered(
    U,
    dx,
    dy,
    dt,
    geometry,
    correction_solver=None,
    velocity_bc_manager=None,
    ppe_bcs=None,
    psi=None,
    div_threshold=0.05,
    max_div_threshold=0.05,
    mean_div_threshold=0.1,
    min_iterations=1,
    max_iterations=50,
    debug_output_dir=None,
    debug_step=None,
    under_relaxation=1.0,
    phi=None,
    rho1=None,
    rho2=None,
    convergence_mode="interior",
    u_face_in=None,
    v_face_in=None,
):
    """Staggered PPE. Grid is fluid-only (bottom-aligned)."""
    if phi is None or rho1 is None or rho2 is None:
        raise ValueError(
            "ppe_solve_staggered requires variable-density inputs: phi, rho1, rho2."
        )
    use_terrain_projection = _has_terrain(geometry)

    divergence_cc, max_div, mean_div, max_div_interior = check_divergence(U, dx, dy, geometry)
    info = {
        "applied": False,
        "iterations": 0,
        "div_before_max": float(max_div),
        "div_before_mean": float(mean_div),
        "div_after_max": float(max_div),
        "div_after_mean": float(mean_div),
    }
    mode = str(convergence_mode).lower()
    if mode not in ("interior", "global", "both"):
        mode = "interior"

    def _converged(max_div_val, max_div_interior_val, mean_div_val, mean_thr):
        if mode == "global":
            max_ok = max_div_val <= max_div_threshold
        elif mode == "both":
            max_ok = (max_div_interior_val <= max_div_threshold) and (max_div_val <= max_div_threshold)
        else:  # interior
            max_ok = max_div_interior_val <= max_div_threshold
        return max_ok and (mean_div_val <= mean_thr)

    if _converged(max_div, max_div_interior, mean_div, mean_div_threshold):
        return U, info

    if correction_solver is None:
        raise ValueError("ppe_solve_staggered requires correction_solver")
    info["applied"] = True

    # Keep initial velocity for diagnostics
    U_before = U.copy() if isinstance(U, np.ndarray) else np.array(U)

    # Use predictor face state directly when available; falling back to
    # to_staggered(U) loses MAC information at boundaries/corners.
    if u_face_in is not None and v_face_in is not None:
        u_face = jnp.array(u_face_in) if not isinstance(u_face_in, jnp.ndarray) else u_face_in
        v_face = jnp.array(v_face_in) if not isinstance(v_face_in, jnp.ndarray) else v_face_in
        Uj = to_collocated(u_face, v_face)
    else:
        Uj = jnp.array(U) if not isinstance(U, jnp.ndarray) else U
        u_face, v_face = to_staggered(Uj)
    rho_cc = jnp.maximum(jax_calculate_density(phi, rho1, rho2), 1e-6)
    inv_rho_cc = 1.0 / rho_cc
    Nx_rho, Ny_rho = rho_cc.shape
    inv_rho_u_face = jnp.zeros((Nx_rho + 1, Ny_rho), dtype=rho_cc.dtype)
    inv_rho_v_face = jnp.zeros((Nx_rho, Ny_rho + 1), dtype=rho_cc.dtype)

    # Harmonic-like face averaging via averaging inverse density.
    inv_rho_u_face = inv_rho_u_face.at[1:Nx_rho, :].set(
        0.5 * (inv_rho_cc[1:, :] + inv_rho_cc[:-1, :])
    )
    inv_rho_u_face = inv_rho_u_face.at[0, :].set(inv_rho_cc[0, :])
    inv_rho_u_face = inv_rho_u_face.at[Nx_rho, :].set(inv_rho_cc[-1, :])

    inv_rho_v_face = inv_rho_v_face.at[:, 1:Ny_rho].set(
        0.5 * (inv_rho_cc[:, 1:] + inv_rho_cc[:, :-1])
    )
    inv_rho_v_face = inv_rho_v_face.at[:, 0].set(inv_rho_cc[:, 0])
    inv_rho_v_face = inv_rho_v_face.at[:, Ny_rho].set(inv_rho_cc[:, -1])
    projection_inv_rho_u_face = inv_rho_u_face
    projection_inv_rho_v_face = inv_rho_v_face

    use_variable_density_projection = True
    varcoef_matrix = None
    varcoef_all_neumann = False
    varcoef_gauge_ij = None
    varcoef_x0 = None
    if use_variable_density_projection and not use_terrain_projection:
        ppe_bcs_effective = ppe_bcs or {
            "left": "neumann",
            "right": "neumann",
            "bottom": "neumann",
            "top": "neumann",
        }
        varcoef_matrix, varcoef_all_neumann, varcoef_gauge_ij = _build_variable_coefficient_ppe_matrix(
            np.array(inv_rho_u_face),
            np.array(inv_rho_v_face),
            dx,
            dy,
            ppe_bcs_effective,
        )
        info["variable_density_projection"] = True
    else:
        info["variable_density_projection"] = False
        if use_terrain_projection:
            info["terrain_projection"] = True
            info["variable_density_projection"] = True

    p_corr = None
    min_iterations = max(int(min_iterations), 1)
    for it in range(max_iterations):
        u_face_old, v_face_old = u_face, v_face

        # Apply BCs in face space so divergence/RHS are consistent.
        if velocity_bc_manager is not None and hasattr(velocity_bc_manager, "apply_to_faces"):
            u_face, v_face = velocity_bc_manager.apply_to_faces(
                u_face, v_face, dx, dy, psi=psi, geometry=geometry, phi=phi
            )
        elif velocity_bc_manager is not None:
            raise ValueError("Staggered-only PPE requires face BC manager with apply_to_faces().")
        U_cc = to_collocated(u_face, v_face)
        div = _mac_divergence_geometry(u_face, v_face, dx, dy, geometry)
        rhs = (1.0 / dt) * div

        rhs_np = np.array(rhs)

        # Mean-zero RHS only when all BCs are Neumann (otherwise pressure is unique).
        if ppe_bcs is not None and all(bc == "neumann" for bc in ppe_bcs.values()):
            rhs_np = rhs_np - np.mean(rhs_np)

        # Dirichlet pressure (e.g. outlet p' = 0): RHS at that boundary is the prescribed value.
        if ppe_bcs is not None:
            Nx, Ny = rhs_np.shape
            if ppe_bcs.get("left") == "dirichlet":
                rhs_np[0, :] = 0.0
            if ppe_bcs.get("right") == "dirichlet":
                rhs_np[Nx - 1, :] = 0.0
            if ppe_bcs.get("bottom") == "dirichlet":
                rhs_np[:, 0] = 0.0
            if ppe_bcs.get("top") == "dirichlet":
                rhs_np[:, Ny - 1] = 0.0

        # Sanitize RHS: NaN/Inf can come from bad velocity; BiCGSTAB can also break when
        # RHS is all zeros or nearly zero (e.g. v=0 everywhere, or div≈0) leading to omega=0.
        if not np.all(np.isfinite(rhs_np)):
            rhs_np = np.nan_to_num(rhs_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Solve for pressure correction. Flat runs use an explicit sparse
        # variable-density operator. Terrain runs use a matrix-free MAC
        # operator that exactly matches _mac_divergence_geometry(_terrain_grad).
        if use_variable_density_projection and not use_terrain_projection:
            rhs_var = rhs_np.copy()
            if varcoef_all_neumann and varcoef_gauge_ij is not None:
                gi, gj = varcoef_gauge_ij
                rhs_var[gi, gj] = 0.0
            p_corr = _solve_variable_coefficient_ppe(
                varcoef_matrix,
                rhs_var,
                x0=varcoef_x0,
                tol=getattr(correction_solver, "tol", 1e-8),
                maxiter=getattr(correction_solver, "maxiter", 2000),
            )
            varcoef_x0 = p_corr
        elif use_terrain_projection:
            p_corr = _solve_terrain_mac_ppe(
                rhs_np,
                np.array(projection_inv_rho_u_face),
                np.array(projection_inv_rho_v_face),
                np.array(geometry.f_1_grid),
                dx,
                dy,
                ppe_bcs,
                x0=varcoef_x0,
                tol=getattr(correction_solver, "tol", 1e-8),
                maxiter=getattr(correction_solver, "maxiter", 2000),
            )
            varcoef_x0 = p_corr
        else:
            try:
                p_corr = correction_solver.solve(rhs_np, x0=varcoef_x0)
            except TypeError:
                p_corr = correction_solver.solve(rhs_np)
            varcoef_x0 = p_corr

        # Compatibility BC: make pressure-correction gradient consistent with Dirichlet
        # velocity boundaries. For variable-density projection use rho-aware scaling.
        if velocity_bc_manager is not None and not use_terrain_projection:
            p_corr = _apply_pressure_compatibility_from_dirichlet_velocity(
                p_corr=p_corr,
                u_face=u_face,
                v_face=v_face,
                velocity_bc_manager=velocity_bc_manager,
                dx=dx,
                dy=dy,
                dt=dt,
                inv_rho_u_face=np.array(inv_rho_u_face),
                inv_rho_v_face=np.array(inv_rho_v_face),
                ppe_bcs=ppe_bcs,
            )


        # If solver returned NaN/Inf (e.g. BiCGSTAB breakdown), do not apply correction
        if not np.all(np.isfinite(p_corr)):
            u_face, v_face = u_face_old, v_face_old
            Uj = to_collocated(u_face, v_face)
            if velocity_bc_manager is not None and hasattr(velocity_bc_manager, "apply_to_faces"):
                u_face, v_face = velocity_bc_manager.apply_to_faces(
                    u_face, v_face, dx, dy, psi=psi, geometry=geometry, phi=phi
                )
                Uj = to_collocated(u_face, v_face)
            elif velocity_bc_manager is not None:
                raise ValueError("Staggered-only PPE requires face BC manager with apply_to_faces().")
            import sys
            sys.stdout.write(
                f"\n⚠️  PPE (staggered): iter {it} solver returned NaN/Inf; stopping without applying correction.\n"
            )
            sys.stdout.flush()
            break

        # Correct in face space on MAC grid with variable density.
        p_corr_jax = jnp.array(p_corr)
        dpdx_face, dpdy_face = _terrain_grad_p_to_faces(p_corr_jax, dx, dy, geometry)
        u_face = u_face - under_relaxation * dt * (projection_inv_rho_u_face * dpdx_face)
        v_face = v_face - under_relaxation * dt * (projection_inv_rho_v_face * dpdy_face)
        Uj = to_collocated(u_face, v_face)
        # Re-apply velocity BCs in face space.
        if velocity_bc_manager is not None and hasattr(velocity_bc_manager, "apply_to_faces"):
            u_face, v_face = velocity_bc_manager.apply_to_faces(
                u_face, v_face, dx, dy, psi=psi, geometry=geometry, phi=phi
            )
            Uj = to_collocated(u_face, v_face)
        elif velocity_bc_manager is not None:
            raise ValueError("Staggered-only PPE requires face BC manager with apply_to_faces().")

        div_face = _mac_divergence_geometry(u_face, v_face, dx, dy, geometry)
        div_np = np.array(div_face)
        interior = div_np[1:-1, 1:-1]
        max_div = float(np.max(np.abs(div_np)))
        mean_div = float(np.mean(np.abs(div_np)))
        max_div_interior = float(np.max(np.abs(interior))) if interior.size > 0 else max_div
        # Progress output every 10 steps (handle nan for display)
        if it % 10 == 0 or it < 5:
            import sys
            md = max_div_interior if np.isfinite(max_div_interior) else np.nan
            ad = mean_div if np.isfinite(mean_div) else np.nan
            sys.stdout.write(
                f"\rPPE (staggered): iter {it}, max_div_interior: {md:.6f}, mean_div: {ad:.6f}"
            )
            sys.stdout.flush()
        if (it + 1) >= min_iterations and _converged(max_div, max_div_interior, mean_div, mean_div_threshold):
            info["iterations"] = it + 1
            break
        info["iterations"] = it + 1

    import sys
    sys.stdout.write("\n")
    sys.stdout.flush()
    # Final stats (BCs were applied every iteration above)
    div_face = _mac_divergence_geometry(u_face, v_face, dx, dy, geometry)
    div_np = np.array(div_face)
    interior = div_np[1:-1, 1:-1]
    max_div = float(np.max(np.abs(div_np)))
    mean_div = float(np.mean(np.abs(div_np)))
    max_div_interior = float(np.max(np.abs(interior))) if interior.size > 0 else max_div
    info["div_after_max"] = float(max_div)
    info["div_after_mean"] = float(mean_div)
    info["div_after_max_interior"] = float(max_div_interior)
    if p_corr is not None:
        info["p_corr_out"] = np.array(p_corr)
    # Preserve converged face fields for callers running a staggered pipeline.
    # Reconstructing faces from collocated U can re-introduce divergence.
    info["u_face_out"] = u_face
    info["v_face_out"] = v_face

    # Optional diagnostics dump: BEFORE vs AFTER (cycle-level)
    if debug_output_dir is not None and debug_step is not None:
        try:
            dump_ppe_diagnostics(
                U_before,
                np.array(Uj),
                dx,
                dy,
                debug_step,
                -1,
                debug_output_dir,
            )
        except Exception as e:
            print(f"⚠️  Failed to dump staggered PPE diagnostics: {e}")

    return Uj, info

# Backward compatibility alias
ppe = ppe_solve
ppe_global = ppe_solve


# ============= Helper Functions =============

def _apply_velocity_bcs(U, dx, dy, velocity_bc_manager, psi, geometry):
    """Apply velocity boundary conditions."""
    if velocity_bc_manager is not None:
        U = velocity_bc_manager.apply_boundary_conditions(U, dx, dy, use_jax=True, psi=psi, geometry=geometry)
    else:
        from boundary_conditions.velocity_bc import jax_apply_velocity_do_nothing_bc
        U = jax_apply_velocity_do_nothing_bc(U)
        # Mask ice regions
        if psi is not None:
            psi_jax = jnp.array(psi) if not isinstance(psi, jnp.ndarray) else psi
            ice_mask = psi_jax > 0.0
            U = U.at[..., 0].set(jnp.where(ice_mask, 0.0, U[..., 0]))
            U = U.at[..., 1].set(jnp.where(ice_mask, 0.0, U[..., 1]))
    return U


def _build_variable_coefficient_ppe_matrix(inv_rho_u_face, inv_rho_v_face, dx, dy, ppe_bcs):
    """Build matrix for div(beta grad p)=rhs on cell centers, beta=1/rho on faces."""
    Nx = inv_rho_v_face.shape[0]
    Ny = inv_rho_u_face.shape[1]
    dx2 = dx * dx
    dy2 = dy * dy

    bcs = ppe_bcs or {}
    left_bc = bcs.get("left", "neumann")
    right_bc = bcs.get("right", "neumann")
    bottom_bc = bcs.get("bottom", "neumann")
    top_bc = bcs.get("top", "neumann")
    all_neumann = all(bc == "neumann" for bc in (left_bc, right_bc, bottom_bc, top_bc))
    gauge_ij = None

    n = Nx * Ny
    A = scipy.sparse.lil_matrix((n, n), dtype=np.float64)

    def idx(i, j):
        return j * Nx + i

    for i in range(Nx):
        for j in range(Ny):
            on_left = i == 0
            on_right = i == Nx - 1
            on_bottom = j == 0
            on_top = j == Ny - 1
            is_dirichlet = (
                (on_left and left_bc == "dirichlet")
                or (on_right and right_bc == "dirichlet")
                or (on_bottom and bottom_bc == "dirichlet")
                or (on_top and top_bc == "dirichlet")
            )
            k = idx(i, j)
            if is_dirichlet:
                A[k, k] = 1.0
                continue

            beta_w = 0.0 if on_left else float(inv_rho_u_face[i, j])
            beta_e = 0.0 if on_right else float(inv_rho_u_face[i + 1, j])
            beta_s = 0.0 if on_bottom else float(inv_rho_v_face[i, j])
            beta_n = 0.0 if on_top else float(inv_rho_v_face[i, j + 1])

            diag = 0.0
            if i > 0:
                A[k, idx(i - 1, j)] = beta_w / dx2
                diag -= beta_w / dx2
            if i < Nx - 1:
                A[k, idx(i + 1, j)] = beta_e / dx2
                diag -= beta_e / dx2
            if j > 0:
                A[k, idx(i, j - 1)] = beta_s / dy2
                diag -= beta_s / dy2
            if j < Ny - 1:
                A[k, idx(i, j + 1)] = beta_n / dy2
                diag -= beta_n / dy2

            A[k, k] = diag

    if all_neumann:
        # Fix pressure gauge for pure-Neumann system at the domain center
        # to reduce corner bias in strong variable-density runs.
        gauge_ij = (Nx // 2, Ny // 2)
        k0 = idx(gauge_ij[0], gauge_ij[1])
        A[k0, :] = 0.0
        A[k0, k0] = 1.0

    return A.tocsr(), all_neumann, gauge_ij


def _solve_variable_coefficient_ppe(A, rhs, x0=None, tol=1e-8, maxiter=2000):
    """Solve variable-coefficient PPE matrix system with robust fallback."""
    rhs_t = np.asarray(rhs.T, dtype=np.float64)
    rhs_flat = rhs_t.flatten()

    x0_flat = None
    if x0 is not None:
        x0_flat = np.asarray(x0.T, dtype=np.float64).flatten()

    try:
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, rtol=tol, atol=0.0, maxiter=maxiter
        )
    except TypeError:
        # SciPy API compatibility on older versions.
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, tol=tol, maxiter=maxiter
        )

    if info != 0 or not np.all(np.isfinite(sol)):
        try:
            sol = scipy.sparse.linalg.spsolve(A, rhs_flat)
        except Exception:
            sol = np.zeros_like(rhs_flat)

    if not np.all(np.isfinite(sol)):
        sol = np.zeros_like(rhs_flat)

    return sol.reshape(rhs_t.shape).T


def _terrain_grad_p_to_faces_np(p, dx, dy, f_1_grid):
    """NumPy version of _terrain_grad_p_to_faces for matrix-free PPE solves."""
    Nx, Ny = p.shape
    dpdx = np.zeros((Nx + 1, Ny), dtype=np.float64)
    dpdx[1:Nx, :] = (p[1:, :] - p[:-1, :]) / dx

    dpeta = np.zeros((Nx, Ny + 1), dtype=np.float64)
    dpeta[:, 1:Ny] = (p[:, 1:] - p[:, :-1]) / dy

    p_eta_cc = np.zeros_like(p, dtype=np.float64)
    if Ny > 1:
        p_eta_cc[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dy)
        p_eta_cc[:, 0] = (p[:, 1] - p[:, 0]) / dy
        p_eta_cc[:, -1] = (p[:, -1] - p[:, -2]) / dy

    p_eta_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
    p_eta_u[1:Nx, :] = 0.5 * (p_eta_cc[1:, :] + p_eta_cc[:-1, :])
    p_eta_u[0, :] = p_eta_cc[0, :]
    p_eta_u[Nx, :] = p_eta_cc[-1, :]

    f1 = np.asarray(f_1_grid, dtype=np.float64)
    f1_u = np.zeros((Nx + 1, Ny), dtype=np.float64)
    f1_u[1:Nx, :] = 0.5 * (f1[1:, :] + f1[:-1, :])
    f1_u[0, :] = f1[0, :]
    f1_u[Nx, :] = f1[-1, :]

    return dpdx - f1_u * p_eta_u, dpeta


def _terrain_mac_divergence_np(u_face, v_face, dx, dy, f_1_grid):
    """NumPy version of terrain MAC divergence."""
    div = (u_face[1:, :] - u_face[:-1, :]) / dx + (v_face[:, 1:] - v_face[:, :-1]) / dy
    u_center = 0.5 * (u_face[1:, :] + u_face[:-1, :])
    u_eta = np.zeros_like(u_center, dtype=np.float64)
    if u_center.shape[1] > 1:
        u_eta[:, 1:-1] = (u_center[:, 2:] - u_center[:, :-2]) / (2.0 * dy)
        u_eta[:, 0] = (u_center[:, 1] - u_center[:, 0]) / dy
        u_eta[:, -1] = (u_center[:, -1] - u_center[:, -2]) / dy
    return div - np.asarray(f_1_grid, dtype=np.float64) * u_eta


def _solve_terrain_mac_ppe(
    rhs,
    inv_rho_u_face,
    inv_rho_v_face,
    f_1_grid,
    dx,
    dy,
    ppe_bcs,
    x0=None,
    tol=1e-8,
    maxiter=2000,
):
    """Solve D_terrain(beta G_terrain p)=rhs with the same MAC stencils used for correction."""
    rhs = np.asarray(rhs, dtype=np.float64)
    Nx, Ny = rhs.shape
    bcs = ppe_bcs or {}
    left_bc = bcs.get("left", "neumann")
    right_bc = bcs.get("right", "neumann")
    bottom_bc = bcs.get("bottom", "neumann")
    top_bc = bcs.get("top", "neumann")
    all_neumann = all(bc == "neumann" for bc in (left_bc, right_bc, bottom_bc, top_bc))
    gauge_ij = (Nx // 2, Ny // 2) if all_neumann else None

    rhs_work = rhs.copy()
    if all_neumann:
        rhs_work -= np.mean(rhs_work)
        rhs_work[gauge_ij] = 0.0

    dirichlet_mask = np.zeros((Nx, Ny), dtype=bool)
    if left_bc == "dirichlet":
        dirichlet_mask[0, :] = True
    if right_bc == "dirichlet":
        dirichlet_mask[-1, :] = True
    if bottom_bc == "dirichlet":
        dirichlet_mask[:, 0] = True
    if top_bc == "dirichlet":
        dirichlet_mask[:, -1] = True
    rhs_work[dirichlet_mask] = 0.0

    beta_u = np.asarray(inv_rho_u_face, dtype=np.float64)
    beta_v = np.asarray(inv_rho_v_face, dtype=np.float64)
    f1 = np.asarray(f_1_grid, dtype=np.float64)

    def _pack(a):
        return np.asarray(a, dtype=np.float64).T.ravel()

    def _unpack(vec):
        return np.asarray(vec, dtype=np.float64).reshape((Ny, Nx)).T

    def _matvec(vec):
        p = _unpack(vec)
        gx, gy = _terrain_grad_p_to_faces_np(p, dx, dy, f1)
        out = _terrain_mac_divergence_np(beta_u * gx, beta_v * gy, dx, dy, f1)
        out[dirichlet_mask] = p[dirichlet_mask]
        if gauge_ij is not None:
            out[gauge_ij] = p[gauge_ij]
        return _pack(out)

    n = Nx * Ny
    A = scipy.sparse.linalg.LinearOperator((n, n), matvec=_matvec, dtype=np.float64)
    x0_flat = _pack(x0) if x0 is not None else None
    rhs_flat = _pack(rhs_work)
    try:
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, rtol=tol, atol=0.0, maxiter=maxiter
        )
    except TypeError:
        sol, info = scipy.sparse.linalg.bicgstab(
            A, rhs_flat, x0=x0_flat, tol=tol, maxiter=maxiter
        )

    if info != 0 or not np.all(np.isfinite(sol)):
        sol = scipy.sparse.linalg.lgmres(A, rhs_flat, x0=x0_flat, rtol=tol, maxiter=maxiter)[0]
    if not np.all(np.isfinite(sol)):
        sol = np.zeros_like(rhs_flat)
    return _unpack(sol)


def _apply_pressure_compatibility_from_dirichlet_velocity(
    p_corr,
    u_face,
    v_face,
    velocity_bc_manager,
    dx,
    dy,
    dt,
    inv_rho_u_face=None,
    inv_rho_v_face=None,
    ppe_bcs=None,
):
    """Apply rho-aware pressure compatibility on Dirichlet velocity boundaries.

    Important: if PPE matrix already enforces Dirichlet on a side, skip compatibility
    overwrite there to keep p_corr consistent with the solved linear system.
    """
    vel_bc_cfg = velocity_bc_manager.config.get("boundary_conditions", {}).get("velocity", {})
    dirichlet_values = vel_bc_cfg.get("dirichlet_values", {})

    p = np.array(p_corr, copy=True)
    u = np.array(u_face)
    v = np.array(v_face)
    Nx, Ny = p.shape

    inv_u = np.array(inv_rho_u_face) if inv_rho_u_face is not None else None
    inv_v = np.array(inv_rho_v_face) if inv_rho_v_face is not None else None

    eps_beta = 1e-12

    def _u_target(side):
        if side == "left":
            prof = velocity_bc_manager.get_inlet_profile(Ny, dy)
            if prof is not None:
                return np.asarray(prof, dtype=np.float64)
        dv = dirichlet_values.get(side, {"u": 0.0, "v": 0.0})
        if isinstance(dv, dict):
            return np.full(Ny, float(dv.get("u", 0.0)), dtype=np.float64)
        return np.zeros(Ny, dtype=np.float64)

    def _v_target(side):
        dv = dirichlet_values.get(side, {"u": 0.0, "v": 0.0})
        if isinstance(dv, dict):
            return np.full(Nx, float(dv.get("v", 0.0)), dtype=np.float64)
        return np.zeros(Nx, dtype=np.float64)

    if (
        vel_bc_cfg.get("left") == "dirichlet"
        and Nx >= 2
        and (ppe_bcs is None or ppe_bcs.get("left", "neumann") != "dirichlet")
    ):
        u_star = np.asarray(u[1, :], dtype=np.float64)
        u_t = _u_target("left")
        beta = np.asarray(inv_u[1, :], dtype=np.float64) if inv_u is not None else 1.0
        term = (dx / dt) * (u_star - u_t) / np.maximum(beta, eps_beta)
        p[0, :] = p[1, :] - term

    if (
        vel_bc_cfg.get("right") == "dirichlet"
        and Nx >= 2
        and (ppe_bcs is None or ppe_bcs.get("right", "neumann") != "dirichlet")
    ):
        u_star = np.asarray(u[Nx - 1, :], dtype=np.float64)
        u_t = _u_target("right")
        beta = np.asarray(inv_u[Nx - 1, :], dtype=np.float64) if inv_u is not None else 1.0
        term = (dx / dt) * (u_star - u_t) / np.maximum(beta, eps_beta)
        p[Nx - 1, :] = p[Nx - 2, :] + term

    if (
        vel_bc_cfg.get("bottom") == "dirichlet"
        and Ny >= 2
        and (ppe_bcs is None or ppe_bcs.get("bottom", "neumann") != "dirichlet")
    ):
        v_star = np.asarray(v[:, 1], dtype=np.float64)
        v_t = _v_target("bottom")
        beta = np.asarray(inv_v[:, 1], dtype=np.float64) if inv_v is not None else 1.0
        term = (dy / dt) * (v_star - v_t) / np.maximum(beta, eps_beta)
        p[:, 0] = p[:, 1] - term

    if (
        vel_bc_cfg.get("top") == "dirichlet"
        and Ny >= 2
        and (ppe_bcs is None or ppe_bcs.get("top", "neumann") != "dirichlet")
    ):
        v_star = np.asarray(v[:, Ny - 1], dtype=np.float64)
        v_t = _v_target("top")
        beta = np.asarray(inv_v[:, Ny - 1], dtype=np.float64) if inv_v is not None else 1.0
        term = (dy / dt) * (v_star - v_t) / np.maximum(beta, eps_beta)
        p[:, Ny - 1] = p[:, Ny - 2] + term

    return p
