"""Chemical potential boundary conditions (Cahn-Hilliard equation)."""

import jax.numpy as jnp
from jax import jit
from boundary_conditions.base_bc import BaseBoundaryCondition, BCType


BC_NEUMANN = 0
BC_DIRICHLET = 1
BC_PERIODIC = 2


class ChemicalPotentialBoundaryConditions(BaseBoundaryCondition):
    """Chemical potential BC - typically zero flux (Neumann) everywhere."""
    
    def __init__(self, config=None):
        super().__init__(config, "chemical_potential")
        # Default to zero flux at all boundaries
        if not self.config.get("boundary_conditions", {}).get("chemical_potential"):
            self.bc_raw = {"top": "zero_flux", "bottom": "zero_flux", "left": "zero_flux", "right": "zero_flux"}
            self.bc_types = {b: BCType.NEUMANN for b in self.bc_raw}
    
    def apply(self, mu_c, dx: float, dy: float, **kwargs):
        """Apply chemical potential BCs. mu_c shape: (Nx, Ny)."""
        return self.apply_standard_scalar(mu_c, dx, dy)

    def jax_metadata(self):
        """Return JAX-friendly boundary codes and scalar values."""
        codes = []
        values = []
        for boundary in ("top", "bottom", "left", "right"):
            bc = self.bc_types[boundary]
            if bc == BCType.DIRICHLET:
                code = BC_DIRICHLET
            elif bc == BCType.PERIODIC:
                code = BC_PERIODIC
            else:
                code = BC_NEUMANN
            raw_value = self.dirichlet_values.get(boundary, 0.0)
            value = float(raw_value) if raw_value is not None else 0.0
            codes.append(code)
            values.append(value)
        return tuple(codes), tuple(values)


@jit
def _apply_top_bc(mu_c, bc_code, value):
    neumann = mu_c.at[:, -1].set(mu_c[:, -2])
    dirichlet = mu_c.at[:, -1].set(value)
    periodic = mu_c.at[:, -1].set(mu_c[:, 1])
    return jnp.where(
        bc_code == BC_DIRICHLET,
        dirichlet,
        jnp.where(bc_code == BC_PERIODIC, periodic, neumann),
    )


@jit
def _apply_bottom_bc(mu_c, bc_code, value):
    neumann = mu_c.at[:, 0].set(mu_c[:, 1])
    dirichlet = mu_c.at[:, 0].set(value)
    periodic = mu_c.at[:, 0].set(mu_c[:, -2])
    return jnp.where(
        bc_code == BC_DIRICHLET,
        dirichlet,
        jnp.where(bc_code == BC_PERIODIC, periodic, neumann),
    )


@jit
def _apply_left_bc(mu_c, bc_code, value):
    neumann = mu_c.at[0, :].set(mu_c[1, :])
    dirichlet = mu_c.at[0, :].set(value)
    periodic = mu_c.at[0, :].set(mu_c[-2, :])
    return jnp.where(
        bc_code == BC_DIRICHLET,
        dirichlet,
        jnp.where(bc_code == BC_PERIODIC, periodic, neumann),
    )


@jit
def _apply_right_bc(mu_c, bc_code, value):
    neumann = mu_c.at[-1, :].set(mu_c[-2, :])
    dirichlet = mu_c.at[-1, :].set(value)
    periodic = mu_c.at[-1, :].set(mu_c[1, :])
    return jnp.where(
        bc_code == BC_DIRICHLET,
        dirichlet,
        jnp.where(bc_code == BC_PERIODIC, periodic, neumann),
    )


@jit
def jax_apply_chemical_potential_bc(
    mu_c,
    dx,
    dy,
    top_bc=BC_NEUMANN,
    bottom_bc=BC_NEUMANN,
    left_bc=BC_NEUMANN,
    right_bc=BC_NEUMANN,
    top_value=0.0,
    bottom_value=0.0,
    left_value=0.0,
    right_value=0.0,
):
    """Apply configured scalar BCs to the chemical potential field."""
    del dx, dy
    mu_c = _apply_top_bc(mu_c, top_bc, top_value)
    mu_c = _apply_bottom_bc(mu_c, bottom_bc, bottom_value)
    mu_c = _apply_left_bc(mu_c, left_bc, left_value)
    mu_c = _apply_right_bc(mu_c, right_bc, right_value)
    return mu_c


@jit
def jax_apply_chemical_potential_zero_flux_bc(mu_c, dx, dy):
    """JIT-compiled zero flux BC for chemical potential."""
    return jax_apply_chemical_potential_bc(mu_c, dx, dy)
