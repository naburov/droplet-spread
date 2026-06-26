"""Factory helpers to construct SimulationState from config."""

import numpy as np
import jax.numpy as jnp

from simulation.geometry import Geometry
from simulation.state.specs import GridSpec, PhysicalParams, FeatureFlags, SimulationContext
from simulation.state.bundles import SolverBundle, BCBundle
from simulation.state.core import SimulationState
from simulation.initial_conditions import (
    initialize_phase,
    initialize_phase_two_droplets_touching,
    initialize_ice_phase,
    initialize_temperature,
)
from physics.phase_field import PhaseFieldSolverSimple, PhaseFieldSolverGhostCell
from physics.fluid_dynamics import FluidDynamicsSolver
from physics.surface_tension import SurfaceTensionSolver
from physics.pressure import PressureSolver, compute_hydrostatic_pressure
from physics.properties import jax_calculate_density
from solvers.sparse_solver import SparseSolverWrapper
from boundary_conditions.velocity_bc import VelocityBoundaryConditions
from boundary_conditions.pressure_bc import PressureBoundaryConditions
from boundary_conditions.phase_field_bc import PhaseFieldBoundaryConditions


def _initialize_pressure_field(Nx, Ny, dx, dy, config, rho1, rho2, g, Fr, atm_pressure, include_gravity, phi):
    """Initialize pressure field with hydrostatic and optional inlet-outlet gradient."""
    P = jnp.full((Nx, Ny), atm_pressure)

    if include_gravity:
        rho = jax_calculate_density(phi, rho1, rho2)
        p_hydro = compute_hydrostatic_pressure(rho, g, dy, Fr, atm_pressure)
        P = P + p_hydro

    bc_config = config.get("boundary_conditions", {})
    pressure_bc = bc_config.get("pressure", {})
    velocity_bc = bc_config.get("velocity", {})

    left_vel_type = velocity_bc.get("left", {})
    if isinstance(left_vel_type, dict):
        left_vel_type = left_vel_type.get("type", "")
    else:
        left_vel_type = str(left_vel_type) if left_vel_type else ""

    right_p_type = pressure_bc.get("right", {})
    if isinstance(right_p_type, dict):
        right_p_type = right_p_type.get("type", "")
    else:
        right_p_type = str(right_p_type) if right_p_type else ""

    is_flow_simulation = (left_vel_type == "dirichlet" and right_p_type == "dirichlet")
    if is_flow_simulation:
        p_outlet = pressure_bc.get("dirichlet_values", {}).get("right", 0.0)
        if isinstance(p_outlet, dict):
            p_outlet = p_outlet.get("value", 0.0)

        x_coords = jnp.linspace(0, 1, Nx)
        x_grid = x_coords[:, jnp.newaxis]
        ic = config.get("initial_conditions", {})
        pressure_drop = ic.get("pressure_drop", 0.01)

        p_dynamic = p_outlet + pressure_drop * (1.0 - x_grid)
        p_dynamic = jnp.broadcast_to(p_dynamic, (Nx, Ny))
        P = P + p_dynamic

    return P


def _derive_and_apply_ppe_bcs(config, ppe_settings, correction_solver):
    """Resolve PPE BCs and apply them to correction solver."""
    from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config, explain_ppe_bc_derivation

    ppe_bc_settings_explicit = ppe_settings.get("boundary_conditions")
    if ppe_bc_settings_explicit is not None:
        ppe_bc_settings = ppe_bc_settings_explicit
        ppe_bc_settings_derived = derive_ppe_bcs_from_config(config)

        mismatches = []
        for boundary in ["top", "bottom", "left", "right"]:
            if ppe_bc_settings.get(boundary) != ppe_bc_settings_derived.get(boundary):
                mismatches.append(
                    f"{boundary}: explicit={ppe_bc_settings.get(boundary)}, derived={ppe_bc_settings_derived.get(boundary)}"
                )

        if mismatches and not config.get("_bc_check", {}).get("ppe_explicit_ok"):
            print("⚠️  Warning: PPE BCs in config don't match derived BCs from velocity BCs:")
            for m in mismatches:
                print(f"   {m}")
            print("   Consider removing explicit PPE BCs to use automatic derivation.")
    else:
        ppe_bc_settings = derive_ppe_bcs_from_config(config)
        velocity_bcs = config.get("boundary_conditions", {}).get("velocity", {})
        explain_ppe_bc_derivation(velocity_bcs, ppe_bc_settings)

    correction_solver.set_top_boundary_condition(ppe_bc_settings["top"])
    correction_solver.set_bottom_boundary_condition(ppe_bc_settings["bottom"])
    correction_solver.set_left_boundary_condition(ppe_bc_settings["left"])
    correction_solver.set_right_boundary_condition(ppe_bc_settings["right"])


def create_state_from_config(config, restart_from=None):
    """Create initial simulation state from configuration."""
    rho1 = config["physical_params"]["rho1"]
    rho2 = config["physical_params"]["rho2"]
    Re1 = config["physical_params"]["Re1"]
    Re2 = config["physical_params"]["Re2"]
    We1 = config["physical_params"]["We1"]
    We2 = config["physical_params"]["We2"]
    Pe = config["physical_params"]["Pe"]
    epsilon = config["physical_params"]["epsilon"]
    contact_angle = config["physical_params"]["contact_angle"]
    include_gravity = config["physical_params"]["include_gravity"]
    g = config["physical_params"]["g"]
    atm_pressure = config["physical_params"]["atm_pressure"]
    Fr = config["physical_params"]["Fr"]

    Lx, Ly = config["grid_params"]["Lx"], config["grid_params"]["Ly"]
    Nx, Ny = config["grid_params"]["Nx"], config["grid_params"]["Ny"]
    dx, dy = Lx / Nx, Ly / Ny
    dt_initial = config["time_params"]["dt_initial"]
    include_ice_water = config.get("physical_params", {}).get("include_ice_water_transition", False)
    ic = config.get("initial_conditions", {})

    ic_type = ic.get("type", "droplet")
    if ic_type == "uniform_air":
        phi_value = ic.get("phi_value", 1.0)
        phi = np.full((Nx, Ny), phi_value, dtype=np.float32)
    elif ic_type == "uniform_liquid":
        phi_value = ic.get("phi_value", -1.0)
        phi = np.full((Nx, Ny), phi_value, dtype=np.float32)
    elif ic_type == "two_droplets_touching":
        radius = ic.get("droplet_radius", 0.2)
        phi = initialize_phase_two_droplets_touching(Nx, Ny, radius, epsilon=epsilon, config=config)
    else:
        radius = ic.get("droplet_radius", 0.2)
        phi = initialize_phase(Nx, Ny, radius, epsilon=epsilon, config=config)
    phi = jnp.array(phi)

    U = jnp.zeros((Nx, Ny, 2))
    initial_velocity = ic.get("initial_velocity", None)
    if initial_velocity is not None:
        u0 = initial_velocity.get("u", 0.0)
        v0 = initial_velocity.get("v", 0.0)
        U = U.at[:, :, 0].add(u0)
        U = U.at[:, :, 1].add(v0)

    P = _initialize_pressure_field(
        Nx, Ny, dx, dy, config, rho1, rho2, g, Fr, atm_pressure, include_gravity, phi
    )

    if include_ice_water:
        psi = initialize_ice_phase(Nx, Ny, config, phi=phi)
        T = initialize_temperature(Nx, Ny, config)
        psi = jnp.array(psi)
        T = jnp.array(T)
    else:
        psi = jnp.zeros((Nx, Ny))
        T = jnp.zeros((Nx, Ny))

    geom_cfg = config.get("initial_conditions", {}).get("geometry", {}) or config.get("geometry", {})
    geom_type = geom_cfg.get("type", "flat")
    if geom_type == "tilted":
        angle_degrees = float(geom_cfg.get("degree", 10.0))
        origin = str(geom_cfg.get("origin", "bottom_left"))
        geometry = Geometry.tilted(Nx, Ny, dx, dy, Lx, Ly, angle_degrees=angle_degrees, origin=origin)
    elif geom_type == "hump":
        amplitude = float(geom_cfg.get("amplitude", 0.1))
        sigma = float(geom_cfg.get("sigma", 0.2))
        center_x = float(geom_cfg.get("center_x", Lx / 2.0))
        geometry = Geometry.hump(Nx, Ny, dx, amplitude, sigma, center_x, dy=dy)
    elif geom_type in ("groove", "sinusoidal_groove"):
        amplitude = float(geom_cfg.get("amplitude", 0.03))
        waves = float(geom_cfg.get("waves", geom_cfg.get("n_waves", 2.0)))
        offset = float(geom_cfg.get("offset", 0.0))
        phase = float(geom_cfg.get("phase", 0.0))
        geometry = Geometry.sinusoidal_groove(
            Nx, Ny, dx, Lx, amplitude, waves=waves, offset=offset, phase=phase
        )
    else:
        geometry = Geometry.flat(Nx, Ny)

    phase_solver_kind = config.get("solver_params", {}).get("phase_field_solver")
    if phase_solver_kind is None:
        phase_solver_kind = config.get("boundary_conditions", {}).get("phase_field", {}).get("contact_angle_method", "simple")
    phase_solver_kind = str(phase_solver_kind).lower()
    if phase_solver_kind == "ghost_cell":
        phase_solver = PhaseFieldSolverGhostCell(Pe, epsilon, contact_angle, config)
    elif phase_solver_kind == "simple":
        phase_solver = PhaseFieldSolverSimple(Pe, epsilon, contact_angle, config)
    else:
        raise ValueError(
            f"Unknown solver_params.phase_field_solver: {phase_solver_kind}. "
            "Supported values are 'simple' and 'ghost_cell'."
        )
    fluid_solver = FluidDynamicsSolver(rho1, rho2, Re1, Re2, Fr, g, config=config)
    surface_tension_params = config.get("physical_params", {}).get("surface_tension", {})
    smooth_curvature = surface_tension_params.get("smooth_curvature", True)
    smoothing_radius = surface_tension_params.get("smoothing_radius", 1)
    use_composition_field = surface_tension_params.get("use_composition_field", True)
    composition_force_scale = surface_tension_params.get("composition_force_scale", 1.0)
    weber_interpolation = surface_tension_params.get("weber_interpolation", "constant_liquid")
    apply_boundary_overwrite = surface_tension_params.get("apply_boundary_overwrite", True)
    force_form = surface_tension_params.get("force_form", "csf")
    potential_wall_laplacian = surface_tension_params.get("potential_wall_laplacian", "plain")
    potential_wall_energy_scale = surface_tension_params.get("potential_wall_energy_scale", 1.0)
    capillary_rhs_smoothing_radius = int(surface_tension_params.get("capillary_rhs_smoothing_radius", 1))

    # The potential force form must use the same free energy as the CH solve.
    ch_solver_params = config.get("solver_params", {})
    potential_params = {
        "phase_potential": ch_solver_params.get("phase_potential", "polynomial"),
        "phase_log_theta": ch_solver_params.get("phase_log_theta", 0.25),
        "phase_log_theta_c": ch_solver_params.get("phase_log_theta_c", 1.0),
        "phase_log_delta": ch_solver_params.get("phase_log_delta", 1e-6),
    }

    surface_tension_solver = SurfaceTensionSolver(
        epsilon,
        We1,
        We2,
        contact_angle,
        smooth_curvature=smooth_curvature,
        smoothing_radius=smoothing_radius,
        use_composition_field=use_composition_field,
        composition_force_scale=composition_force_scale,
        weber_interpolation=weber_interpolation,
        apply_boundary_overwrite=apply_boundary_overwrite,
        force_form=force_form,
        potential_params=potential_params,
        potential_wall_laplacian=potential_wall_laplacian,
        potential_wall_energy_scale=potential_wall_energy_scale,
    )
    solver_params = config.get("solver_params", {})
    physical_pressure_params = solver_params.get("physical_pressure", {})
    physical_pressure_bcs = physical_pressure_params.get("boundary_conditions")

    pressure_solver = PressureSolver(
        rho1,
        rho2,
        g,
        atm_pressure,
        Fr=Fr,
        include_gravity=include_gravity,
        capillary_rhs_smoothing_radius=capillary_rhs_smoothing_radius,
        pressure_bcs=physical_pressure_bcs,
    )

    default_pressure_solver = {
        "backend": "pyamg",
        "accel": "bicgstab",
        "tol": 0.05,
        "maxiter": 10000,
    }
    default_correction_solver = {
        "backend": "pyamg",
        "accel": "bicgstab",
        "tol": 0.05,
        "maxiter": 10000,
    }
    default_ppe = {
        "mean_div_threshold": 0.1,
        "max_div_threshold": 0.05,
        "max_iterations": 1000,
        "boundary_conditions": {
            "top": "neumann",
            "bottom": "neumann",
            "left": "neumann",
            "right": "neumann",
        },
    }

    pressure_solver_params = solver_params.get("pressure_solver", default_pressure_solver)
    correction_solver_params = solver_params.get("correction_solver", default_correction_solver)
    ppe_settings = solver_params.get("ppe", default_ppe)

    correction_solver = SparseSolverWrapper(
        Nx, Ny, dx, dy, correction_solver_params.get("backend", "pyamg"), correction_solver_params
    )
    pressure_linear_solver = SparseSolverWrapper(
        Nx, Ny, dx, dy, pressure_solver_params.get("backend", "pyamg"), pressure_solver_params
    )

    pressure_bc_settings = config["boundary_conditions"]["pressure"]
    _derive_and_apply_ppe_bcs(config, ppe_settings, correction_solver)

    if geometry.has_geometry:
        correction_solver.set_terrain(geometry.f_1_grid, geometry.f_2_grid)

    def map_bc_type(bc_type):
        return "dirichlet" if bc_type == "open" else bc_type

    pressure_linear_solver.set_top_boundary_condition(map_bc_type(pressure_bc_settings["top"]))
    pressure_linear_solver.set_bottom_boundary_condition(map_bc_type(pressure_bc_settings["bottom"]))
    pressure_linear_solver.set_left_boundary_condition(map_bc_type(pressure_bc_settings["left"]))
    pressure_linear_solver.set_right_boundary_condition(map_bc_type(pressure_bc_settings["right"]))
    correction_solver.create_sparse_matrix()
    pressure_linear_solver.create_sparse_matrix()

    velocity_bc = VelocityBoundaryConditions(config)
    pressure_bc = PressureBoundaryConditions(config)
    phase_field_bc = PhaseFieldBoundaryConditions(config)

    if restart_from is None:
        U_init = jnp.zeros((Nx, Ny, 2))
        phi = phase_field_bc.apply_boundary_conditions(phi, dx, dy, use_jax=True, geometry=geometry, U=U_init)

    velocity_layout_cfg = str(config.get("solver_params", {}).get("velocity_layout", "staggered")).lower()
    if velocity_layout_cfg != "staggered":
        raise ValueError(
            "Only staggered velocity layout is supported. "
            "Set solver_params.velocity_layout='staggered'."
        )
    velocity_layout = "staggered"

    u_face, v_face = None, None
    start_time = 0.0
    start_dt = dt_initial
    if restart_from is not None:
        from visualization.checkpointing import load_checkpoint

        checkpoint_data = load_checkpoint(restart_from)
        start_step = int(checkpoint_data["step"]) + 1
        phi = checkpoint_data["phi"]
        U = checkpoint_data["U"]
        P = checkpoint_data["P"]
        start_time = float(checkpoint_data.get("t", config.get("_resume", {}).get("time", 0.0)))
        start_dt = float(checkpoint_data.get("dt", config.get("_resume", {}).get("dt", dt_initial)))
        if include_ice_water:
            if "psi" in checkpoint_data:
                psi = checkpoint_data["psi"]
            if "T" in checkpoint_data:
                T = checkpoint_data["T"]
        if "u_face" in checkpoint_data:
            u_face = jnp.array(checkpoint_data["u_face"])
        if "v_face" in checkpoint_data:
            v_face = jnp.array(checkpoint_data["v_face"])
    else:
        start_step = 0

    context = SimulationContext(
        grid=GridSpec(Nx=Nx, Ny=Ny, dx=dx, dy=dy, Lx=Lx, Ly=Ly),
        physical=PhysicalParams(
            rho1=rho1,
            rho2=rho2,
            Re1=Re1,
            Re2=Re2,
            We1=We1,
            We2=We2,
            Pe=Pe,
            epsilon=epsilon,
            contact_angle=contact_angle,
            g=g,
            atm_pressure=atm_pressure,
            Fr=Fr,
        ),
        flags=FeatureFlags(include_ice_water=include_ice_water, include_gravity=include_gravity),
        geometry_type=geom_type,
    )
    solver_bundle = SolverBundle(
        phase_solver=phase_solver,
        fluid_solver=fluid_solver,
        surface_tension_solver=surface_tension_solver,
        pressure_solver=pressure_solver,
        correction_solver=correction_solver,
        pressure_linear_solver=pressure_linear_solver,
    )
    bc_bundle = BCBundle(velocity_bc=velocity_bc, pressure_bc=pressure_bc, phase_field_bc=phase_field_bc)

    state = SimulationState(
        phi=phi,
        U=U,
        P=P,
        psi=psi,
        T=T,
        geometry=geometry,
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        Lx=Lx,
        Ly=Ly,
        rho1=rho1,
        rho2=rho2,
        Re1=Re1,
        Re2=Re2,
        We1=We1,
        We2=We2,
        Pe=Pe,
        epsilon=epsilon,
        contact_angle=contact_angle,
        g=g,
        atm_pressure=atm_pressure,
        Fr=Fr,
        t=start_time,
        dt=start_dt,
        step=start_step,
        include_ice_water=include_ice_water,
        include_gravity=include_gravity,
        phase_solver=phase_solver,
        fluid_solver=fluid_solver,
        surface_tension_solver=surface_tension_solver,
        pressure_solver=pressure_solver,
        correction_solver=correction_solver,
        pressure_linear_solver=pressure_linear_solver,
        velocity_bc=velocity_bc,
        pressure_bc=pressure_bc,
        phase_field_bc=phase_field_bc,
        context=context,
        solver_bundle=solver_bundle,
        bc_bundle=bc_bundle,
        u_face=u_face,
        v_face=v_face,
    )
    state.velocity_layout = velocity_layout
    return state
