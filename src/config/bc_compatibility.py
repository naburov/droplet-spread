"""
Boundary condition compatibility check.

Runs before simulation to validate that velocity, pressure, phase_field,
advection, and PPE boundary conditions are consistent. Reports errors,
warnings, and a short summary. Can set config["_bc_check"] so that state
setup can avoid duplicate or misleading warnings (e.g. when explicit PPE
BCs intentionally pin pressure at an outlet).
"""


def check_bc_compatibility(config, verbose=True):
    """
    Check that all BCs in config are compatible. Call after load_config().

    Args:
        config: Full simulation config (with boundary_conditions, solver_params.ppe, etc.)
        verbose: If True, print a short report to stdout.

    Returns:
        dict with keys:
            ok: bool – no errors
            errors: list of str
            warnings: list of str
            ppe_explicit_ok: bool – explicit PPE BCs are set and are a valid override
            summary: str
    """
    errors = []
    warnings = []
    ppe_explicit_ok = False

    bc = config.get("boundary_conditions") or {}
    velocity = bc.get("velocity") or {}
    pressure = bc.get("pressure") or {}
    phase_field = bc.get("phase_field") or {}
    advection = bc.get("advection") or {}
    ppe_settings = config.get("solver_params", {}).get("ppe") or {}
    ppe_explicit = ppe_settings.get("boundary_conditions")

    # --- Velocity ↔ Pressure ---
    # Pressure "open" or dirichlet should be on the outlet side (where flow leaves).
    pressure_open_sides = [
        s for s in ["top", "bottom", "left", "right"]
        if (pressure.get(s) in ("open", "dirichlet") or
            (isinstance(pressure.get(s), str) and "dirichlet" in pressure.get(s).lower()))
    ]
    vel_dirichlet_sides = [
        s for s in ["top", "bottom", "left", "right"]
        if velocity.get(s) == "dirichlet"
    ]
    # Dirichlet with u=0,v=0 everywhere = static boundaries (no inlet); skip inlet/outlet warnings.
    dv = velocity.get("dirichlet_values") or {}
    all_zero = all(
        (dv.get(side) or {}).get("u", 0) == 0 and (dv.get(side) or {}).get("v", 0) == 0
        for side in vel_dirichlet_sides
    )
    static_dirichlet = bool(vel_dirichlet_sides and all_zero)
    # If there is a velocity inlet (non-zero dirichlet), outlet is typically the opposite;
    # pressure open/dirichlet at outlet is correct. Static (zero) dirichlet is valid with pressure open anywhere.
    if vel_dirichlet_sides and pressure_open_sides and not static_dirichlet:
        opposites = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        inlets = set(vel_dirichlet_sides)
        for side in pressure_open_sides:
            if side in inlets:
                warnings.append(
                    f"Pressure has open/dirichlet at {side} but velocity has Dirichlet (inlet) at {side}; "
                    "usually pressure open is at the outlet."
                )

    # --- PPE: explicit vs derived ---
    try:
        from solvers.ppe_bc_derivation import derive_ppe_bcs_from_config
        ppe_derived = derive_ppe_bcs_from_config(config)
    except Exception as e:
        ppe_derived = None
        if verbose:
            warnings.append(f"Could not derive PPE BCs: {e}")

    if ppe_explicit is not None and ppe_derived is not None:
        all_sides = ["top", "bottom", "left", "right"]
        mismatches = [
            (b, ppe_explicit.get(b), ppe_derived.get(b))
            for b in all_sides
            if ppe_explicit.get(b) != ppe_derived.get(b)
        ]
        if mismatches:
            # Valid override 1: exactly one dirichlet (outlet pin), rest neumann (channel flow)
            explicit_dirichlet_sides = [s for s in all_sides if ppe_explicit.get(s) == "dirichlet"]
            rest_neumann = all(ppe_explicit.get(s) == "neumann" for s in all_sides if s not in explicit_dirichlet_sides)
            vel_dirichlet_sides = [s for s in all_sides if velocity.get(s) == "dirichlet"]
            # Valid override 2: PPE dirichlet on same sides as velocity dirichlet (closed box / atmosphere)
            ppe_matches_velocity = set(explicit_dirichlet_sides) == set(vel_dirichlet_sides) and rest_neumann
            if (len(explicit_dirichlet_sides) == 1 and rest_neumann) or ppe_matches_velocity:
                ppe_explicit_ok = True
                # Don't add warning; state.py will skip the duplicate PPE warning via _bc_check
            else:
                for boundary, ex, dr in mismatches:
                    warnings.append(
                        f"PPE {boundary}: explicit={ex}, derived={dr}. "
                        "Remove solver_params.ppe.boundary_conditions to use automatic derivation, or ensure intent is correct."
                    )
        else:
            ppe_explicit_ok = True
    elif ppe_explicit is None:
        ppe_explicit_ok = True  # no explicit PPE, derivation will be used
    elif ppe_derived is None and ppe_explicit is not None:
        # Derivation failed (e.g. missing deps); accept explicit one-dirichlet pattern
        all_sides = ["top", "bottom", "left", "right"]
        explicit_dirichlet = [s for s in all_sides if ppe_explicit.get(s) == "dirichlet"]
        rest_neumann = all(ppe_explicit.get(s) == "neumann" for s in all_sides if s not in explicit_dirichlet)
        if len(explicit_dirichlet) == 1 and rest_neumann:
            ppe_explicit_ok = True

    # --- Advection vs velocity ---
    # Outflow (do_nothing) sides should usually have advection "open" so phase can leave.
    # Dirichlet with u=0,v=0 = static boundary; impermeable is correct (no inflow).
    for side in ["left", "right", "top", "bottom"]:
        v = velocity.get(side)
        a = advection.get(side)
        if v in ("do_nothing", "neumann") and a == "impermeable":
            pass
        if v == "dirichlet" and a == "impermeable" and not static_dirichlet:
            warnings.append(
                f"Velocity {side}=dirichlet (inlet) but advection {side}=impermeable; "
                "inlet usually has open or compatible advection for inflow."
            )

    # --- Phase field ---
    pf_bottom = phase_field.get("bottom")
    vel_bottom = velocity.get("bottom")
    if pf_bottom == "contact_angle" and vel_bottom not in ("no_slip", "navier_slip", "slip"):
        warnings.append(
            f"Phase field bottom=contact_angle but velocity bottom={vel_bottom}; "
            "contact_angle is typically used with no_slip or navier_slip."
        )

    ok = len(errors) == 0
    summary = "BC compatibility: OK" if ok else "BC compatibility: ERRORS"
    if warnings and ok:
        summary += f" ({len(warnings)} note(s))"

    result = {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "ppe_explicit_ok": ppe_explicit_ok,
        "summary": summary,
    }
    config["_bc_check"] = result

    if verbose:
        _print_report(result)

    return result


def _print_report(result):
    """Print a short compatibility report to stdout."""
    if result["errors"]:
        print("BC compatibility – errors:")
        for e in result["errors"]:
            print(f"  ❌ {e}")
    if result["warnings"]:
        print("BC compatibility – notes:")
        for w in result["warnings"]:
            print(f"  ⚠️  {w}")
    print(result["summary"])
