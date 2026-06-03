"""
Automatic derivation of PPE boundary conditions from global domain BCs.

The PPE (Pressure Poisson Equation) solves for pressure correction p' such that:
    ∇²p' = ∇·u*/Δt

The boundary conditions for p' should be derived from velocity BCs, not pressure BCs,
because the PPE corrects velocity to enforce incompressibility.

Key principles:
1. Dirichlet velocity BC (inlet) → Compatibility Neumann BC for p' at inlet (handled by post-solve fix)
2. Outlet (boundary opposite to Dirichlet inlet) → Dirichlet p' = 0 to pin pressure level
3. Neumann velocity BC → Neumann BC for p' (∂p'/∂n = 0)
4. No-slip BC → Neumann BC for p' (velocity is fixed, no correction)
5. Do-nothing BC at outlet → Dirichlet p' = 0 when inlet is Dirichlet (non-zero pressure at outlet otherwise)
"""


def _normalize_velocity_bc(value):
    """Normalize velocity BC value to a simple lowercase token."""
    if isinstance(value, dict):
        value = value.get("type", "")
    if value is None:
        return "neumann"
    return str(value).lower()


def derive_ppe_bcs_from_velocity_bcs(velocity_bcs, pressure_bcs=None):
    """Derive PPE boundary conditions from velocity boundary conditions.
    
    Args:
        velocity_bcs: Dict with keys 'top', 'bottom', 'left', 'right'
            Values can be: 'dirichlet', 'neumann', 'no_slip', 'do_nothing'
        pressure_bcs: Optional dict of pressure BCs (for reference, not used directly)
    
    Returns:
        dict: PPE boundary conditions with keys 'top', 'bottom', 'left', 'right'
            Values are: 'neumann' or 'dirichlet'
    """
    # Matrix-level default is Neumann on all sides.
    ppe_bcs = {boundary: "neumann" for boundary in ["top", "bottom", "left", "right"]}
    vbc = {b: _normalize_velocity_bc(velocity_bcs.get(b, "neumann")) for b in ppe_bcs}
    
    # Pin pressure at outlet: Dirichlet p' = 0 opposite to Dirichlet inlet
    if vbc.get("left") == "dirichlet" and vbc.get("right") in ("do_nothing", "neumann", None):
        ppe_bcs['right'] = 'dirichlet'
    if vbc.get("right") == "dirichlet" and vbc.get("left") in ("do_nothing", "neumann", None):
        ppe_bcs['left'] = 'dirichlet'
    if vbc.get("bottom") == "dirichlet" and vbc.get("top") in ("do_nothing", "neumann", None):
        ppe_bcs['top'] = 'dirichlet'
    if vbc.get("top") == "dirichlet" and vbc.get("bottom") in ("do_nothing", "neumann", None):
        ppe_bcs['bottom'] = 'dirichlet'

    # If pressure BC explicitly specifies an absolute reference (open/dirichlet),
    # carry that reference into PPE to avoid all-Neumann gauge pin artifacts.
    if pressure_bcs is not None:
        for boundary in ['top', 'bottom', 'left', 'right']:
            p_bc = pressure_bcs.get(boundary)
            if p_bc in ('open', 'dirichlet'):
                ppe_bcs[boundary] = 'dirichlet'
    
    return ppe_bcs


def derive_ppe_bcs_from_config(config):
    """Derive PPE boundary conditions from full config.
    
    This is the main function to use - it extracts velocity BCs from config
    and derives appropriate PPE BCs.
    
    Args:
        config: Full simulation config dict
    
    Returns:
        dict: PPE boundary conditions
    """
    boundary_conditions = config.get('boundary_conditions', {})
    velocity_bcs = boundary_conditions.get('velocity', {})
    pressure_bcs = boundary_conditions.get('pressure', {})
    
    vel_bc_dict = {boundary: _normalize_velocity_bc(velocity_bcs.get(boundary, "neumann"))
                   for boundary in ['top', 'bottom', 'left', 'right']}
    
    return derive_ppe_bcs_from_velocity_bcs(vel_bc_dict, pressure_bcs)


def explain_ppe_bc_derivation(velocity_bcs, ppe_bcs):
    """Print explanation of how PPE BCs were derived.
    
    Useful for debugging and understanding the derivation.
    """
    print("\n📋 PPE Boundary Condition Derivation:")
    print("=" * 60)
    for boundary in ['top', 'bottom', 'left', 'right']:
        vel_bc = velocity_bcs.get(boundary, 'unknown')
        ppe_bc = ppe_bcs.get(boundary, 'unknown')
        print(f"  {boundary:8s}: velocity={vel_bc:12s} → PPE={ppe_bc}")
    print("=" * 60)
