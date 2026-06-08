"""Config helpers for momentum viscous-term formulation."""

from __future__ import annotations

VALID_VISCOUS_FORMS = frozenset({"component_laplacian", "stress_divergence"})
VALID_MU_CONVENTIONS = frozenset({"inv_re", "rho_over_re"})


def normalize_viscous_form(mode: str | None) -> str:
    if mode is None:
        return "component_laplacian"
    mode = str(mode)
    if mode not in VALID_VISCOUS_FORMS:
        raise ValueError(
            f"Unknown viscous_form={mode!r}. Allowed: {sorted(VALID_VISCOUS_FORMS)}"
        )
    return mode


def normalize_mu_convention(convention: str | None) -> str:
    if convention is None:
        return "inv_re"
    convention = str(convention)
    if convention not in VALID_MU_CONVENTIONS:
        raise ValueError(
            f"Unknown viscous_mu_convention={convention!r}. "
            f"Allowed: {sorted(VALID_MU_CONVENTIONS)}"
        )
    return convention
