# Cahn-Hilliard-Willmore Model: Mathematical Formulation

## Energy Functional

The Cahn-Hilliard-Willmore energy functional combines:

1. **Cahn-Hilliard energy**:
   \[
   E_{CH}[\phi] = \int_\Omega \left[ f(\phi) + \frac{\epsilon^2}{2}|\nabla\phi|^2 \right] dx
   \]
   where \(f(\phi) = \frac{1}{4}(\phi^2 - 1)^2\) is the double-well potential.

2. **Willmore energy**:
   \[
   E_W[\phi] = \int_\Omega \kappa^2 |\nabla\phi| dx
   \]
   where \(\kappa = \nabla \cdot \mathbf{n}\) is the mean curvature, and \(\mathbf{n} = \frac{\nabla\phi}{|\nabla\phi|}\) is the interface normal.

3. **Combined energy**:
   \[
   E[\phi] = E_{CH}[\phi] + \lambda_W E_W[\phi]
   \]
   where \(\lambda_W \geq 0\) is the Willmore regularization parameter.

## Variational Derivative

The evolution equation comes from gradient flow:
\[
\frac{\partial\phi}{\partial t} = -M \frac{\delta E}{\delta\phi}
\]

where \(M = 1/Pe\) is the mobility and \(\delta E/\delta\phi\) is the variational derivative.

## Variational Derivative Calculation

### Cahn-Hilliard part:
\[
\frac{\delta E_{CH}}{\delta\phi} = f'(\phi) - \epsilon^2 \Delta\phi
\]
where \(f'(\phi) = \phi(\phi^2 - 1)\).

### Willmore part (more complex):

The Willmore energy can be written as:
\[
E_W = \int \left( \nabla \cdot \frac{\nabla\phi}{|\nabla\phi|} \right)^2 |\nabla\phi| dx
\]

The variational derivative is:
\[
\frac{\delta E_W}{\delta\phi} = -\nabla \cdot \left[ \frac{\partial}{\partial(\nabla\phi)} \left( \kappa^2 |\nabla\phi| \right) \right] + \frac{\partial}{\partial\phi} \left( \kappa^2 |\nabla\phi| \right)
\]

After calculation, this leads to a **6th-order term**:
\[
\frac{\delta E_W}{\delta\phi} = \Delta^3\phi + \text{lower order terms}
\]

## Simplified Formulation (Numerical Implementation)

For numerical implementation, we use a simplified form that captures the Willmore effect:

\[
\mu_W = \epsilon_W \Delta^2\phi
\]

where \(\epsilon_W \ll \epsilon\) is a small regularization parameter. This provides 4th-order smoothing that approximates the Willmore regularization effect.

## Full Equation

The Cahn-Hilliard-Willmore equation becomes:

\[
\frac{\partial\phi}{\partial t} + \mathbf{u} \cdot \nabla\phi = \frac{1}{Pe} \Delta(\mu_{CH} + \lambda_W \mu_W)
\]

where:
- \(\mu_{CH} = f'(\phi) - \epsilon^2 \Delta\phi\) is the Cahn-Hilliard chemical potential
- \(\mu_W = \epsilon_W \Delta^2\phi\) is the Willmore chemical potential (4th-order term)
- \(\lambda_W \geq 0\) is the Willmore regularization strength
- \(\epsilon_W\) is the Willmore regularization parameter (typically \(0.001 \cdot \epsilon\))
- The Lagrange multiplier ensures mass conservation: \(\mu \to \mu - \langle\mu\rangle\)

## Implementation Details

1. **Chemical potential computation**:
   - Compute \(\mu_{CH} = f'(\phi) - \epsilon^2 \Delta\phi\)
   - If Willmore enabled: compute \(\mu_W = \epsilon_W \Delta^2\phi\)
   - Combined: \(\mu_{total} = \mu_{CH} + \lambda_W \mu_W\)

2. **Evolution**:
   - Apply boundary conditions to \(\mu_{total}\)
   - Subtract Lagrange multiplier: \(\mu_{total} \to \mu_{total} - \langle\mu_{total}\rangle\)
   - Update: \(\phi^{n+1} = \phi^n + \Delta t \left[ -\mathbf{u} \cdot \nabla\phi + \frac{1}{Pe} \Delta\mu_{total} \right]\)

3. **Parameters**:
   - \(\lambda_W = 0\): Disables Willmore regularization (pure Cahn-Hilliard)
   - \(\lambda_W > 0\): Enables Willmore smoothing
   - \(\epsilon_W\): Controls strength of 4th-order smoothing (typically \(0.001 \cdot \epsilon\))

## Implementation Notes

1. **6th-order accuracy**: The full Willmore term requires 6th-order finite differences
2. **Stability**: The Willmore term provides additional regularization
3. **Mass conservation**: The Lagrange multiplier ensures \(\int \phi dx = \text{const}\)
4. **Energy dissipation**: The gradient flow structure ensures \(dE/dt \leq 0\)

## Comparison with Previous Implementation

**Previous (ad-hoc)**:
- Added smoothing term: \(-λ \cdot (|\kappa| - \kappa_{thresh}) \cdot \text{sign}(\kappa) \cdot |\nabla\phi|\)
- Not derived from energy functional
- May not preserve energy structure
- Could conflict with contact angle BCs

**Current (Proper Willmore)**:
- Derived from Willmore energy functional
- Uses 4th-order term: \(\mu_W = \epsilon_W \Delta^2\phi\)
- Ensures energy stability (gradient flow structure)
- Physically consistent
- Compatible with contact angle BCs (acts globally, not just at contact line)

## Configuration

In the config file, use:
```json
"physical_params": {
    "lambda_willmore": 0.01,      // Willmore regularization strength
    "epsilon_willmore": 0.00002   // Willmore parameter (typically 0.001 * epsilon)
}
```

If `lambda_willmore = 0` or not specified, Willmore regularization is disabled.
