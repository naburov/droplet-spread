# Surface Tension Scaling Issue

## Problem

With `We2 = 0.001`, surface tension forces become unphysically large (~1e6), causing:
1. **Excessive forces** at the interface
2. **Asymmetric behavior**: Lower part spreads (contact line moves) while upper part is held by huge forces
3. **Numerical instability**: Forces too large for stable time stepping

## Root Cause

Surface tension force formula:
```
F_st = (3√2 ε / (4 We)) * κ * |∇φ| * ∇φ
```

With `We2 = 0.001`:
- Prefactor = (3√2 * 0.015) / (4 * 0.001) = **15.9**
- With curvature κ ~ 50-100 and |∇φ| ~ 1.5:
- Force magnitude = 15.9 * 50 * 1.5 = **~1,200** (can reach 1e6 with higher curvature)

## Solution

Use `We2 = 0.02` (or 0.01-0.05 range):
- Prefactor = (3√2 * 0.015) / (4 * 0.02) = **0.795**
- Force magnitude = 0.795 * 50 * 1.5 = **~60** (reasonable)
- Still strong enough to prevent excessive spreading
- Combined with hydrophobic contact angle (120°), should prevent spreading

## Why Lower Part Spreads But Upper Part Holds

1. **Lower part (contact line)**:
   - Contact angle BC might not be strong enough to prevent spreading
   - Or contact line detection is not working correctly
   - Despite huge forces, contact line can still move if BC is not enforced strongly

2. **Upper part (droplet top)**:
   - High curvature at top → huge surface tension forces
   - Forces hold the shape, preventing deformation
   - This creates the asymmetric behavior

## Recommendations

1. ✅ Use `We2 = 0.02` (updated in config)
2. ✅ Keep `contact_angle = 120°` (hydrophobic, fixed the 180-θ bug)
3. Verify contact angle BC is being enforced correctly at contact line
4. Consider tightening CFL conditions if instability persists

## Force Scaling Table

| We2   | Prefactor | Force (κ=50, |∇φ|=1.5) | Status      |
|-------|-----------|----------------------------|-------------|
| 0.001 | 15.9      | ~1,200                     | TOO HIGH    |
| 0.01  | 1.59      | ~120                       | OK          |
| 0.02  | 0.795     | ~60                        | OK          |
| 0.05  | 0.318     | ~24                        | OK          |
| 0.1   | 0.159     | ~12                        | OK          |

**Optimal range**: We2 = 0.01-0.05 for strong but reasonable surface tension.
