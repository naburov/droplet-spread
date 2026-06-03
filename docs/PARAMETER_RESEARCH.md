# Parameter Research Summary

## Literature Findings

### Weber Number for Droplet Spreading

**Typical Values:**
- **Low Weber numbers (0.01 - 0.1)**: Surface tension dominates, minimal deformation
- **For spreading simulations**: We < 0.01 recommended for surface-tension-dominated regime
- **Very small droplets**: We should be even smaller (0.001 or lower) to prevent spreading

**Physical Interpretation:**
- We = ρU²L/σ measures inertia vs surface tension
- Low We → surface tension dominates → droplet resists spreading
- High We → inertia dominates → droplet can spread/deform

### Contact Angle

**Hydrophobic vs Hydrophilic:**
- **Hydrophilic**: θ < 90° (e.g., 60°) → droplet wants to spread (wetting)
- **Hydrophobic**: θ > 90° (e.g., 120°) → droplet resists spreading (non-wetting)
- **For minimal spreading**: Use hydrophobic contact angle (120°)

### Bond Number

**For 1mm water droplet:**
- Bo = ρgR²/σ ≈ 0.1345
- Bo < 1 → Surface tension dominates gravity ✓
- This is correct for small droplets

### Capillary Number

**Definition:** Ca = μU/σ (viscous vs surface tension)
- Ca < 1: Surface tension dominates
- For spreading: Ca should be small

## Recommended Parameters for 1mm Droplet

Based on literature and physical analysis:

```json
{
  "We1": 0.001,      // Air (very small, surface tension negligible)
  "We2": 0.001,     // Water (very small, strong surface tension)
  "contact_angle": 120.0,  // Hydrophobic (resists spreading)
  "Bo": 0.1345,     // < 1, surface tension dominates
  "Pe": 10.0,       // Good interface resolution
  "epsilon": 0.015  // Interface thickness
}
```

## Key Insights

1. **We2 = 0.01 is still too high** for preventing spreading
   - Should be 0.001 or lower for very small droplets

2. **Contact angle 60° promotes spreading** (hydrophilic)
   - Should use 120° for hydrophobic behavior

3. **Bond number is correct** (Bo < 1)
   - Surface tension dominates gravity ✓

4. **Combined effect:**
   - Low We2 (0.001) + Hydrophobic contact angle (120°) → minimal spreading
   - High We2 (0.01) + Hydrophilic contact angle (60°) → promotes spreading

## References

1. "Numerical simulation and modeling of droplet spreading under smaller Weber numbers"
2. "Phase-field model for quantitative analysis of droplet wetting"
3. "Evolution dynamics of thin liquid structures investigated using a phase-field model"
4. "Capillary currents and viscous droplet spreading"
