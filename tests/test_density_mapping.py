#!/usr/bin/env python3
"""
Test density mapping to understand the correct formula.
"""

import numpy as np
import jax.numpy as jnp

# Test values
rho1 = 1.0    # Air density
rho2 = 1000.0 # Water density

# Test phase field values
phi_droplet = -1.0  # Droplet
phi_air = 1.0       # Air

print("Testing density mapping:")
print(f"rho1 (air): {rho1}")
print(f"rho2 (water): {rho2}")
print()

# Current formula
phi_mapped_droplet = (phi_droplet + 1) / 2.0  # Should be 0
phi_mapped_air = (phi_air + 1) / 2.0          # Should be 1

print(f"phi_droplet = {phi_droplet} -> phi_mapped = {phi_mapped_droplet}")
print(f"phi_air = {phi_air} -> phi_mapped = {phi_mapped_air}")
print()

# Current formula
rho_droplet_current = rho2 * (1 - phi_mapped_droplet) + rho1 * phi_mapped_droplet
rho_air_current = rho2 * (1 - phi_mapped_air) + rho1 * phi_mapped_air

print("Current formula:")
print(f"rho_droplet = rho2 * (1 - 0) + rho1 * 0 = {rho2} * 1 + {rho1} * 0 = {rho_droplet_current}")
print(f"rho_air = rho2 * (1 - 1) + rho1 * 1 = {rho2} * 0 + {rho1} * 1 = {rho_air_current}")
print()

# Expected results
print("Expected results:")
print(f"Droplet (phi=-1) should have density = {rho2} (water)")
print(f"Air (phi=+1) should have density = {rho1} (air)")
print()

if abs(rho_droplet_current - rho2) < 0.1:
    print("✓ Droplet density is correct")
else:
    print("✗ Droplet density is wrong")
    print(f"  Expected: {rho2}, Got: {rho_droplet_current}")

if abs(rho_air_current - rho1) < 0.1:
    print("✓ Air density is correct")
else:
    print("✗ Air density is wrong")
    print(f"  Expected: {rho1}, Got: {rho_air_current}")

# Test the original formula
print("\nTesting original formula:")
rho_droplet_original = 1 / ((1 + phi_mapped_droplet) / (2 * rho2) + (1 - phi_mapped_droplet) / (2 * rho1))
rho_air_original = 1 / ((1 + phi_mapped_air) / (2 * rho2) + (1 - phi_mapped_air) / (2 * rho1))

print(f"Original formula - Droplet: {rho_droplet_original}")
print(f"Original formula - Air: {rho_air_original}")

# Test with actual phase field values
print("\nTesting with actual phase field:")
phi_test = jnp.array([[-1.0, 1.0], [0.0, 0.5]])
print(f"Phase field:\n{phi_test}")

# Current formula
phi_mapped_test = (phi_test + 1) / 2.0
rho_test = rho2 * (1 - phi_mapped_test) + rho1 * phi_mapped_test
print(f"Current formula result:\n{rho_test}")

# What we expect
print("Expected:")
print(f"Droplet (-1): {rho2}")
print(f"Air (+1): {rho1}")
print(f"Interface (0): {(rho1 + rho2) / 2}")
print(f"Interface (0.5): {rho1 * 0.5 + rho2 * 0.5}")
