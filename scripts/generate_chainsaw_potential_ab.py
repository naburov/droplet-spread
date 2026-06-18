#!/usr/bin/env python3
"""Generate A/B configs to isolate the interface chainsaw seen with the
potential-form capillary force on the logdeg_convex grid.

Base: paper_air_shear_u0p5_ca90_r0p15 (We2=0.5, the most unstable production
run). Variants toggle the force form, mobility blend, and FH log clamp.
"""

import copy
import json
import os

BASE = "configs/generated_sliding_long_realpaper_longrun/paper_air_shear_u0p5_ca90_r0p15.json"
OUT_DIR = "configs/debug/chainsaw_potential_ab"

with open(BASE) as f:
    base = json.load(f)

base["time_params"]["t_max"] = 0.06
base["time_params"]["checkpoint_interval"] = 100

VARIANTS = {
    # production stack: potential force, degenerate mobility, blend 0
    "ab_potential": {},
    # isolate the force form: CSF, everything else identical
    "ab_csf": {("physical_params", "surface_tension", "force_form"): "csf"},
    # control: no capillary force at all
    "ab_zeroforce": {
        ("physical_params", "surface_tension", "composition_force_scale"): 0.0
    },
    # restore mobility floor so CH can damp grid noise in the FH tails
    "ab_potential_blend001": {
        ("solver_params", "degenerate_mobility_blend"): 0.01
    },
    "ab_potential_blend005": {
        ("solver_params", "degenerate_mobility_blend"): 0.05
    },
    # softer FH log clamp (bounded f'' at the clamp)
    "ab_potential_delta1em4": {
        ("solver_params", "phase_log_delta"): 1e-4
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
for name, patch in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["description"] = f"chainsaw potential A/B: {name} (base air_shear_u0p5)"
    for path, value in patch.items():
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    out = os.path.join(OUT_DIR, f"{name}.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", out)
