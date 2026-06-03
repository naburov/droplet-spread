# Static droplet contact angle configs

Sessile droplet on a flat plate under atmosphere. Same BC pattern for all angles; only `contact_angle` (and optionally grid/IC) differ.

## Configs

| File | Contact angle | Grid | Initial condition |
|------|---------------|------|-------------------|
| `contact_angle_60.json` | 60° | 64×64 | droplet (semicircle, r=0.15, center (0.5,0)) |
| `contact_angle_90.json` | 90° | 128×128 | rectangle (0.3–0.7 × 0–0.12) |
| `contact_angle_120.json` | 120° | 128×128 | rectangle (0.3–0.7 × 0–0.12) |

## Shared boundary conditions (closed box)

- **Velocity:** top, left, right = **dirichlet** (u=0, v=0); bottom = **navier_slip** (slip_length 1 or 1000).
- **Pressure:** top, left, right = **open** (p=0); bottom = **neumann**.
- **PPE:** top, left, right = **dirichlet** (p'=0); bottom = **neumann**.
- **Phase field:** top, left, right = **neumann**; bottom = **contact_angle** (Cox–Voinov optional).
- **Chemical potential:** all sides **zero_flux**.
- **Advection:** all sides **impermeable**.

Pressure and PPE are set to open/dirichlet on the same three sides as velocity so there are no spurious pressure gradients along the boundary.

## Other files

- `CORNERS_README.md` – corner behavior (bottom-left/right).
- `CONFIG_AUDIT_contact_angle_60.md` – key-by-key audit for the 60° config.
