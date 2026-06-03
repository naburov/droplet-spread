# Config scenarios: bubble rise vs droplet spread (60° / 120°)

## Three main scenarios for running simulations

| Scenario | Config file | Contact angle | Notes |
|----------|-------------|---------------|--------|
| **Bubble rise** | `config_rising_bubble.json` | 90° | Air bubble (φ=+1 inside) rising through water; `initial_conditions.is_bubble: true`, bubble centered in domain with initial upward velocity. |
| **Droplet spread 120°** | `config_droplet_simple.json` | **120°** | Sessile droplet on surface; hydrophobic (large angle). |
| **Droplet spread 60°** | `config_template.json` | **60°** | Sessile droplet on surface; base template. |

## Other configs by contact angle / type

- **120°**: `config_droplet_simple.json`, `config_droplet_realistic_auto.json`
- **60°**: `config_template.json`, `config_droplet_realistic.json`, `sliding_droplet_tilted.json`, `config_droplet_geometry.json`, `config_upstream_flow_cox_voinov*.json`, and most others
- **Bubble** (`is_bubble: true`): `config_rising_bubble.json` only
- **Falling droplet** (no bubble): `config_falling_droplet.json` (90°)

## Run the 3 simulations

From the project root:

```bash
export PYTHONPATH=src

# 1. Bubble rise
python main.py --config configs/config_rising_bubble.json --output experiment_bubble_rise

# 2. Droplet spread 120°
python main.py --config configs/config_droplet_simple.json --output experiment_droplet_120

# 3. Droplet spread 60°
python main.py --config configs/config_template.json --output experiment_droplet_60
```
