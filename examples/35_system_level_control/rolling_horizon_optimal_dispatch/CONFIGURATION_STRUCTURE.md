# Configuration Structure

This directory contains YAML configurations for rolling horizon optimal dispatch (RHO) examples demonstrating different objectives and technology combinations.

## Configuration Organization

Configurations are organized into **base files** (containing common settings) and **case-specific files** (containing objective-specific parameters).

### Base Configuration Files

These files define the common platform setup (site, plant, finance, solver) shared across all cases:

- **`plant_config_rho_base.yaml`**: Base configuration for cases WITHOUT battery storage
  - Technology interconnections: Wind + Natural Gas plant only
  - Finance subgroups for wind and natural gas only
  
- **`plant_config_rho_base_battery.yaml`**: Base configuration for cases WITH battery storage
  - Technology interconnections: Wind + Natural Gas plant + Battery
  - Finance subgroups include battery technologies

### Case-Specific Configuration Files

These files extend the base configurations with objective-specific parameters:

| Case | Base Config | Config File | Objective | Technologies |
|------|---|---|---|---|
| Cost Minimization (no battery) | `plant_config_rho_base.yaml` | `plant_config_cost_min.yaml` | `cost_minimization` | Wind + NG |
| Profit Maximization (no battery) | `plant_config_rho_base.yaml` | `plant_config_profit_max.yaml` | `profit_maximization` | Wind + NG |
| Cost Minimization (battery) | `plant_config_rho_base_battery.yaml` | `plant_config_cost_min_battery.yaml` | `cost_minimization` | Wind + NG + Battery |
| Demand Following (battery) | `plant_config_rho_base_battery.yaml` | `plant_config_demand_following_battery.yaml` | `demand_following` | Wind + NG + Battery |

### Wrapper Configuration Files

Small wrapper files reference driver, technology, and plant configs:

- `wind_ng_demand_cost_min.yaml`
- `wind_ng_demand_profit_max.yaml`
- `wind_ng_battery_demand_cost_min.yaml`
- `wind_ng_battery_demand_following.yaml`

### Other Shared Configs

- **`driver_config.yaml`**: OpenMDAO driver configuration (same for all cases)
- **`tech_config.yaml`**: Technology definitions for wind/NG only cases
- **`tech_config_battery.yaml`**: Technology definitions including battery

## Common Settings (in Base Files)

All base configurations share:

```yaml
sites:                        # Location: West Texas
  - latitude: 30.6617
  - longitude: -101.7096
  - wind_resource_year: 2013

plant:
  - plant_life: 30 years
  - simulation: 8760 hours (1 year), dt=3600s

solver_options:              # All use same solver settings
  - solver_name: gauss_seidel
  - max_iter: 20
  - convergence_tolerance: 1.0e-6

finance_parameters:          # Common finance config
  - profast_lco model
  - profast_npv model
  - Cost adjustment inflation: 2.5%
  - Target dollar year: 2022
```

## Case-Specific Settings (in Case Files)

Each case overrides:

```yaml
system_level_control:
  control_parameters:
    rolling_horizon_optimal_dispatch:
      objective_function: [cost_minimization | profit_maximization | demand_following]
      horizon_length_hours: 24  # (common)
      cost_per_tech: {natural_gas_plant: 0.06}  # (varies by case)
      commodity_sell_price: 0.055  # (profit_maximization only)
```

## Adding a New Case

To add a new case (e.g., cost minimization with new tech):

1. Decide: does it use battery? → Pick appropriate base config
2. Create minimal case-specific config that overrides only:
   - `objective_function`
   - `cost_per_tech` (if costs differ)
   - `commodity_sell_price` (if profit objective)
   - Any other case-specific parameters
3. Create wrapper config referencing:
   - `driver_config.yaml`
   - Appropriate `tech_config.yaml` (or create new one)
   - Your new case config file

## File Size Comparison

| Approach | Total Lines | Duplication |
|---|---|---|
| **Before (no base files)** | 3,624 × 4 = 14,496 lines | 90% duplication |
| **After (with base files)** | (2,000 base) + (850 × 4 case-specific) ≈ 5,400 lines | 25% duplication |

**Result**: ~62% reduction in configuration code through consolidation
