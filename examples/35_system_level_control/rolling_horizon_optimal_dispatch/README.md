# Rolling Horizon Optimal Dispatch Examples

This directory demonstrates the `RollingHorizonOptimalDispatch` system-level controller with multiple dispatch objectives.

## Overview

The rolling horizon optimal dispatch (RHO) controller solves a model predictive control (MPC) problem to determine generator dispatch decisions over a configurable lookahead window. Unlike greedy merit-order heuristics, RHO makes globally optimal decisions by considering future demand and cost information.

## Key Features

- **Multiple Objectives**: Supports demand-following, cost minimization, and profit maximization dispatch strategies
- **Marginal Cost Modes**: Numeric costs, "buy_price", "VarOpEx", and "feedstock" cost specifications
- **Configurable Horizon**: Prediction window length (e.g., 24-hour lookahead)
- **Storage Support**: Integrates battery dispatch with optimal generation decisions
- **Scipy-based Optimizer**: Phase A-C uses scipy.optimize.minimize for flexibility; Phase D+ will use Pyomo/IPOPT for advanced constraints

## Running the Examples

Execute the main run script to demonstrate all scenarios:

```bash
python run_rolling_horizon_dispatch.py
```

This will run three cases and generate comparison plots:
1. **Cost Minimization**: Dispatch natural gas only when cheaper than wind curtailment
2. **Profit Maximization**: Dispatch only when sell price exceeds marginal cost
3. **Cost Min + Battery**: Optimize dispatch with battery charge/discharge decisions

## Configuration Files

### Scenario Configs

- `wind_ng_demand_cost_min.yaml`: Wind + NG with cost minimization (24-hour horizon)
- `wind_ng_demand_profit_max.yaml`: Wind + NG with profit maximization (sell price = $0.055/kWh)
- `wind_ng_battery_demand_cost_min.yaml`: Wind + NG + battery storage with cost minimization

### Technology Configs

- `tech_config.yaml`: Wind turbines, natural gas plant, demand, and combiner
- `tech_config_battery.yaml`: Adds 80 MWh battery with 20 MW charge rate

### Plant Configs

- `plant_config_cost_min.yaml`: RHO with cost minimization
- `plant_config_profit_max.yaml`: RHO with profit maximization
- `plant_config_cost_min_battery.yaml`: RHO with battery storage

## Dispatch Behaviors

### Cost Minimization
Minimizes total operational cost over the horizon subject to meeting demand.
- **Strategy**: Prefer cheaper generators (wind is free, NG costs $0.06/kWh)
- **Result**: Maximize wind utilization, minimize NG generation
- **With Battery**: Use battery to shift peak demands and smooth generator output

### Profit Maximization
Maximizes revenue (sell price × output) minus costs over the horizon.
- **Strategy**: Only dispatch if sell price > marginal cost
- **Result**: Selective dispatch based on profitability
- **Note**: Allows demand shedding when uneconomical to meet demand

## Model Parameters

### Wind Plant
- 20 turbines × 6 MW = 120 MW capacity
- 2013 wind resource (West Texas)
- CAPEX: $1,300/kW, OPEX: $39/kW/year

### Natural Gas Plant
- 100 MW capacity
- Heat rate: 7.5 MMBtu/MWh (efficient NGCC)
- Marginal cost: ~$0.06/kWh (feedstock price $4.2/MMBtu)
- CAPEX: $1,000/kW, OPEX: $10/kW/year

### Demand
- Fixed baseload: 30–50 kW (depending on scenario)
- Annual profile: 8,760 hours (full year)

### Battery (where applicable)
- 80 MWh capacity (4-hour duration at 20 MW rate)
- 90% round-trip efficiency
- Cost: $310/kWh + $311/kW

## Output

The script generates:
- **Console output**: Case completion status and verification
- **PNG plots** for each scenario:
  - Top panel: Stacked bar chart of generation dispatch + demand
  - Middle panel: Battery state-of-charge (if applicable)
  - Bottom panel: Curtailed wind

## Advanced Configuration

Edit the control parameters in plant configs to customize RHO behavior:

```yaml
control_parameters:
  rolling_horizon_optimal_dispatch:
    objective_function: cost_minimization      # or profit_maximization
    horizon_length_hours: 24                    # lookahead window
    cost_per_tech:
      natural_gas_plant: 0.06                  # $/kWh
    enable_ramping_constraints: false          # Phase D feature
    enable_demand_shed: false                  # Allow unmet demand
```

## Future Work (Phase D+)

- **Ramping Constraints**: Limit generator rate-of-change (e.g., 10%/hour)
- **Storage State Tracking**: Carry over SOC across rolling windows
- **Pyomo Integration**: Use Pyomo + IPOPT for large-scale optimization
- **Multi-commodity**: Handle hydrogen, electricity, heat in unified framework

## References

- Base class: `SystemLevelControlBase` (h2integrate.control.control_strategies.system_level)
- Controller: `RollingHorizonOptimalDispatch` (h2integrate.control.control_strategies.system_level.rolling_horizon_optimal_dispatch)
- Alternative controllers: `DemandFollowingControl`, `CostMinimizationControl`, `ProfitMaximizationControl`

---

For questions or issues, refer to the H2Integrate documentation or contact the development team.
