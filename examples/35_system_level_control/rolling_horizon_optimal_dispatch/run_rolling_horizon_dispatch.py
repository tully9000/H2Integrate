"""Rolling horizon optimal dispatch system-level control with multiple objectives.

Demonstrates the rolling horizon optimizer meeting electricity demand using
wind, natural gas, and optional battery storage under different control objectives:

- `wind_ng_demand_cost_min.yaml`: Cost minimization dispatch (prefer cheaper gen).
- `wind_ng_demand_profit_max.yaml`: Profit maximization (dispatch when profitable).
- `wind_ng_battery_demand_cost_min.yaml`: Cost min + battery storage dispatch.

Each case is run, post-processed, and plotted for the first week of the year.
The rolling horizon optimizer considers a 24-hour lookahead window when making
dispatch decisions, providing globally optimal solutions vs. greedy merit-order.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the H2Integrate repo is in the Python path
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from h2integrate.core.h2integrate_model import H2IntegrateModel



# Get the directory where this script is located
script_dir = Path(__file__).parent

cases = {
    "cost_minimization": script_dir / "wind_ng_demand_cost_min.yaml",
    "profit_maximization": script_dir / "wind_ng_demand_profit_max.yaml",
    "cost_min_battery": script_dir / "wind_ng_battery_demand_cost_min.yaml",
}

# Plot the first 168 hours (1 week)
n_hours = 168
hours = np.arange(n_hours)

for case, config_file in cases.items():
    print(f"\nRunning {case}...")
    h2i = H2IntegrateModel(str(config_file))
    h2i.run()
    h2i.post_process()


    slc_inputs = {k:v["val"] for (k, v) in h2i.prob.model.plant.system_level_controller.list_inputs(out_stream=None)}
    slc_outputs = {k:v["val"] for (k, v) in h2i.prob.model.plant.system_level_controller.list_outputs(out_stream=None)}







    has_battery = "battery" in h2i.technology_config["technologies"]

    wind_out = h2i.prob.get_val("plant.wind.electricity_out")[:n_hours]
    ng_out = h2i.prob.get_val("plant.natural_gas_plant.electricity_out", units="kW")[:n_hours]
    demand = h2i.prob.get_val("plant.electrical_load_demand.electricity_demand")[:n_hours]
    curtailed = h2i.prob.get_val("plant.electrical_load_demand.unused_electricity_out")[:n_hours]

    if has_battery:
        batt_discharge = h2i.prob.get_val("plant.battery.storage_electricity_discharge")[:n_hours]
        batt_soc = h2i.prob.get_val("plant.battery.SOC")[:n_hours]
    else:
        batt_discharge = np.zeros(n_hours)

    n_rows = 3 if has_battery else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2 + 8 * n_rows / 3), sharex=True)

    # Stacked bar chart: wind + battery discharge + NG = total supply
    axes[0].bar(hours, wind_out, width=1.0, color="tab:blue", label="Wind", align="edge")
    if has_battery:
        axes[0].bar(
            hours,
            batt_discharge,
            width=1.0,
            bottom=wind_out,
            color="tab:purple",
            label="Battery Discharge",
            align="edge",
        )
    axes[0].bar(
        hours,
        ng_out,
        width=1.0,
        bottom=wind_out + batt_discharge,
        color="tab:orange",
        label="Natural Gas",
        align="edge",
    )
    axes[0].plot(hours, demand, color="black", linewidth=1.5, linestyle="--", label="Demand")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title(f"Rolling Horizon Optimal Dispatch ({case.replace('_', ' ').title()}): First 168 Hours")
    axes[0].legend(loc="upper right")

    if has_battery:
        axes[1].plot(hours, batt_soc, color="tab:cyan")
        axes[1].set_ylabel("Battery SOC (%)")

    axes[-1].bar(hours, curtailed, width=1.0, color="tab:red", align="edge")
    axes[-1].set_ylabel("Curtailed (kW)")
    axes[-1].set_xlabel("Hour")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{Path(__file__).parent / "plots"}/rho_dispatch_results_{case}.png", dpi=150)
    print(f"✓ Saved plot: rho_dispatch_results_{case}.png")

print("\n✓ All rolling horizon dispatch scenarios complete!")
plt.show()
