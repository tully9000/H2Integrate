import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

from h2integrate.core.h2integrate_model import H2IntegrateModel


sys.path.append(str(Path(__file__).resolve().parents[1]))
from CustomNLSolver import CustomNonLinearRunOnce
from comparison_tools import Profiler


# Run one of both simulation paradigms by changing the flags in this dict
run_dict = {
    # "run_sequential": True,
    "run_concurrent": True,
}


def load_yaml_to_dict(fpath):
    with Path(fpath).open() as f:
        config = yaml.safe_load(f)
    return config


# Load config files into dict
config_root = Path(__file__).parent
config_path = config_root / "wind_ng_demand.yaml"

# Load top level config
config = load_yaml_to_dict(config_path)

# Fill driver config
driver_config_path = config_root / config["driver_config"]
config["driver_config"] = load_yaml_to_dict(driver_config_path)

# Fill technology config
technology_config_path = config_root / config["technology_config"]
config["technology_config"] = load_yaml_to_dict(technology_config_path)

# Fill plant config
plant_config_path = config_root / config["plant_config"]
config["plant_config"] = load_yaml_to_dict(plant_config_path)


# ##################################
# # Create an H2I model with a fixed electricity load demand
# h2i = H2IntegrateModel("wind_ng_demand.yaml")

# # Run the model
# h2i.run()

# # Post-process the results
# h2i.post_process()


# Run simulation sequentially one subsystem at a time
if run_dict.get("run_sequential", False):
    config_seq = deepcopy(config)
    config_seq["plant_config"]["plant"]["simulation"]["n_timesteps"] = 8760
    config_seq["plant_config"]["plant"]["simulation"]["n_steps_per_compute"] = 8760

    # Create an H2I model for standard year-long simulation
    h2i_seq = H2IntegrateModel(config_seq)

    t0 = time.time()
    # Run the model
    h2i_seq.run()
    t1 = time.time()

    print(f"Sequential took {t1 - t0:.3f} seconds")

    # Post-process the results
    h2i_seq.post_process(print_results=False)

    inputs_seq = h2i_seq.model.list_inputs(out_stream=None)
    outputs_seq = h2i_seq.model.list_outputs(out_stream=None)


# Run the simulation concurrently for all subsystems one step at a time
if run_dict.get("run_concurrent", False):
    config_con = deepcopy(config)

    config_con["plant_config"]["plant"]["simulation"]["n_timesteps"] = 8760
    config_con["plant_config"]["plant"]["simulation"]["n_steps_per_compute"] = 1

    # Create an H2I model for steppable simulation
    h2i_con = H2IntegrateModel(config_con)

    # Set plant group nonlinear solver to custom steppable solver
    h2i_con.prob.model.plant.nonlinear_solver = CustomNonLinearRunOnce(
        plant_config=h2i_con.plant_config
    )
    # h2i_con.prob.model.plant.linear_solver = CustomLinearRunOnce()

    # h2i_con.prob.model.plant.system_level_controller.add_input("SOC")

    t0 = time.time()
    # Run the model

    pf = Profiler()

    with pf:
        h2i_con.run()

    t1 = time.time()

    print(f"Concurrent took {t1 - t0:.3f} seconds")

    # Post-process the results
    h2i_con.post_process(print_results=False)

    inputs_con = h2i_con.model.list_inputs(out_stream=None)
    outputs_con = h2i_con.model.list_outputs(out_stream=None)


# # Plot the first 168 hours (1 week)
n_hours = 168
# hours = np.arange(n_hours)

# wind_out = h2i_con.prob.get_val("plant.wind.electricity_out")[:n_hours]
# ng_out = h2i.prob.get_val("plant.natural_gas_plant.electricity_out", units="kW")[:n_hours]
# batt_discharge = h2i.prob.get_val("plant.battery.storage_electricity_discharge")[:n_hours]
# batt_soc = h2i.prob.get_val("plant.battery.SOC")[:n_hours]
# demand = h2i.prob.get_val("plant.electrical_load_demand.electricity_demand")[:n_hours]
# curtailed = h2i.prob.get_val("plant.electrical_load_demand.unused_electricity_out")[:n_hours]

# fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# # Stacked bar chart: wind + battery discharge + NG = total supply
# axes[0].bar(hours, wind_out, width=1.0, color="tab:blue", label="Wind", align="edge")
# axes[0].bar(
#     hours,
#     batt_discharge,
#     width=1.0,
#     bottom=wind_out,
#     color="tab:purple",
#     label="Battery Discharge",
#     align="edge",
# )
# axes[0].bar(
#     hours,
#     ng_out,
#     width=1.0,
#     bottom=wind_out + batt_discharge,
#     color="tab:orange",
#     label="Natural Gas",
#     align="edge",
# )
# axes[0].plot(hours, demand, color="black", linewidth=1.5, linestyle="--", label="Demand")
# axes[0].set_ylabel("Power (kW)")
# axes[0].set_title("System-Level Control: First 168 Hours")
# axes[0].legend()

# axes[1].plot(hours, batt_soc, color="tab:cyan")
# axes[1].set_ylabel("Battery SOC (%)")

# axes[2].bar(hours, curtailed, width=1.0, color="tab:red", align="edge")
# axes[2].set_ylabel("Curtailed (kW)")
# axes[2].set_xlabel("Hour")

# for ax in axes:
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("slc_results.png", dpi=150)
# plt.show()
