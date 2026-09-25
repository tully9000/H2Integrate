import pprint
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from h2integrate import H2IntegrateModel
from h2integrate.core.dict_utils import percent_diff_dicts, find_nonzero_percent_diffs


run_dict = {"pyomo": True, "SLC": True}

pyomo_dispatch_config = (
    Path(__file__).parent / "pyomo_optimized_dispatch" / "pyomo_optimized_dispatch.yaml"
)
slc_dispatch_config = (
    Path(__file__).parent / "SLC_optimized_dispatch" / "pyomo_optimized_dispatch.yaml"
)

if run_dict.get("pyomo", False):
    # Create an H2Integrate model
    model = H2IntegrateModel(pyomo_dispatch_config)

    demand_profile = np.ones(8760) * 100.0
    # TODO: Update with demand module once it is developed
    model.setup()
    model.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")
    # Run the model
    model.run()
    model.post_process(print_results=False)

    inputs_pyomo = dict(model.model.list_inputs(out_stream=None))
    outputs_pyomo = dict(model.model.list_outputs(out_stream=None))

if run_dict.get("SLC", False):
    # Create an H2Integrate model
    model = H2IntegrateModel(slc_dispatch_config)

    demand_profile = np.ones(8760) * 100.0
    # TODO: Update with demand module once it is developed
    model.setup()
    model.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")
    # Run the model
    model.run()
    model.post_process(print_results=False)

    inputs_SLC = dict(model.model.list_inputs(out_stream=None))
    outputs_SLC = dict(model.model.list_outputs(out_stream=None))


# Compare results
if run_dict.get("pyomo", False) and run_dict.get("SLC", False):
    inputs_pd_dict = percent_diff_dicts(inputs_pyomo, inputs_SLC, allow_dissimilar_keys=True)
    outputs_pd_dict = percent_diff_dicts(outputs_pyomo, outputs_SLC, allow_dissimilar_keys=True)

    in_abs, in_rel = find_nonzero_percent_diffs(inputs_pd_dict, dict(inputs_pyomo))
    out_abs, out_rel = find_nonzero_percent_diffs(outputs_pd_dict, dict(outputs_pyomo))

    pprint.pprint(in_abs)
    pprint.pprint(out_abs)


# Compare results
if run_dict.get("pyomo", False) and run_dict.get("SLC", False):
    fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")

    ax[0, 0].plot(outputs_pyomo["plant.battery.PySAMBatteryPerformanceModel.SOC"]["val"])
    ax[0, 0].plot(outputs_SLC["plant.battery.PySAMBatteryPerformanceModel.SOC"]["val"])


inputs = dict(model.model.list_inputs(out_stream=None))
outputs = dict(model.model.list_outputs(out_stream=None))


battery_electricity_out = outputs["plant.battery.PySAMBatteryPerformanceModel.electricity_out"][
    "val"
]
demand_electricity_out = outputs[
    "plant.electrical_load_demand.GenericDemandComponent.electricity_out"
]["val"]
demand_electricity_curtailed = outputs[
    "plant.electrical_load_demand.GenericDemandComponent.unused_electricity_out"
]["val"]
demand_electricity_unmet = outputs[
    "plant.electrical_load_demand.GenericDemandComponent.unmet_electricity_demand_out"
]["val"]
wind_electricity_out = outputs["plant.wind.PYSAMWindPlantPerformanceModel.electricity_out"]["val"]

fig, ax = plt.subplots(3, 1, sharex="all", layout="constrained")

time = np.arange(0, len(demand_electricity_out), 1)


def fb(ax, top, bottom=None, kw={}):
    time = np.arange(0, len(top), 1)

    if bottom is None:
        bottom = np.zeros(len(top))

    ax.fill_between(time, bottom, bottom + top, step="post", **kw)


kw = {"alpha": 0.5}

fb(ax[0], wind_electricity_out, kw=kw)
fb(ax[1], battery_electricity_out, kw=kw)

fb(ax[2], wind_electricity_out, kw=kw)
fb(ax[2], battery_electricity_out, wind_electricity_out, kw=kw)


ax[2].step(
    time,
    inputs["plant.electrical_load_demand.GenericDemandComponent.electricity_demand"]["val"],
    where="post",
    color="black",
)


# Plot the results
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

start_hour = 0
end_hour = 200

ax[0].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.SOC", units="percent")[start_hour:end_hour],
    label="SOC",
)
ax[0].set_ylabel("SOC (%)")
ax[0].set_ylim([0, 110])
ax[0].axhline(y=90.0, linestyle=":", color="k", alpha=0.5, label="Max Charge")
ax[0].legend()

ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_in", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity In (MW)",
)

ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.unmet_electricity_demand_out", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Unmet Electrical Demand (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_out", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity Out (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.battery_electricity", units="MW")[start_hour:end_hour],
    linestyle="-.",
    label="Battery Electricity Out (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    demand_profile[start_hour:end_hour],
    linestyle="--",
    label="Electrical Demand (MW)",
)
ax[1].set_ylim([-1e2, 2.5e2])
ax[1].set_ylabel("Electricity Hourly (MW)")
ax[1].set_xlabel("Timestep (hr)")

plt.legend(ncol=2, frameon=False)
plt.tight_layout()
# plt.savefig("optimized_dispatch_plot.png", dpi=300)
