import pprint
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from h2integrate import H2IntegrateModel
from h2integrate.core.dict_utils import percent_diff_dicts, find_nonzero_percent_diffs


run_dict = {
    "pyomo": True,
    # "slc_pyomo": True,
    # "SLC": True,
}


pyomo_dispatch_config = (
    Path(__file__).parent / "pyomo_optimized_dispatch" / "pyomo_optimized_dispatch.yaml"
)
slc_pyomo_dispatch_config = Path(__file__).parent / "SLC_pyomo" / "pyomo_optimized_dispatch.yaml"
slc_dispatch_config = (
    Path(__file__).parent / "SLC_optimized_dispatch" / "pyomo_optimized_dispatch.yaml"
)

if run_dict.get("pyomo", False):
    # Create an H2Integrate model
    h2i_pyo = H2IntegrateModel(pyomo_dispatch_config)

    demand_profile = np.ones(8760) * 100.0
    # TODO: Update with demand module once it is developed
    h2i_pyo.setup()
    h2i_pyo.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")
    # Run the model
    h2i_pyo.run()
    h2i_pyo.post_process(print_results=False)

    inputs_pyomo = dict(h2i_pyo.model.list_inputs(out_stream=None))
    outputs_pyomo = dict(h2i_pyo.model.list_outputs(out_stream=None))

    battery_pyo = h2i_pyo.model.plant.battery.PySAMBatteryPerformanceModel.system_model

if run_dict.get("slc_pyomo", False):
    # Create an H2Integrate model
    h2i_spy = H2IntegrateModel(slc_pyomo_dispatch_config)

    demand_profile = np.ones(8760) * 100.0
    # TODO: Update with demand module once it is developed
    h2i_spy.setup()
    h2i_spy.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")
    # Run the model
    h2i_spy.run()
    h2i_spy.post_process(print_results=False)

    inputs_spy = dict(h2i_spy.model.list_inputs(out_stream=None))
    outputs_spy = dict(h2i_spy.model.list_outputs(out_stream=None))

    battery_spy = h2i_spy.model.plant.battery.PySAMBatteryPerformanceModel.system_model

if run_dict.get("SLC", False):
    # Create an H2Integrate model
    h2i_slc = H2IntegrateModel(slc_dispatch_config)

    demand_profile = np.ones(8760) * 100.0
    # TODO: Update with demand module once it is developed
    h2i_slc.setup()
    h2i_slc.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")
    # Run the model
    h2i_slc.run()
    h2i_slc.post_process(print_results=False)

    inputs_SLC = dict(h2i_slc.model.list_inputs(out_stream=None))
    outputs_SLC = dict(h2i_slc.model.list_outputs(out_stream=None))

    battery_slc = h2i_slc.model.plant.battery.PySAMBatteryPerformanceModel.system_model

cases_ran = {k: run_dict.get(k, False) for k in ["pyomo", "slc_pyomo", "SLC"]}
num_cases_ran = np.sum([v for k, v in cases_ran.items()])


if num_cases_ran > 1:
    # Compare results
    if run_dict.get("pyomo", False) and run_dict.get("SLC", False):
        inputs_pd_dict = percent_diff_dicts(inputs_pyomo, inputs_SLC, allow_dissimilar_keys=True)
        outputs_pd_dict = percent_diff_dicts(outputs_pyomo, outputs_SLC, allow_dissimilar_keys=True)

        in_abs, in_rel = find_nonzero_percent_diffs(inputs_pd_dict, dict(inputs_pyomo))
        out_abs, out_rel = find_nonzero_percent_diffs(outputs_pd_dict, dict(outputs_pyomo))

        pprint.pprint(in_abs)
        pprint.pprint(out_abs)

    # Compare results
    if (
        run_dict.get("pyomo", False)
        and run_dict.get("slc_pyomo", False)
        and run_dict.get("SLC", False)
    ):
        fig, ax = plt.subplots(4, 1, sharex="all", layout="constrained")

        def plot_output(ax, k):
            ax.plot(outputs_pyomo[k]["val"])
            ax.plot(outputs_spy[k]["val"])
            ax.plot(outputs_SLC[k]["val"])

        def plot_input(ax, k):
            ax.plot(inputs_pyomo[k]["val"])
            ax.plot(inputs_spy[k]["val"])
            ax.plot(inputs_SLC[k]["val"])

        plot_output(ax[0], "plant.battery.PySAMBatteryPerformanceModel.SOC")
        plot_output(ax[1], "plant.battery.PySAMBatteryPerformanceModel.electricity_out")
        # plot_input(ax[2], "plant.battery.PySAMBatteryPerformanceModel.electricity_set_point")
        plot_input(ax[2], "plant.battery.PySAMBatteryPerformanceModel.electricity_in")

        # []


# inputs = dict(model.model.list_inputs(out_stream=None))
# outputs = dict(model.model.list_outputs(out_stream=None))


# battery_electricity_out = outputs["plant.battery.PySAMBatteryPerformanceModel.electricity_out"][
#     "val"
# ]
# demand_electricity_out = outputs[
#     "plant.electrical_load_demand.GenericDemandComponent.electricity_out"
# ]["val"]
# demand_electricity_curtailed = outputs[
#     "plant.electrical_load_demand.GenericDemandComponent.unused_electricity_out"
# ]["val"]
# demand_electricity_unmet = outputs[
#     "plant.electrical_load_demand.GenericDemandComponent.unmet_electricity_demand_out"
# ]["val"]
# wind_electricity_out = outputs["plant.wind.PYSAMWindPlantPerformanceModel.electricity_out"]["val"]

# fig, ax = plt.subplots(3, 1, sharex="all", layout="constrained")

# time = np.arange(0, len(demand_electricity_out), 1)


# def fb(ax, top, bottom=None, kw={}):
#     time = np.arange(0, len(top), 1)

#     if bottom is None:
#         bottom = np.zeros(len(top))

#     ax.fill_between(time, bottom, bottom + top, step="post", **kw)


# kw = {"alpha": 0.5}

# fb(ax[0], wind_electricity_out, kw=kw)
# fb(ax[1], battery_electricity_out, kw=kw)

# fb(ax[2], wind_electricity_out, kw=kw)
# fb(ax[2], battery_electricity_out, wind_electricity_out, kw=kw)


# ax[2].step(
#     time,
#     inputs["plant.electrical_load_demand.GenericDemandComponent.electricity_demand"]["val"],
#     where="post",
#     color="black",
# )
