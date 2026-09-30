import sys
import time
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

from h2integrate import H2IntegrateModel, load_tech_yaml, load_plant_yaml, load_driver_yaml
from h2integrate.core.dict_utils import percent_diff_dicts, find_nonzero_percent_diffs
from h2integrate.core.supported_models import supported_models


# Register the custom system-level controller. H2Integrate's SLC framework
# only looks up `control_strategy` in `supported_models` (there is no
# `model_location` scan for the SLC block), so we inject the class into the
# module-level registry before instantiating `H2IntegrateModel`. The shared
# UCControl lives in the repo-root `uc_control` package (see its docstring for
# why a custom UC-based SLC is needed).
sys.path.insert(0, str(Path(__file__).resolve().parent) + "/mppoc_control")
from uc_control import UCControl


supported_models["UCControl"] = UCControl

run_dict = {
    "seq": True,
    "con": True,
}

run_dir = Path(__file__).parent
fpath_config_sequential = Path(run_dir) / "config_sequential/case.yaml"
fpath_config_steppable = Path(run_dir) / "config_steppable/case.yaml"


def load_config(fpath):
    config_root = fpath.parent
    with fpath.open() as f:
        config = yaml.safe_load(f)
    config["driver_config"] = load_driver_yaml(config_root / config["driver_config"])
    config["technology_config"] = load_tech_yaml(config_root / config["technology_config"])
    config["plant_config"] = load_plant_yaml(config_root / config["plant_config"])

    return config


config_seq = load_config(fpath_config_sequential)
config_con = load_config(fpath_config_steppable)

if run_dict.get("seq", False):
    print("\n=== Sequential setup/run timing ===")

    start = time.perf_counter()
    h2i_seq = H2IntegrateModel(config_seq)
    setup_time = time.perf_counter() - start
    print(f"Sequential setup time: {setup_time:.3f} s")

    start = time.perf_counter()
    h2i_seq.run()
    run_time = time.perf_counter() - start
    print(f"Sequential run time: {run_time:.3f} s")

    inputs_seq = dict(h2i_seq.model.list_inputs(out_stream=None))
    outputs_seq = dict(h2i_seq.model.list_outputs(out_stream=None))

if run_dict.get("con", False):
    print("\n=== Concurrent setup/run timing ===")

    start = time.perf_counter()
    h2i_con = H2IntegrateModel(config_con)
    setup_time = time.perf_counter() - start
    print(f"Concurrent setup time: {setup_time:.3f} s")

    start = time.perf_counter()
    h2i_con.run()
    run_time = time.perf_counter() - start
    print(f"Concurrent run time: {run_time:.3f} s")

    inputs_con = dict(h2i_con.model.list_inputs(out_stream=None))
    outputs_con = dict(h2i_con.model.list_outputs(out_stream=None))

if run_dict.get("seq", False) and run_dict.get("con", False):
    import pprint

    inputs_pd_dict = percent_diff_dicts(inputs_seq, inputs_con, allow_dissimilar_keys=True)
    outputs_pd_dict = percent_diff_dicts(outputs_seq, outputs_con, allow_dissimilar_keys=True)

    in_abs, in_rel = find_nonzero_percent_diffs(inputs_pd_dict, dict(inputs_seq))
    out_abs, out_rel = find_nonzero_percent_diffs(outputs_pd_dict, dict(outputs_seq))

    pprint.pprint(in_abs)
    pprint.pprint(out_abs)

    all_inputs = {"seq": inputs_seq, "con": inputs_con}

    all_outputs = {"seq": outputs_seq, "con": outputs_con}

    style_kw = {
        "seq": {"color": "blue", "label": "seq."},
        "con": {"color": "orange", "label": "con."},
    }

    def plot_series(ax, dicts, k):
        ax.plot(dicts["seq"][k]["val"], **style_kw["seq"])
        ax.plot(dicts["con"][k]["val"], **style_kw["con"])
        ax.set_title(k)
        ax.legend()

    def plot_key(ax, k):
        if k in inputs_seq.keys():
            plot_series(ax, all_inputs, k)
        elif k in outputs_seq.keys():
            plot_series(ax, all_outputs, k)

    fig, ax = plt.subplots(3, 1, sharex="all", layout="constrained")

    plot_key(ax[0], "plant.battery.StoragePerformanceModel.SOC")
    plot_key(ax[1], "plant.system_level_controller.wind_uncurtailed_electricity_out")
    plot_key(ax[2], "plant.system_level_controller.solar_uncurtailed_electricity_out")

    ax[0].set_xlim([0, 300])

    print("\n\n\n\n")

    print(
        "LCOE Electricity: ",
        h2i_seq.prob.get_val("finance_subgroup_electricity.LCOE")[0],
    )
    print(
        "LCOE Electricity: ",
        h2i_con.prob.get_val("finance_subgroup_electricity.LCOE")[0],
    )
    print(
        "NPV Electricity: ",
        h2i_seq.prob.get_val("finance_subgroup_natural_gas.NPV_electricity__profast_npv")[0]
        + h2i_con.prob.get_val(
            "finance_subgroup_renewable_generation.NPV_electricity__profast_npv"
        )[0],
    )
    print(
        "NPV Electricity: ",
        h2i_con.prob.get_val("finance_subgroup_natural_gas.NPV_electricity__profast_npv")[0]
        + h2i_con.prob.get_val(
            "finance_subgroup_renewable_generation.NPV_electricity__profast_npv"
        )[0],
    )

    # []
