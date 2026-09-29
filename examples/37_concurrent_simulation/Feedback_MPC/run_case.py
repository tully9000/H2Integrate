import sys
from pathlib import Path

from h2integrate.core.supported_models import supported_models
from h2integrate.core.h2integrate_model import H2IntegrateModel


# Register the custom system-level controller. H2Integrate's SLC framework
# only looks up `control_strategy` in `supported_models` (there is no
# `model_location` scan for the SLC block), so we inject the class into the
# module-level registry before instantiating `H2IntegrateModel`. The shared
# UCControl lives in the repo-root `uc_control` package (see its docstring for
# why a custom UC-based SLC is needed).
sys.path.insert(0, str(Path(__file__).resolve().parent) + "/mppoc_control")
from uc_control import UCControl


supported_models["UCControl"] = UCControl


run_dict = {"seq": True, "con": True}

run_dir = Path(__file__).parent
fpath_config_sequential = Path(run_dir) / "config_sequential/case.yaml"
fpath_config_steppable = Path(run_dir) / "config_steppable/case.yaml"


if run_dict.get("seq", False):
    h2i_seq = H2IntegrateModel(fpath_config_sequential)
    h2i_seq.run()

if run_dict.get("con", False):
    h2i_con = H2IntegrateModel(fpath_config_steppable)
    h2i_con.run()


if run_dict.get("seq", False) and run_dict.get("con", False):
    print("\n\n\n\n")

    print("LCOE Electricity: ", h2i_seq.prob.get_val("finance_subgroup_electricity.LCOE")[0])
    print("LCOE Electricity: ", h2i_con.prob.get_val("finance_subgroup_electricity.LCOE")[0])
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
