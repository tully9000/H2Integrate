import sys
import cProfile
from copy import deepcopy
from pathlib import Path

import yaml

from h2integrate import H2IntegrateModel


sys.path.append(str(Path(__file__).resolve().parents[0]))
from CustomNLSolver import CustomNonLinearRunOnce


def load_yaml_to_dict(fpath):
    with Path(fpath).open() as f:
        config = yaml.safe_load(f)
    return config


# Load config files into dict
config_root = Path(__file__).parent
config_path = config_root / "solar_battery_grid.yaml"

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


# Run the simulation concurrently for all subsystems one step at a time

config_con = deepcopy(config)

config_con["plant_config"]["plant"]["simulation"]["n_timesteps"] = 8760
config_con["plant_config"]["plant"]["simulation"]["n_steps_per_compute"] = 24

# Create an H2I model for steppable simulation
h2i_con = H2IntegrateModel(config_con)

# Set plant group nonlinear solver to custom steppable solver
h2i_con.prob.model.plant.nonlinear_solver = CustomNonLinearRunOnce()
# h2i_con.prob.model.plant.linear_solver = CustomLinearRunOnce()

profiler = cProfile.Profile()
profiler.enable()


# Run the model
h2i_con.run()

profiler.disable()


# Post-process the results
h2i_con.post_process(print_results=False)

inputs_con = h2i_con.model.list_inputs(out_stream=None)
outputs_con = h2i_con.model.list_outputs(out_stream=None)
