import numpy as np
from matplotlib import pyplot as plt

from h2integrate import H2IntegrateModel


# Create an H2Integrate model
model = H2IntegrateModel("pyomo_optimized_dispatch.yaml")

demand_profile = np.ones(8760) * 100.0


# TODO: Update with demand module once it is developed
model.setup()
model.prob.set_val("battery.electricity_set_point", demand_profile, units="MW")

# Run the model
model.run()

model.post_process()


from io import StringIO
from pathlib import Path

import pyomo.environ as pyo
from pyomo.network import Arc, Port


pyomo_model = model.plant.battery.OptimizedDispatchStorageController.pyomo_model
hybrid_dispatch_model = model.plant.battery.OptimizedDispatchStorageController.hybrid_dispatch_model

with (Path(__file__).parent / "run_pyomo_optimized_dispatch_out" / "pyomo_problem.txt").open() as f:
    buffer = StringIO()
    # pyomo_model.pprint(ostream = buffer)
    hybrid_dispatch_model.pprint(ostream=buffer)
    f.write(buffer.getvalue())

with (Path(__file__).parent / "run_pyomo_optimized_dispatch_out" / "pyomo_arcs.txt").open() as f:
    buffer = StringIO()

    for arc in hybrid_dispatch_model.component_objects(Arc):
        arc.pprint(ostream=buffer)
    f.write(buffer.getvalue())


with (Path(__file__).parent / "run_pyomo_optimized_dispatch_out" / "pyomo_ports.txt").open() as f:
    buffer = StringIO()

    for port in hybrid_dispatch_model.component_objects(Port):
        port.pprint(ostream=buffer)
    f.write(buffer.getvalue())


# list(pyomo_model.component_objects(Arc))
# list(pyomo_model.component_objects(Port))
list(hybrid_dispatch_model.component_objects(Arc))
list(hybrid_dispatch_model.component_objects(Port))


# --- Inspect all Ports ---
print("--- PORTS ---")
for port in hybrid_dispatch_model.component_objects(Port, active=True):
    print(f"Port Name: {port.name}")
    # Inspect variables inside the port
    for var_name, var_obj in port.vars.items():
        print(f"  - Member: {var_name} -> {var_obj}")

# --- Inspect all Arcs ---
print("\n--- ARCS ---")
for arc in hybrid_dispatch_model.component_objects(Arc, active=True):
    # Retrieve the source and destination ports
    src = arc.source  # or arc.src
    dest = arc.destination  # or arc.dest
    print(f"Arc: {arc.name} | Source: {src.name} -> Destination: {dest.name}")

# Traverse and print the block tree hierarchy
for block in hybrid_dispatch_model.block_data_objects(descend_into=True):
    # Calculate depth based on the block's naming dots
    depth = block.name.count(".")
    indent = "  " * depth
    print(f"{indent}└── [Block] {block.name}")

# Print only active variables and their current values
print("--- Variables ---")
for v in hybrid_dispatch_model.component_objects(pyo.Var, active=True):
    if "[0]" not in v.name:
        continue
    print(f"Variable {v.name}:")
    # for index in v:
    #     print(f"  Index {index}: Value = {pyo.value(v[index])}")

# Print only constraints and their body evaluations
print("\n--- Constraints ---")
for c in hybrid_dispatch_model.component_objects(pyo.Constraint, active=True):
    print(f"Constraint {c.name}:")
    for index in c:
        print(f"  Index {index}: {pyo.value(c[index].body)} <= {pyo.value(c[index].upper)}")

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
    model.prob.get_val("battery.unused_electricity_out", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Unused Electricity (MW)",
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
plt.savefig("optimized_dispatch_plot.png", dpi=300)
