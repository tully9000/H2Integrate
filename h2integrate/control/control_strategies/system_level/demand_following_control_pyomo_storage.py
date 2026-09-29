import numpy as np
import pyomo.environ as pyomo

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.control.control_strategies.storage.optimized_pyomo_controller import (
    OptimizedDispatchStorageController,
    OptimizedDispatchStorageControllerConfig,
)
from h2integrate.control.control_strategies.system_level.demand_following_control import (
    DemandFollowingControl,
)


class DemandFollowingControlPyomo(DemandFollowingControl, OptimizedDispatchStorageController):
    def setup(self):
        # super().setup()

        self.config = OptimizedDispatchStorageControllerConfig.from_dict(
            merge_shared_inputs(
                self.options["tech_config"]["technologies"]["battery"]["model_inputs"], "control"
            )
            | {"tech_name": "battery"}
        )

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Storage charge rate",
        )

        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=f"{self.config.commodity_rate_units}*h",
            desc="Storage capacity",
        )

        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        super().setup()

        self.n_control_window_hours = int(self.config.n_control_window_hours)
        self.updated_initial_soc = self.config.init_soc_fraction

        # Is this the best place to put this???
        self.commodity_info = {
            "commodity_name": self.config.commodity,
            "commodity_storage_units": self.config.commodity_rate_units,
        }
        # TODO: note that this definition of cost_per_production is not generalizable to multiple
        #       production technologies. Would need a name adjustment to connect it to
        #       production tech

        self.dispatch_inputs = self.config.make_dispatch_inputs()

        # get technology group name
        self.tech_group_name = self.pathname.split(".")

        # initialize dispatch inputs to None
        self.dispatch_options = None

        # create inputs for all pyomo object creation functions from all connected technologies
        # self.dispatch_connections = self.options["plant_config"]["tech_to_dispatch_connections"]
        # self.dispatch_connections = [("wind", "battery"), ("battery", "battery")]
        self.dispatch_connections = [
            ("wind", "system_level_controller"),
            ("battery", "system_level_controller"),
        ]

        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])

        self.n_steps_per_compute = int(
            self.options["plant_config"]["plant"]["simulation"].get(
                "n_steps_per_compute", self.n_timesteps
            )
        )

        self.pyomo_setup()
        self.dispatch_tech = ["battery"]

    def compute(self, inputs, outputs):
        simulation_range = self._get_compute_time_range(inputs["timestep_index"])

        commodity = self.commodity
        demand = inputs[self.demand_input_name].copy()

        # 1. Fixed techs: always produce, subtract from demand
        for fixed_tech in self.fixed_techs:
            commodity_from_tech = self._get_commodity_for_tech(fixed_tech)
            for tech_commodity in commodity_from_tech:
                if tech_commodity == commodity:
                    demand = self._subtract_fixed(fixed_tech, demand, commodity, inputs)

        # 2. Flexible techs: operate at full production
        for flexible_tech in self.flexible_techs:
            commodity_from_tech = self._get_commodity_for_tech(flexible_tech)
            for tech_commodity in commodity_from_tech:
                if tech_commodity == commodity:
                    demand = self._subtract_flexible(
                        flexible_tech, demand, commodity, inputs, outputs
                    )
                else:
                    if f"{flexible_tech}_rated_{tech_commodity}_production" in inputs:
                        # set the per-tech set-point as the rated production
                        outputs[f"{flexible_tech}_{tech_commodity}_set_point"] = inputs[
                            f"{flexible_tech}_rated_{tech_commodity}_production"
                        ] * np.ones(self.n_timesteps)

        # 3. Storage dispatch
        # number of storage components that produce the demanded commodity
        n_storage = len(
            [s for s in self.storage_techs if commodity in self._get_commodity_for_tech(s)]
        )
        for storage_tech in self.storage_techs:
            commodity_from_tech = self._get_commodity_for_tech(storage_tech)
            if commodity in commodity_from_tech:
                demand = self._dispatch_storage(
                    storage_tech, demand / n_storage, commodity, inputs, outputs
                )

        # 4. Dispatchable techs
        remaining_demand = np.maximum(demand, 0.0)

        # calculate the number of dispatchable technologies that
        # produce the demanded commodity
        n_dispatchable = len(
            [s for s in self.dispatchable_techs if commodity in self._get_commodity_for_tech(s)]
        )
        for dispatchable_tech in self.dispatchable_techs:
            commodity_from_tech = self._get_commodity_for_tech(dispatchable_tech)
            if commodity in commodity_from_tech:
                outputs[f"{dispatchable_tech}_{commodity}_set_point"][simulation_range] = (
                    remaining_demand[simulation_range] / n_dispatchable
                )

        if inputs["timestep_index"] < 50 and False:
            import matplotlib.pyplot as plt

            sim_start_index = int(inputs["timestep_index"][0])

            fig_label = f"SLC_start{sim_start_index}"

            preexisting_fig = False

            open_fig_labels = plt.get_figlabels()
            if fig_label in open_fig_labels:
                fig = plt.figure(fig_label)
                ax = fig.get_axes()

                preexisting_fig = True
            else:
                fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")
                ax = np.ravel(ax)
                fig.suptitle(f"SLC start index: {sim_start_index}")
                fig.set_label(fig_label)

            kw = {}
            kw["color"] = "orange"
            # kw["linewidth"] = 3

            if preexisting_fig and self.n_steps_per_compute != 8760:
                for axs in ax:
                    for ln in axs.lines:
                        if ln.get_color() == "orange":
                            ln.set_alpha(0.25)

            # ax[0].plot(inputs['SOC'][simulation_range], **kw)
            ax[0].plot(inputs["SOC"][: simulation_range.stop], **kw)
            ax[0].set_title("SOC")

            ax[0].scatter(0, inputs["SOC"][simulation_range.start - 1], color=kw["color"])

            # ax[0].plot(inputs["wind_electricity_out"][simulation_range], **kw)
            # ax[0].set_title("Wind electricity out")
            ax[1].plot(inputs["battery_electricity_out"][: simulation_range.stop], **kw)
            ax[1].set_title("Battery electricity out")

            # ax[2].plot(outputs["wind_electricity_set_point"][simulation_range], **kw)
            # ax[2].set_title("Wind electricity set point")
            ax[3].plot(outputs["battery_electricity_set_point"][: simulation_range.stop], **kw)
            ax[3].set_title("Battery electricity set point")

            # []

    def _dispatch_storage(self, storage_tech, remaining_demand, commodity, inputs, outputs):
        commodity_in = inputs["wind_electricity_out"]

        simulation_range = self._get_compute_time_range(inputs["timestep_index"])

        # initialize outputs
        np.zeros(self.n_timesteps)
        # if soc_timeseries is None:
        #     soc = np.zeros(self.n_timesteps)
        # else:
        #     soc = soc_timeseries

        soc = inputs["SOC"]

        # get the starting index for each control window
        window_start_indices = list(range(0, self.n_timesteps, self.n_control_window_hours))

        window_start_indices = [
            wsi
            for wsi in window_start_indices
            if ((wsi >= simulation_range.start) and (wsi < simulation_range.stop))
        ]

        upstream_techs = self.get_upstream_techs_for_commodity(storage_tech, commodity)
        commodity_into_storage = np.zeros(self.n_timesteps)
        for tech_name in upstream_techs:
            commodity_into_storage += inputs[f"{tech_name}_{commodity}_out"]

        input_parameters = {
            f"{commodity}_in": commodity_into_storage,
            f"{commodity}_set_point": inputs[f"{commodity}_demand"][simulation_range],
            # f"{commodity}_set_point" : remaining_demand
        }

        # Initialize parameters for optimized dispatch strategy
        self.initialize_parameters(input_parameters)

        # loop over all control windows, where t is the starting index of each window
        for t in window_start_indices:
            # get the inputs over the current control window
            commodity_in = inputs["wind_electricity_out"][t : t + self.n_control_window_hours]
            demand_in = inputs[f"{commodity}_demand"][t : t + self.n_control_window_hours]

            # Progress report
            if t % (self.n_timesteps // 4) < self.n_control_window_hours:
                percentage = round((t / self.n_timesteps) * 100)
                print(f"{percentage}% done with optimal dispatch")

            if t == 0:
                soc_init = self.config.init_soc_fraction
            else:
                soc_init = soc[t - 1]

            if soc_init > self.config.max_soc_fraction:
                soc_init = self.config.max_soc_fraction
            if soc_init < self.config.min_soc_fraction:
                soc_init = self.config.min_soc_fraction

            # Update time series parameters for the optimization method
            self.update_time_series_parameters(
                commodity_in=commodity_in,
                commodity_demand=demand_in,
                updated_initial_soc=soc_init,
                # updated_initial_soc=self.updated_initial_soc,
            )
            # Run dispatch optimization to minimize costs while meeting demand
            self.solve_dispatch_model(
                start_time=t,
                n_days=self.n_timesteps // 24,
            )

        outputs["battery_electricity_set_point"][simulation_range] = self.storage_dispatch_commands
        remaining_demand[simulation_range] -= inputs[f"{storage_tech}_{commodity}_out"][
            simulation_range
        ]

        if inputs["timestep_index"] < 50 and False:
            import matplotlib.pyplot as plt

            sim_start_index = int(inputs["timestep_index"][0])

            fig_label = f"dispatch_storage{sim_start_index}"

            preexisting_fig = False

            open_fig_labels = plt.get_figlabels()
            if fig_label in open_fig_labels:
                fig = plt.figure(fig_label)
                ax = fig.get_axes()

                preexisting_fig = True
            else:
                fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")
                ax = np.ravel(ax)
                fig.suptitle(f"Dispatch storage start index: {sim_start_index}")
                fig.set_label(fig_label)

            kw = {}
            kw["color"] = "orange"
            # kw["linewidth"] = 3

            if preexisting_fig and self.n_steps_per_compute != 8760:
                for axs in ax:
                    for ln in axs.lines:
                        if ln.get_color() == "orange":
                            ln.set_alpha(0.25)

            ax[0].plot(commodity_in, **kw)
            ax[0].set_title("Commodity in")
            ax[1].plot(demand_in, **kw)
            ax[1].set_title("Demand in")
            ax[2].scatter(0, soc_init, color=kw["color"])
            ax[2].set_title("SOC init")

            # []

        return remaining_demand

    def pyomo_setup(self):
        """Create the Pyomo model, extract dispatch technology names, and return dispatch solver.

        Returns:
            callable: Function(performance_model, performance_model_kwargs, inputs, commodity)
                executing rolling-window optimization to determine dispatch and returning:
                (total_out, storage_out, unmet_demand, unused_commodity, soc)
        """
        # initialize the pyomo model
        self.pyomo_model = pyomo.ConcreteModel()

        pyomo.Set(initialize=range(self.n_control_window_hours))

        self.source_techs = []
        self.dispatch_tech = []

        for connection in self.dispatch_connections:
            # get connection definition
            source_tech, intended_dispatch_tech = connection
            # only add connections to intended dispatch tech
            if any(intended_dispatch_tech in name for name in self.tech_group_name):
                # record source and dispatch techs
                if source_tech == intended_dispatch_tech:
                    self.dispatch_tech.append(source_tech)
                self.source_techs.append(source_tech)
            else:
                continue
