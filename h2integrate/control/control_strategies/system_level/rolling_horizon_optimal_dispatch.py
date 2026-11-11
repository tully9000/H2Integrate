import numpy as np
import warnings
from attrs import field, define
import openmdao.api as om
import matplotlib.pyplot as plt
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    Param,
    Constraint,
    Objective,
    NonNegativeReals,
    minimize,
    SolverFactory,
    SolverStatus,
    value,
)

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.supported_models import register
from h2integrate.control.control_strategies.system_level.system_level_control_baseclass import (
    SystemLevelControlBase,
)


@define(kw_only=True)
class RollingHorizonOptimalDispatchConfig(BaseConfig):
    """Configure rolling horizon optimal dispatch controller.

    Attributes:
        objective_function (str): Optimization objective: "demand_following" |
            "cost_minimization" | "profit_maximization". Default: "demand_following".
        horizon_length_hours (int): Length of prediction/optimization window in hours.
            If not specified, defaults to full simulation horizon.
        solver_name (str): Pyomo solver name ("ipopt", "glpk", "cbc", etc.).
            Default: "ipopt".
        solver_options (dict): Solver-specific options (e.g., max iterations, tolerances).
        enable_ramping_constraints (bool): If True, enforce max rate-of-change limits.
            Default: False (Phase D feature).
        enable_demand_shed (bool): If True, allow unmet demand with penalty.
            Default: False (meet demand is hard constraint).
        ramp_rate_limits_pct_per_hour (dict): Tech-specific ramp limits as percent
            of rated capacity per hour. Example: {"ng1": 10.0, "ng2": 20.0}.
        cost_per_tech (dict): Marginal cost specification per dispatchable tech.
            Values can be numeric ($/kWh), "buy_price", "VarOpEx", or "feedstock".
        commodity_sell_price (float | str): Sell price for profit maximization.
            Can be numeric ($/kWh) or reference to finance group name.
            Default: 0.0 (no revenue).
    """

    objective_function: str = field(default="demand_following")
    horizon_length_hours: int | None = field(default=None)
    solver_name: str = field(default="ipopt")
    solver_options: dict = field(factory=dict)
    enable_ramping_constraints: bool = field(default=False)
    enable_demand_shed: bool = field(default=False)
    ramp_rate_limits_pct_per_hour: dict = field(factory=dict)
    cost_per_tech: dict = field(factory=dict)
    commodity_sell_price: float | str = field(default=0.0)


@register
class RollingHorizonOptimalDispatch(SystemLevelControlBase):
    """Rolling Horizon Optimal Dispatch system-level controller.

    Solves a model predictive control (MPC) optimization problem at each time window,
    choosing dispatch decisions that minimize cost, maximize profit, or follow demand
    over a future prediction horizon. Only the first timestep's solution is applied;
    the horizon rolls forward at the next control step.

    Unlike merit-order heuristics (CostMinimization, ProfitMaximization), this controller:
    - Solves a full optimization problem using Pyomo + external solver (IPOPT/CPLEX/etc.)
    - Considers a lookahead window (default: full horizon)
    - Makes globally optimal decisions vs. greedy local dispatch
    - Handles ramping constraints, demand satisfaction penalties, and storage dynamics

    Configuration:
        ``plant_config["system_level_control"]["control_parameters"]`` must contain
        a ``rolling_horizon_optimal_dispatch`` dict with fields from
        :class:`RollingHorizonOptimalDispatchConfig`.

    See :class:`RollingHorizonOptimalDispatchConfig` for all options.
    """

    _is_steppable = True

    def setup(self):
        super().setup()

        config = RollingHorizonOptimalDispatchConfig.from_dict(
            self.options["plant_config"]["system_level_control"]
            .get("control_parameters", {})
            .get("rolling_horizon_optimal_dispatch", {})
        )

        self.config = config

        # Derive horizon in timesteps from hours
        if config.horizon_length_hours is not None:
            dt_hours = self.options["plant_config"]["plant"]["simulation"]["dt"] / 3600
            self.horizon_timesteps = max(1, int(config.horizon_length_hours / dt_hours))
        else:
            self.horizon_timesteps = self.n_timesteps

        # For objectives that require costs, set up marginal cost inputs
        if config.objective_function in ("cost_minimization", "profit_maximization"):
            self._setup_marginal_costs()

        # For profit maximization: add commodity sell price input
        if config.objective_function == "profit_maximization":
            commodity_sell_price = config.commodity_sell_price

            # If string, resolve from finance groups (like ProfitMaximizationControl does)
            if isinstance(commodity_sell_price, str):
                finance_groups = (
                    self.options["plant_config"].get("finance_parameters", {}).get("finance_groups", {})
                )
                group = finance_groups.get(commodity_sell_price)
                if group is None:
                    raise ValueError(
                        f"commodity_sell_price references finance group '{commodity_sell_price}', "
                        f"but it was not found in finance_parameters.finance_groups. "
                        f"Available groups: {list(finance_groups.keys())}"
                    )
                commodity_sell_price = group.get("model_inputs", {}).get("commodity_sell_price", 0.0)

            self.add_input(
                "commodity_sell_price",
                val=commodity_sell_price,
                shape=self.n_timesteps,
                units=f"USD/({self.commodity_rate_units}*h)",
                desc=f"Sell price per unit of {self.commodity}",
            )

        # Phase D: Initialize window state tracking for SOC continuity
        self.last_soc_per_tech = {}  # {storage_tech: soc_value}
        self.last_output_per_tech = {}  # {dispatchable_tech: output_value}

    def compute(self, inputs, outputs):
        """Solve rolling horizon optimization and extract first-window dispatch.

        At each call:
        1. Determine current window boundaries
        2. Build Pyomo optimization model
        3. Solve using configured solver
        4. Extract first timestep's optimal dispatch
        5. Apply as set-point outputs for all controlled techs
        """
        simulation_range = self._get_compute_time_range(inputs["timestep_index"])
        timestep_index = int(inputs["timestep_index"][0])

        # Determine optimization window: [timestep_index, timestep_index + horizon_timesteps)
        window_start = timestep_index
        window_end = min(timestep_index + self.horizon_timesteps, self.n_timesteps)
        window_length = window_end - window_start

        if window_length <= 0:
            # Past end of simulation; no more timesteps to dispatch
            for set_point_name in self.dispatchable_set_point_names:
                outputs[set_point_name][simulation_range] = np.zeros(self.n_timesteps_window)
            for set_point_name in self.storage_set_point_names:
                outputs[set_point_name][simulation_range] = np.zeros(self.n_timesteps_window)
            return

        # Build and solve Pyomo model
        model, dispatch_decisions = self._solve_optimization(
            inputs, window_start, window_end, window_length
        )

        if self.iter_count < 5:
            self._debug_plotting(model)

        # Extract dispatch from first timestep and write to outputs
        self._extract_and_output_dispatch(dispatch_decisions, window_start, simulation_range, outputs)

    def _solve_optimization(self, inputs, window_start, window_end, window_length):
        """Solve optimization using Pyomo + external solver (Phase D).

        Builds a ConcreteModel with:
        - Dispatchable tech outputs: output[t, tech]
        - Storage SOC dynamics: soc[t, storage], charge[t, storage], discharge[t, storage]
        - Ramping constraints (optional)
        - Energy balance constraint

        Args:
            inputs: OpenMDAO inputs dict
            window_start (int): Starting timestep of optimization window
            window_end (int): Ending timestep (exclusive)
            window_length (int): Number of timesteps in window

        Returns:
            tuple: (model, dispatch_decisions) where dispatch_decisions maps
                tech_name -> array of optimal outputs for window
        """
        # Collect all necessary data from configs and inputs
        (demand_window, dt_hours, fixed_production, flexible_production,
         dispatchable_capacities, storage_info, marginal_costs) = self._collect_window_data(
            inputs, window_start, window_end, window_length
        )

        # Build Pyomo model with variables and bounds
        model = self._build_pyomo_model(
            window_length, dispatchable_capacities, storage_info, dt_hours
        )

        # Add all constraints to model
        self._add_model_constraints(
            model, window_length, window_start, demand_window,
            fixed_production, flexible_production, dispatchable_capacities, storage_info, dt_hours
        )

        # Set objective function based on configuration
        self._set_model_objective(
            model, window_start, demand_window, marginal_costs, inputs
        )

        # Solve model and extract results
        dispatch_decisions = self._solve_and_extract_results(
            model, window_length, demand_window, fixed_production,
            flexible_production, dispatchable_capacities, storage_info
        )

        return model, dispatch_decisions

    def _collect_window_data(self, inputs, window_start, window_end, window_length):
        """Collect all tech configuration and input data for the optimization window.

        Returns:
            tuple: (demand_window, dt_hours, fixed_production, flexible_production,
                    dispatchable_capacities, storage_info, marginal_costs)
        """
        # Extract window data
        demand_window = inputs[self.demand_input_name][window_start:window_end]
        dt_hours = self.options["plant_config"]["plant"]["simulation"]["dt"] / 3600

        # Pre-compute fixed and flexible production
        fixed_production = np.zeros(window_length)
        flexible_production = np.zeros(window_length)

        for tau in range(window_length):
            abs_tau = window_start + tau
            for fixed_tech in self.fixed_techs:
                in_name = f"{fixed_tech}_{self.commodity}_out"
                if in_name in inputs:
                    fixed_production[tau] += inputs[in_name][abs_tau]
            for flexible_tech in self.flexible_techs:
                rated_name = f"{flexible_tech}_rated_{self.commodity}_production"
                if rated_name in inputs:
                    flexible_production[tau] += inputs[rated_name][0]

        # Get dispatchable capacity bounds
        dispatchable_capacities = []
        for i, tech in enumerate(self.dispatchable_techs):
            rated_name = self.dispatchable_rated_names[i]
            cap_val = inputs[rated_name]
            if hasattr(cap_val, 'item'):
                cap_val = cap_val.item()
            dispatchable_capacities.append(cap_val)

        # Get storage capacities and efficiencies
        storage_info = self._get_storage_config_info()

        # Compute marginal costs
        marginal_costs = self._compute_marginal_costs(inputs) if self.config.objective_function in ("cost_minimization", "profit_maximization") else []

        return demand_window, dt_hours, fixed_production, flexible_production, dispatchable_capacities, storage_info, marginal_costs

    def _get_storage_config_info(self):
        """Extract storage capacities, efficiencies, and bounds from configs.

        Returns:
            dict: Storage configuration with keys: capacities, charge_efficiencies,
                  discharge_efficiencies, min_soc_frac, max_soc_frac,
                  max_charge_rate, max_discharge_rate
        """
        storage_capacities = {}
        storage_charge_efficiencies = {}
        storage_discharge_efficiencies = {}
        storage_min_soc_frac = {}
        storage_max_soc_frac = {}
        storage_max_charge_rate = {}
        storage_max_discharge_rate = {}

        for storage_tech in self.storage_techs:
            # Try to get tech config from plant_config["technologies"], or use defaults
            tech_config_dict = {}
            try:
                plant_cfg = self.options["plant_config"]
                if isinstance(plant_cfg, dict) and "technologies" in plant_cfg:
                    tech_config_dict = plant_cfg["technologies"].get(storage_tech, {})
            except:
                pass

            if not tech_config_dict:
                try:
                    tech_cfg = self.options["tech_config"]
                    if isinstance(tech_cfg, dict):
                        tech_config_dict = tech_cfg.get(storage_tech, {})
                except:
                    pass

            model_inputs = tech_config_dict.get("model_inputs", {})
            shared = model_inputs.get("shared_parameters", {})

            # Use defaults if config not found
            capacity = shared.get("max_capacity", 100000)  # kWh, default 100 MWh
            storage_capacities[storage_tech] = capacity

            # Efficiencies
            perf = model_inputs.get("performance_parameters", {})
            rte = perf.get("round_trip_efficiency", 0.9)
            eff_c = np.sqrt(rte)  # Sqrt split
            eff_d = np.sqrt(rte)
            storage_charge_efficiencies[storage_tech] = eff_c
            storage_discharge_efficiencies[storage_tech] = eff_d

            # SOC bounds
            storage_min_soc_frac[storage_tech] = shared.get("min_soc_fraction", 0.0)
            storage_max_soc_frac[storage_tech] = shared.get("max_soc_fraction", 1.0)

            # Charge/discharge limits
            storage_max_charge_rate[storage_tech] = shared.get("max_charge_rate", 1e6)  # kW
            storage_max_discharge_rate[storage_tech] = shared.get("max_discharge_rate", 1e6)  # kW

        return {
            'capacities': storage_capacities,
            'charge_efficiencies': storage_charge_efficiencies,
            'discharge_efficiencies': storage_discharge_efficiencies,
            'min_soc_frac': storage_min_soc_frac,
            'max_soc_frac': storage_max_soc_frac,
            'max_charge_rate': storage_max_charge_rate,
            'max_discharge_rate': storage_max_discharge_rate,
        }

    def _build_pyomo_model(self, window_length, dispatchable_capacities, storage_info, dt_hours):
        """Build Pyomo ConcreteModel with variables and bounds.

        Returns:
            ConcreteModel: Initialized model with variables, bounds, and initial values
        """
        model = ConcreteModel()
        model.t = Set(initialize=range(window_length))

        # Dispatchable output variables
        model.output = Var(model.t, range(len(self.dispatchable_techs)), within=NonNegativeReals)

        # Storage variables
        model.soc = Var(model.t, range(len(self.storage_techs)), within=NonNegativeReals)
        model.charge = Var(model.t, range(len(self.storage_techs)), within=NonNegativeReals)
        model.discharge = Var(model.t, range(len(self.storage_techs)), within=NonNegativeReals)

        # Demand satisfaction slack (for soft constraints)
        model.demand_satisfied = Var(model.t, within=NonNegativeReals)
        for t in model.t:
            model.demand_satisfied[t].set_value(0.0)

        # Set bounds for dispatchable techs
        for t in model.t:
            for i, cap in enumerate(dispatchable_capacities):
                model.output[t, i].setub(cap)
                model.output[t, i].set_value(0.0)

        # Set bounds and initialize storage variables
        storage_capacities = storage_info['capacities']
        storage_max_soc_frac = storage_info['max_soc_frac']
        storage_min_soc_frac = storage_info['min_soc_frac']
        storage_max_charge_rate = storage_info['max_charge_rate']
        storage_max_discharge_rate = storage_info['max_discharge_rate']

        for t in model.t:
            for i, storage_tech in enumerate(self.storage_techs):
                cap = storage_capacities[storage_tech]
                model.soc[t, i].setub(cap * storage_max_soc_frac[storage_tech])
                model.soc[t, i].setlb(cap * storage_min_soc_frac[storage_tech])
                model.charge[t, i].setub(storage_max_charge_rate[storage_tech] * dt_hours)
                model.discharge[t, i].setub(storage_max_discharge_rate[storage_tech] * dt_hours)
                # Initialize to zero
                model.charge[t, i].set_value(0.0)
                model.discharge[t, i].set_value(0.0)
                # Initial SOC
                if t == 0:
                    init_soc = self.last_soc_per_tech.get(storage_tech, 0.5 * cap)
                    model.soc[t, i].set_value(init_soc)
                else:
                    model.soc[t, i].set_value(0.0)

        return model

    def _add_model_constraints(self, model, window_length, window_start, demand_window,
                             fixed_production, flexible_production, dispatchable_capacities,
                             storage_info, dt_hours):
        """Add all constraints to the Pyomo model."""
        storage_capacities = storage_info['capacities']
        storage_charge_efficiencies = storage_info['charge_efficiencies']
        storage_discharge_efficiencies = storage_info['discharge_efficiencies']

        # Energy balance constraint
        def energy_balance_rule(m, t):
            disp_prod = sum(m.output[t, i] for i in range(len(self.dispatchable_techs)))
            storage_net = sum(m.charge[t, i] - m.discharge[t, i] for i in range(len(self.storage_techs)))
            return disp_prod + fixed_production[t] + flexible_production[t] == m.demand_satisfied[t] + storage_net
        model.energy_balance = Constraint(model.t, rule=energy_balance_rule)

        # Storage SOC dynamics
        def soc_balance_rule(m, t, i):
            storage_tech = self.storage_techs[i]
            cap = storage_capacities[storage_tech]
            eff_c = storage_charge_efficiencies[storage_tech]
            eff_d = storage_discharge_efficiencies[storage_tech]

            if t == 0:
                prev_soc = self.last_soc_per_tech.get(storage_tech, 0.5 * cap)
            else:
                prev_soc = m.soc[t-1, i]

            return m.soc[t, i] == prev_soc + (m.charge[t, i] * eff_c - m.discharge[t, i] / eff_d) * dt_hours / cap
        model.soc_balance = Constraint(model.t, range(len(self.storage_techs)), rule=soc_balance_rule)

        # Ramping constraints (optional)
        if self.config.enable_ramping_constraints:
            ramp_limits = self.config.ramp_rate_limits_pct_per_hour

            def ramp_up_rule(m, t, i):
                if t == 0:
                    return Constraint.Skip
                tech = self.dispatchable_techs[i]
                ramp_pct = ramp_limits.get(tech, 100.0)
                ramp_mw_per_hour = (ramp_pct / 100.0) * dispatchable_capacities[i]
                return m.output[t, i] - m.output[t-1, i] <= ramp_mw_per_hour * dt_hours
            model.ramp_up = Constraint(model.t, range(len(self.dispatchable_techs)), rule=ramp_up_rule)

            def ramp_down_rule(m, t, i):
                if t == 0:
                    return Constraint.Skip
                tech = self.dispatchable_techs[i]
                ramp_pct = ramp_limits.get(tech, 100.0)
                ramp_mw_per_hour = (ramp_pct / 100.0) * dispatchable_capacities[i]
                return m.output[t-1, i] - m.output[t, i] <= ramp_mw_per_hour * dt_hours
            model.ramp_down = Constraint(model.t, range(len(self.dispatchable_techs)), rule=ramp_down_rule)

        # Demand satisfaction constraints (hard or soft depending on mode)
        if not self.config.enable_demand_shed:
            def demand_rule(m, t):
                return m.demand_satisfied[t] == demand_window[t]
            model.demand_rule = Constraint(model.t, rule=demand_rule)

    def _set_model_objective(self, model, window_start, demand_window, marginal_costs, inputs):
        """Set the objective function based on configuration."""
        if self.config.objective_function == "demand_following":
            def objective_expr(m):
                return sum((demand_window[t] - m.demand_satisfied[t])**2 for t in m.t)
            model.obj = Objective(expr=objective_expr, sense=minimize)

        elif self.config.objective_function == "cost_minimization":
            def objective_expr(m):
                cost = 0.0
                dt_hours = self.options["plant_config"]["plant"]["simulation"]["dt"] / 3600
                for t in m.t:
                    abs_t = window_start + t
                    for i, tech in enumerate(self.dispatchable_techs):
                        mc = marginal_costs[i]
                        cost += mc[abs_t] * m.output[t, i] * dt_hours

                if self.config.enable_demand_shed:
                    demand_penalty = sum(1e6 * (demand_window[t] - m.demand_satisfied[t])**2 for t in m.t)
                    cost += demand_penalty
                return cost
            model.obj = Objective(expr=objective_expr, sense=minimize)

        elif self.config.objective_function == "profit_maximization":
            sell_price = inputs["commodity_sell_price"]
            def objective_expr(m):
                profit = 0.0
                dt_hours = self.options["plant_config"]["plant"]["simulation"]["dt"] / 3600
                for t in m.t:
                    abs_t = window_start + t
                    price = sell_price[abs_t]
                    for i, tech in enumerate(self.dispatchable_techs):
                        mc = marginal_costs[i]
                        profit += (mc[abs_t] - price) * m.output[t, i] * dt_hours
                return profit
            model.obj = Objective(expr=objective_expr, sense=minimize)

    def _solve_and_extract_results(self, model, window_length, demand_window,
                                   fixed_production, flexible_production,
                                   dispatchable_capacities, storage_info):
        """Solve the Pyomo model and extract dispatch decisions.

        Returns:
            dict: dispatch_decisions mapping tech_name -> array of outputs for window
        """
        # Solve using Pyomo solver
        solver = self._get_pyomo_solver()
        solve_successful = False
        if solver is not None:
            try:
                results = solver.solve(model, tee=False)
                # Check both solver status and termination condition
                if hasattr(results, 'solver'):
                    status = results.solver.status
                    termination = results.solver.termination_condition if hasattr(results.solver, 'termination_condition') else None
                    if status == SolverStatus.ok or (termination and str(termination) == 'optimal'):
                        solve_successful = True
                    else:
                        warnings.warn(f"Pyomo solver status: {status}, termination: {termination}")
            except Exception as e:
                warnings.warn(f"Pyomo solve error: {e}")

        # Fallback: use greedy heuristic if solver fails or unavailable
        if not solve_successful:
            warnings.warn("Using greedy dispatch heuristic (solver unavailable or failed)")
            self._solve_optimization_greedy(model, demand_window, fixed_production, flexible_production,
                                           dispatchable_capacities, window_length)

        # Extract dispatch decisions
        dispatch_decisions = {}

        for i, tech in enumerate(self.dispatchable_techs):
            outputs = np.array([value(model.output[t, i]) for t in model.t])
            dispatch_decisions[tech] = outputs
            self.last_output_per_tech[tech] = outputs[-1]

        for i, tech in enumerate(self.storage_techs):
            outputs = np.array([value(model.discharge[t, i] - model.charge[t, i]) for t in model.t])
            dispatch_decisions[tech] = outputs
            self.last_soc_per_tech[tech] = value(model.soc[window_length-1, i])

        return dispatch_decisions

    def _solve_optimization_greedy(self, model, demand_window, fixed_production, flexible_production,
                                   dispatchable_capacities, window_length):
        """Fallback greedy heuristic when no Pyomo solver available.

        Sets Pyomo variables to feasible values without optimization.
        """
        for t in range(window_length):
            remaining = demand_window[t] - fixed_production[t] - flexible_production[t]

            # Dispatch generators in order to meet demand
            for i in range(len(self.dispatchable_techs)):
                if remaining <= 1e-6:
                    model.output[t, i].set_value(0.0)
                else:
                    dispatch = min(remaining, dispatchable_capacities[i])
                    model.output[t, i].set_value(dispatch)
                    remaining -= dispatch

            # Satisfy demand
            model.demand_satisfied[t].set_value(demand_window[t] - remaining)

            # Storage: no charge/discharge by default
            for i in range(len(self.storage_techs)):
                model.charge[t, i].set_value(0.0)
                model.discharge[t, i].set_value(0.0)
                # Initialize SOC
                if t == 0:
                    storage_tech = self.storage_techs[i]
                    # Try to get capacity from config
                    cap = 100000  # Default 100 MWh
                    try:
                        plant_cfg = self.options["plant_config"]
                        if isinstance(plant_cfg, dict) and "technologies" in plant_cfg:
                            cap = plant_cfg["technologies"].get(storage_tech, {}).get(
                                "model_inputs", {}).get("shared_parameters", {}).get("max_capacity", cap)
                    except:
                        pass

                    if cap == 100000:  # Still default, try tech_config
                        try:
                            tech_cfg = self.options["tech_config"]
                            if isinstance(tech_cfg, dict):
                                cap = tech_cfg.get(storage_tech, {}).get(
                                    "model_inputs", {}).get("shared_parameters", {}).get("max_capacity", cap)
                        except:
                            pass

                    init_soc = self.last_soc_per_tech.get(storage_tech, 0.5 * cap)
                    model.soc[t, i].set_value(init_soc)
                else:
                    model.soc[t, i].set_value(model.soc[t-1, i].value)

    def _get_pyomo_solver(self):
        """Get Pyomo solver with fallback logic.

        Tries configured solver, falls back to glpk if unavailable.
        """
        try:
            solver = SolverFactory(self.config.solver_name)
            if solver.available(exception_flag=False):
                return solver
        except:
            pass

        # Fallback to glpk
        try:
            warnings.warn(f"{self.config.solver_name} not available, trying glpk")
            solver = SolverFactory("glpk")
            if solver.available(exception_flag=False):
                return solver
        except:
            pass

        # Final fallback to ipopt
        warnings.warn("glpk not available, trying ipopt")
        return SolverFactory("ipopt")

    def _extract_and_output_dispatch(self, dispatch_decisions, window_start, simulation_range, outputs):
        """Map first-window dispatch decision to OpenMDAO set-point outputs.

        Only the first timestep's dispatch (tau=0 in the optimization window) is applied
        to the current control window. On next iteration, window rolls forward.

        Args:
            dispatch_decisions (dict): Maps tech_name -> array of outputs for window
            window_start (int): Starting timestep in full simulation
            simulation_range: OpenMDAO time-slice range
            outputs: OpenMDAO outputs dict
        """
        # Only extract first timestep from optimization window
        tau_0_dispatch = {tech: outputs[0] for tech, outputs in dispatch_decisions.items()}

        # Dispatchable techs
        for tech in self.dispatchable_techs:
            tech_idx = self.dispatchable_techs.index(tech)
            set_point_name = self.dispatchable_set_point_names[tech_idx]
            # Broadcast first timestep solution across entire control window
            outputs[set_point_name][simulation_range] = np.full(
                self.n_timesteps_window, tau_0_dispatch[tech]
            )

        # Storage techs
        for tech in self.storage_techs:
            tech_idx = self.storage_techs.index(tech)
            set_point_name = self.storage_set_point_names[tech_idx]
            outputs[set_point_name][simulation_range] = np.full(
                self.n_timesteps_window, tau_0_dispatch[tech]
            )


    def _has_time_index(self, component, time_set):
        """Check if component's index set includes time dimension."""
        if not component.is_indexed():
            return False
        for entry in component:
            if isinstance(entry, tuple):
                if any(idx in time_set for idx in entry):
                    return True
            else:
                if entry in time_set:
                    return True
        return False

    def _debug_plotting(self, model):
        """Orchestrate debug plotting of Pyomo model for inspection."""
        self._debug_plotting_summary(model)
        self._debug_plotting_components(model)

    def _debug_plotting_summary(self, model):
        """Extract and plot Pyomo model variables and constraints for debugging.

        Extracts time series from model variables and creates matplotlib plots
        showing dispatchable outputs, storage dynamics, and demand satisfaction.
        """
        # Extract time indices
        time_steps = sorted(list(model.t))
        n_steps = len(time_steps)

        # Extract dispatchable tech outputs
        disp_outputs = {}
        for i, tech in enumerate(self.dispatchable_techs):
            disp_outputs[tech] = np.array([value(model.output[t, i]) for t in time_steps])

        # Extract storage state and flows
        storage_soc = {}
        storage_charge = {}
        storage_discharge = {}
        for i, tech in enumerate(self.storage_techs):
            storage_soc[tech] = np.array([value(model.soc[t, i]) for t in time_steps])
            storage_charge[tech] = np.array([value(model.charge[t, i]) for t in time_steps])
            storage_discharge[tech] = np.array([value(model.discharge[t, i]) for t in time_steps])

        # Extract demand satisfaction
        demand_satisfied = np.array([value(model.demand_satisfied[t]) for t in time_steps])

        # Extract constraint residuals
        energy_balance_resid = []
        if hasattr(model, 'energy_balance'):
            for t in time_steps:
                resid = value(model.energy_balance[t].body - model.energy_balance[t].lower)
                energy_balance_resid.append(resid)
        energy_balance_resid = np.array(energy_balance_resid) if energy_balance_resid else None

        # Create figure with subplots
        n_rows = 2 + (1 if self.storage_techs else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)

        # Plot 1: Dispatchable tech outputs
        for tech, outputs in disp_outputs.items():
            axes[0].plot(time_steps, outputs, marker='o', label=tech, linewidth=2)
        axes[0].set_ylabel('Power (kW)')
        axes[0].set_title('Dispatchable Technology Outputs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Demand satisfaction
        axes[1].bar(time_steps, demand_satisfied, width=0.8, label='Satisfied', alpha=0.7)
        axes[1].set_ylabel('Power (kW)')
        axes[1].set_title('Demand Satisfaction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Storage dynamics (if storage exists)
        if self.storage_techs:
            for tech in self.storage_techs:
                axes[2].plot(time_steps, storage_soc[tech], marker='s', label=f'{tech} SOC', linewidth=2)
            axes[2].plot(time_steps, storage_charge.get(list(self.storage_techs)[0], 0),
                        marker='^', label='Charge', linestyle='--', linewidth=1.5)
            axes[2].plot(time_steps, storage_discharge.get(list(self.storage_techs)[0], 0),
                        marker='v', label='Discharge', linestyle='--', linewidth=1.5)
            axes[2].set_ylabel('Energy (kWh) / Power (kW)')
            axes[2].set_title('Storage State of Charge and Flows')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time Step')
        fig.tight_layout()
        # plt.show()

    def _debug_plotting_components(self, model):
        """Inspect and plot time-indexed model components (variables, parameters, constraints).

        Creates three subplots for time-indexed parameters, variables, and constraints.
        Scalar components are printed as summaries instead of plotted.
        """
        time_steps = sorted(list(model.t))
        time_set = set(time_steps)

        # Collect time-indexed and scalar components
        time_indexed_params = {}
        time_indexed_vars = {}
        time_indexed_constrs = {}
        scalar_params = {}
        scalar_vars = {}
        scalar_constrs = {}

        # Inspect parameters
        for param in model.component_objects(ctype=Param, active=True):
            if param.is_indexed():
                if self._has_time_index(param, time_set):
                    time_indexed_params[param.name] = param
                else:
                    # Extract scalar values from indexed param
                    for key in param:
                        scalar_params[f"{param.name}[{key}]"] = value(param[key])
            else:
                scalar_params[param.name] = value(param)

        # Inspect variables
        for var in model.component_objects(ctype=Var, active=True):
            if var.is_indexed():
                if self._has_time_index(var, time_set):
                    time_indexed_vars[var.name] = var
                else:
                    # Extract scalar values from indexed var
                    for key in var:
                        scalar_vars[f"{var.name}[{key}]"] = value(var[key])
            else:
                scalar_vars[var.name] = value(var)

        # Inspect constraints
        for constr in model.component_objects(ctype=Constraint, active=True):
            if constr.is_indexed():
                if self._has_time_index(constr, time_set):
                    time_indexed_constrs[constr.name] = constr
                else:
                    # Extract scalar constraint info
                    for key in constr:
                        scalar_constrs[f"{constr.name}[{key}]"] = True
            else:
                scalar_constrs[constr.name] = True

        # Extract time series data for plotting
        time_param_data = {}
        for name, param in time_indexed_params.items():
            data = []
            for t in time_steps:
                try:
                    # Try 1D indexing (just time)
                    data.append(value(param[t]))
                except:
                    try:
                        # Try finding entries with t as first index
                        for key in param:
                            if isinstance(key, tuple) and key[0] == t:
                                data.append(value(param[key]))
                                break
                        else:
                            data.append(np.nan)
                    except:
                        data.append(np.nan)
            if any(not np.isnan(d) for d in data):
                time_param_data[name] = np.array(data)

        time_var_data = {}
        for name, var in time_indexed_vars.items():
            data = []
            for t in time_steps:
                try:
                    # Try 1D indexing (just time)
                    data.append(value(var[t]))
                except:
                    try:
                        # Try finding entries with t as first index
                        for key in var:
                            if isinstance(key, tuple) and key[0] == t:
                                data.append(value(var[key]))
                                break
                        else:
                            data.append(np.nan)
                    except:
                        data.append(np.nan)
            if any(not np.isnan(d) for d in data):
                time_var_data[name] = np.array(data)

        time_constr_data = {}
        for name, constr in time_indexed_constrs.items():
            data = []
            for t in time_steps:
                try:
                    # Try 1D indexing (just time)
                    data.append(value(constr[t].body))
                except:
                    try:
                        # Try finding entries with t as first index
                        for key in constr:
                            if isinstance(key, tuple) and key[0] == t:
                                data.append(value(constr[key].body))
                                break
                        else:
                            data.append(np.nan)
                    except:
                        data.append(np.nan)
            if any(not np.isnan(d) for d in data):
                time_constr_data[name] = np.array(data)

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Plot 1: Time-indexed parameters
        if time_param_data:
            for name, data in time_param_data.items():
                axes[0].plot(time_steps, data, marker='o', label=name, linewidth=2)
            axes[0].set_ylabel('Value')
            axes[0].set_title('Time-Indexed Parameters')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No time-indexed parameters', ha='center', va='center',
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('Time-Indexed Parameters')

        # Plot 2: Time-indexed variables
        if time_var_data:
            for name, data in time_var_data.items():
                axes[1].plot(time_steps, data, marker='s', label=name, linewidth=2)
            axes[1].set_ylabel('Value')
            axes[1].set_title('Time-Indexed Variables')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No time-indexed variables', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('Time-Indexed Variables')

        # Plot 3: Time-indexed constraint bodies
        if time_constr_data:
            for name, data in time_constr_data.items():
                axes[2].plot(time_steps, data, marker='^', label=name, linewidth=2)
            axes[2].set_ylabel('Constraint Body Value')
            axes[2].set_title('Time-Indexed Constraint Bodies')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No time-indexed constraints', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Time-Indexed Constraint Bodies')

        axes[-1].set_xlabel('Time Step')
        fig.tight_layout()
        plt.show()

        # Print scalar summaries
        print("\n" + "="*60)
        print("SCALAR COMPONENTS SUMMARY")
        print("="*60)

        if scalar_params:
            print("\nScalar Parameters:")
            for name, val in scalar_params.items():
                print(f"  {name} = {val}")

        if scalar_vars:
            print("\nScalar Variables:")
            for name, val in scalar_vars.items():
                print(f"  {name} = {val}")

        if scalar_constrs:
            print("\nScalar Constraints:")
            for name in scalar_constrs.keys():
                print(f"  {name}")
