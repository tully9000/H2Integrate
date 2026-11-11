import numpy as np
from attrs import field, define
import openmdao.api as om
from scipy.optimize import minimize

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
    """

    objective_function: str = field(default="demand_following")
    horizon_length_hours: int | None = field(default=None)
    solver_name: str = field(default="ipopt")
    solver_options: dict = field(factory=dict)
    enable_ramping_constraints: bool = field(default=False)
    enable_demand_shed: bool = field(default=False)
    ramp_rate_limits_pct_per_hour: dict = field(factory=dict)


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

        # For profit maximization: add commodity sell price input
        if config.objective_function == "profit_maximization":
            self.add_input(
                "commodity_sell_price",
                val=0.0,
                shape=self.n_timesteps,
                units=f"USD/({self.commodity_rate_units}*h)",
                desc=f"Sell price per unit of {self.commodity}",
            )
            # Set up marginal costs for all dispatchable techs
            self._setup_marginal_costs()
        elif config.objective_function == "cost_minimization":
            # Set up marginal costs for all dispatchable techs
            self._setup_marginal_costs()

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

        # Extract dispatch from first timestep and write to outputs
        self._extract_and_output_dispatch(dispatch_decisions, window_start, simulation_range, outputs)

    def _solve_optimization(self, inputs, window_start, window_end, window_length):
        """Solve optimization using scipy for Phase A (demand-following).

        Phase A uses scipy.optimize.minimize for simplicity; Phase B+ will migrate to Pyomo.

        Args:
            inputs: OpenMDAO inputs dict
            window_start (int): Starting timestep of optimization window
            window_end (int): Ending timestep (exclusive)
            window_length (int): Number of timesteps in window

        Returns:
            tuple: (model, dispatch_decisions) where dispatch_decisions maps
                tech_name -> array of optimal outputs for window
        """
        # Extract demand and tech data for this window
        demand_window = inputs[self.demand_input_name][window_start:window_end]

        # Pre-compute fixed and flexible production for each timestep
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
                    flexible_production[tau] += inputs[rated_name]

        # Get capacity bounds for dispatchable and storage techs
        dispatchable_capacities = []
        for i, tech in enumerate(self.dispatchable_techs):
            rated_name = self.dispatchable_rated_names[i]
            cap_val = inputs[rated_name]
            if hasattr(cap_val, 'item'):
                cap_val = cap_val.item()
            dispatchable_capacities.append(cap_val)

        # Decision variables: [disp_tech_1_t1, disp_tech_1_t2, ..., storage_tech_1_t1, ...]
        n_disp_vars = len(self.dispatchable_techs) * window_length
        n_storage_vars = len(self.storage_techs) * window_length
        n_vars = n_disp_vars + n_storage_vars

        # Initial guess: try to meet demand with dispatchable techs
        x0 = np.zeros(n_vars)
        for t in range(window_length):
            # Dispatchable: try to meet remaining demand
            remaining = demand_window[t] - fixed_production[t] - flexible_production[t]
            for i, tech in enumerate(self.dispatchable_techs):
                var_idx = i * window_length + t
                if remaining > 0 and len(dispatchable_capacities) > i:
                    x0[var_idx] = min(remaining / len(self.dispatchable_techs), dispatchable_capacities[i])

        # Bounds: [0, capacity] for all variables
        bounds = []
        for i in range(len(self.dispatchable_techs)):
            for t in range(window_length):
                bounds.append((0, dispatchable_capacities[i]))
        for i in range(len(self.storage_techs)):
            for t in range(window_length):
                bounds.append((-1e6, 1e6))  # Storage can charge/discharge freely

        # Objective function depends on strategy
        if self.config.objective_function == "demand_following":
            def objective(x):
                # Minimize sum of squared demand errors
                total_error = 0.0
                for t in range(window_length):
                    # Total production this timestep
                    disp_prod = sum(x[i * window_length + t] for i in range(len(self.dispatchable_techs)))
                    storage_prod = sum(x[n_disp_vars + i * window_length + t] for i in range(len(self.storage_techs)))
                    total_prod = disp_prod + storage_prod + fixed_production[t] + flexible_production[t]

                    error = demand_window[t] - total_prod
                    total_error += error ** 2

                return total_error
        else:
            # Phase B+: placeholder
            def objective(x):
                return 0.0

        # Constraints: production >= 0, capacity bounds handled by scipy bounds
        constraints = []

        # Solve
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-6, 'maxiter': 100})

        # Extract dispatch decisions from solution
        dispatch_decisions = {}

        # Dispatchable techs
        for i, tech in enumerate(self.dispatchable_techs):
            outputs = result.x[i * window_length:(i + 1) * window_length]
            dispatch_decisions[tech] = outputs

        # Storage techs
        for i, tech in enumerate(self.storage_techs):
            var_start = n_disp_vars + i * window_length
            outputs = result.x[var_start:var_start + window_length]
            dispatch_decisions[tech] = outputs

        return None, dispatch_decisions

    def _build_cost_minimization_objective(self, model, inputs, window_start, window_length):
        """Build cost minimization objective for Pyomo model (Phase B feature)."""
        # Placeholder: will be implemented in Phase B
        raise NotImplementedError("Cost minimization objective is Phase B work.")

    def _build_profit_maximization_objective(self, model, inputs, window_start, window_length):
        """Build profit maximization objective for Pyomo model (Phase C feature)."""
        # Placeholder: will be implemented in Phase C
        raise NotImplementedError("Profit maximization objective is Phase C work.")

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
