"""
Parameterized unit commitment (UC) MILP model for rolling-horizon dispatch.

Accepts all parameters via a dict (no module-level constants).
Units: kW throughout (matching hercules convention).
Solver: HiGHS via Pyomo.
"""

import logging

import numpy as np
import pyomo.environ as pyo


logger = logging.getLogger(__name__)

# ── Derive cost parameters from hercules OCGT config ─────────────────────────

KW_TO_MMBTU_HR = 3.412142 / 1000  # 1 kW_thermal = 0.003412 MMBtu/hr


def derive_cost_params(ocgt_config, gas_price, startup_cost=None):
    """
    Derive UC cost parameters from a hercules OCGT config dict.

    Uses a linear fit to the efficiency table:
        fuel_input = A_nolod + B_inchr * power_elec

    Args:
        ocgt_config: dict from hercules_input.yaml for one OCGT, must contain
            'rated_capacity', 'efficiency_table', 'hhv', 'startup_fuel_fraction',
            'hot_startup_time'.
        gas_price: fuel cost in $/MMBtu.
        startup_cost: override for startup cost ($/start). If None, estimated
            from fuel + maintenance components.

    Returns:
        dict with keys: A_nolod (kW_th), B_inchr (kW_th/kW_el),
        gas_price_per_kwh ($/kWh_th), startup_cost ($), P_max (kW), P_min (kW),
        ramp_rate (kW/hr), min_up_time (hr), min_down_time (hr).
    """
    rated_capacity = ocgt_config["rated_capacity"]  # kW
    eff_table = ocgt_config["efficiency_table"]
    power_fraction = np.array(eff_table["power_fraction"])
    efficiency = np.array(eff_table["efficiency"])

    # Compute fuel input at each operating point
    power_elec = power_fraction * rated_capacity  # kW
    fuel_input = power_elec / efficiency  # kW_thermal

    # Linear fit: fuel = A_nolod + B_inchr * power
    B_inchr, A_nolod = np.polyfit(power_elec, fuel_input, 1)

    # Convert gas price from $/MMBtu to $/kWh_thermal
    gas_price_per_kwh = gas_price * KW_TO_MMBTU_HR  # $/kWh_th

    # Estimate startup cost if not provided
    if startup_cost is None:
        startup_fuel_fraction = ocgt_config.get("startup_fuel_fraction", 0.35)
        hot_startup_time = ocgt_config.get("hot_startup_time", 420.0)  # seconds

        rated_fuel_flow = rated_capacity / efficiency[0]  # kW_th at full load
        startup_fuel_kw = startup_fuel_fraction * rated_fuel_flow
        startup_energy_kwh = startup_fuel_kw * (hot_startup_time / 3600)
        startup_fuel_cost = startup_energy_kwh * gas_price_per_kwh

        # Maintenance component: default $2000/start (overhaul_cost / max_starts)
        startup_maint_cost = 2000.0
        startup_cost = startup_fuel_cost + startup_maint_cost

    # Physical constraints
    min_stable_load_fraction = ocgt_config.get("min_stable_load_fraction", 0.2)
    ramp_rate_fraction = ocgt_config.get("ramp_rate_fraction", 0.1)
    min_up_time_s = ocgt_config.get("min_up_time", 3600)
    min_down_time_s = ocgt_config.get("min_down_time", 3600)

    return {
        "A_nolod": A_nolod,  # kW_thermal (no-load fuel)
        "B_inchr": B_inchr,  # kW_fuel / kW_elec (incremental heat rate)
        "gas_price_per_kwh": gas_price_per_kwh,  # $/kWh_thermal
        "startup_cost": startup_cost,  # $/start
        "P_max": rated_capacity,  # kW
        "P_min": min_stable_load_fraction * rated_capacity,  # kW
        "ramp_rate": ramp_rate_fraction * rated_capacity * 60,  # kW/hr (fraction is per-minute)
        "min_up_time": max(1, round(min_up_time_s / 3600)),  # hours (integer, >= 1)
        "min_down_time": max(1, round(min_down_time_s / 3600)),  # hours (integer, >= 1)
    }


# ── Build parameterized UC model ─────────────────────────────────────────────


def build_uc_model(params):
    """
    Build a unit commitment MILP over a horizon of ``len(load_profile)`` hours.

    Args:
        params: dict with keys:
            n_units: int — number of gas turbine units
            unit_params: list of dicts from derive_cost_params(), one per unit
            load_profile: list/array of hourly load values (kW); its length sets
                the optimization horizon.
            solar_profile: list/array of hourly solar availability values (kW)
            wind_profile: list/array of hourly wind availability values (kW).
                Optional; defaults to all zeros.
            shortfall_penalty: cost ($/kWh) charged on unmet load. Optional;
                defaults to 1e4 (far above any fuel cost, so load is only shed
                when physically unavoidable).
            battery: dict with keys:
                P_charge: max charge rate (kW)
                P_discharge: max discharge rate (kW)
                E_capacity: energy capacity (kWh)
                SOC_min: minimum SOC (kWh)
                SOC_max: maximum SOC (kWh)
                SOC_init: current SOC (kWh)
                eta_charge: charge efficiency (0-1)
                eta_discharge: discharge efficiency (0-1)
            initial_commitment: list of 0/1 per unit (current on/off state)

    Returns:
        Pyomo ConcreteModel (unsolved).
    """
    n_units = params["n_units"]
    unit_params = params["unit_params"]
    load_profile = params["load_profile"]
    solar_profile = params["solar_profile"]
    horizon = len(load_profile)
    wind_profile = params.get("wind_profile", np.zeros(horizon))
    shortfall_penalty = params.get("shortfall_penalty", 1.0e4)  # $/kWh unmet load
    # Tiny per-hour tilt on the shortfall penalty so that, when a window has an
    # *unavoidable* shortfall (its total unserved energy is fixed by the supply
    # gap, see the energy-limited floor in mppoc_datacenter_measures.metrics.
    # windows.energy_limited_shortfall), the solver is no longer indifferent to
    # *when* it sheds. Without this tilt the flat penalty makes many dispatches
    # cost-identical, and HiGHS may return one that under-discharges the battery
    # at high SOC (or, with a lossless battery, even charges) during a deficit.
    # Making near-term shed cost fractionally more forces the battery to serve
    # load as early as it can and pushes the unavoidable shed to when it is
    # genuinely depleted — the total shed energy is unchanged, only its timing.
    shortfall_time_weight = params.get("shortfall_time_weight", 1.0e-3)
    batt = params["battery"]
    initial_commitment = params.get("initial_commitment", [0] * n_units)

    HOURS = list(range(horizon))

    m = pyo.ConcreteModel("RollingHorizon_UC")

    # Sets
    m.T = pyo.Set(initialize=HOURS)
    m.I = pyo.RangeSet(1, n_units)

    # Store params on model for result extraction
    m._params = params

    # Parameters
    m.d_load = pyo.Param(m.T, initialize=dict(enumerate(load_profile)))
    m.d_solar = pyo.Param(m.T, initialize=dict(enumerate(solar_profile)))
    m.d_wind = pyo.Param(m.T, initialize=dict(enumerate(wind_profile)))

    # Per-unit parameters (indexed by unit 1..n_units)
    P_max = {i + 1: unit_params[i]["P_max"] for i in range(n_units)}
    P_min = {i + 1: unit_params[i]["P_min"] for i in range(n_units)}
    A_nolod = {i + 1: unit_params[i]["A_nolod"] for i in range(n_units)}
    B_inchr = {i + 1: unit_params[i]["B_inchr"] for i in range(n_units)}
    gas_price = {i + 1: unit_params[i]["gas_price_per_kwh"] for i in range(n_units)}
    C_start = {i + 1: unit_params[i]["startup_cost"] for i in range(n_units)}
    ramp_up = {i + 1: unit_params[i]["ramp_rate"] for i in range(n_units)}
    ramp_dn = {i + 1: unit_params[i]["ramp_rate"] for i in range(n_units)}
    min_up = {i + 1: unit_params[i]["min_up_time"] for i in range(n_units)}
    min_dn = {i + 1: unit_params[i]["min_down_time"] for i in range(n_units)}
    u_init = {i + 1: initial_commitment[i] for i in range(n_units)}

    # ── Decision variables ───────────────────────────────────────────────

    m.u = pyo.Var(m.I, m.T, within=pyo.Binary)  # on/off
    m.v = pyo.Var(m.I, m.T, within=pyo.Binary)  # startup
    m.w = pyo.Var(m.I, m.T, within=pyo.Binary)  # shutdown

    m.p_gas = pyo.Var(
        m.I,
        m.T,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, t: (0, P_max[i]),
    )

    m.p_solar = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_wind = pyo.Var(m.T, within=pyo.NonNegativeReals)

    # Unmet-load slack: lets the UC degrade gracefully when generation cannot
    # physically meet the load (mirrors the greedy controller's allowed
    # shortfall) instead of the MILP becoming infeasible.
    m.p_short = pyo.Var(m.T, within=pyo.NonNegativeReals)

    m.p_ch = pyo.Var(
        m.T,
        within=pyo.NonNegativeReals,
        bounds=(0, batt["P_charge"]),
    )
    m.p_dis = pyo.Var(
        m.T,
        within=pyo.NonNegativeReals,
        bounds=(0, batt["P_discharge"]),
    )
    m.soc = pyo.Var(
        m.T,
        within=pyo.NonNegativeReals,
        bounds=(batt["SOC_min"], batt["SOC_max"]),
    )

    # ── Objective: minimize fuel + startup cost ──────────────────────────

    # Optional terminal-SOC reward (see "charge_bias" strategy below). Valued
    # just under the cheapest gas marginal cost by default, so the battery is
    # topped up from surplus renewables but never charged by burning gas.
    soc_terminal_strategy = batt.get("soc_terminal_strategy", "fixed_50")
    gas_marginal_cost = min(B_inchr[i] * gas_price[i] for i in range(1, n_units + 1))
    soc_terminal_value = batt.get("soc_terminal_value", 0.9 * gas_marginal_cost)

    def obj_rule(m):
        fuel = sum(
            (A_nolod[i] * m.u[i, t] + B_inchr[i] * m.p_gas[i, t]) * gas_price[i]
            for i in m.I
            for t in m.T
        )
        starts = sum(C_start[i] * m.v[i, t] for i in m.I for t in m.T)
        shortfall = sum(
            shortfall_penalty * (1.0 + shortfall_time_weight * (max(m.T) - t)) * m.p_short[t]
            for t in m.T
        )
        # Reward higher end-of-window SOC (negative cost) to bias toward a
        # charged battery. Only active for the "charge_bias" strategy.
        terminal_reward = (
            -soc_terminal_value * m.soc[max(m.T)] if soc_terminal_strategy == "charge_bias" else 0.0
        )
        return fuel + starts + shortfall + terminal_reward

    m.cost = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ── Constraints ──────────────────────────────────────────────────────

    # 1. Power balance
    def power_balance_rule(m, t):
        gas_total = sum(m.p_gas[i, t] for i in m.I)
        return (
            gas_total + m.p_solar[t] + m.p_wind[t] + m.p_dis[t] - m.p_ch[t] + m.p_short[t]
            == m.d_load[t]
        )

    m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)

    # 2. Generator output limits
    def gen_lb_rule(m, i, t):
        return m.p_gas[i, t] >= P_min[i] * m.u[i, t]

    def gen_ub_rule(m, i, t):
        return m.p_gas[i, t] <= P_max[i] * m.u[i, t]

    m.gen_lb = pyo.Constraint(m.I, m.T, rule=gen_lb_rule)
    m.gen_ub = pyo.Constraint(m.I, m.T, rule=gen_ub_rule)

    # 3. Startup / shutdown logic
    def startup_shutdown_rule(m, i, t):
        if t == 0:
            return m.u[i, t] - u_init[i] == m.v[i, t] - m.w[i, t]
        return m.u[i, t] - m.u[i, t - 1] == m.v[i, t] - m.w[i, t]

    def no_simul_rule(m, i, t):
        return m.v[i, t] + m.w[i, t] <= 1

    m.startup_shutdown = pyo.Constraint(m.I, m.T, rule=startup_shutdown_rule)
    m.no_simul = pyo.Constraint(m.I, m.T, rule=no_simul_rule)

    # 4. Minimum up time
    def min_up_rule(m, i, t):
        mu = min_up[i]
        if t + mu > max(HOURS) + 1:
            return pyo.Constraint.Skip
        return sum(m.u[i, tau] for tau in range(t, t + mu)) >= mu * m.v[i, t]

    m.min_up = pyo.Constraint(m.I, m.T, rule=min_up_rule)

    # 5. Minimum down time
    def min_dn_rule(m, i, t):
        md = min_dn[i]
        if t + md > max(HOURS) + 1:
            return pyo.Constraint.Skip
        return sum(1 - m.u[i, tau] for tau in range(t, t + md)) >= md * m.w[i, t]

    m.min_dn = pyo.Constraint(m.I, m.T, rule=min_dn_rule)

    # 6. Ramp rate limits
    def ramp_up_rule(m, i, t):
        if t == 0:
            # Ramp from initial state
            if u_init[i]:
                return m.p_gas[i, t] <= P_max[i]  # already on, no startup ramp issue
            else:
                return pyo.Constraint.Skip  # starting from off, gen_lb handles it
        return m.p_gas[i, t] - m.p_gas[i, t - 1] <= (
            ramp_up[i] * m.u[i, t - 1] + P_min[i] * m.v[i, t]
        )

    def ramp_dn_rule(m, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.p_gas[i, t - 1] - m.p_gas[i, t] <= (ramp_dn[i] * m.u[i, t] + P_min[i] * m.w[i, t])

    m.ramp_up = pyo.Constraint(m.I, m.T, rule=ramp_up_rule)
    m.ramp_dn = pyo.Constraint(m.I, m.T, rule=ramp_dn_rule)

    # 7. Battery SOC dynamics
    eta_ch = batt["eta_charge"]
    eta_dis = batt["eta_discharge"]
    soc_init = batt["SOC_init"]
    m.soc_init_value = soc_init  # store for extract_uc_results

    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == soc_init + m.p_ch[t] * eta_ch - m.p_dis[t] / eta_dis
        return m.soc[t] == m.soc[t - 1] + m.p_ch[t] * eta_ch - m.p_dis[t] / eta_dis

    m.soc_dynamics = pyo.Constraint(m.T, rule=soc_rule)

    # SOC terminal constraint
    soc_terminal_strategy = batt.get("soc_terminal_strategy", "fixed_50")
    if soc_terminal_strategy == "track_init":
        # Original: end near where we started (risks ratcheting to min/max)
        soc_terminal_target = max(batt["SOC_min"], soc_init - 0.05 * batt["E_capacity"])
        m.soc_terminal = pyo.Constraint(expr=m.soc[max(HOURS)] >= soc_terminal_target)
    elif soc_terminal_strategy.startswith("fixed_"):
        # "fixed_<NN>" targets NN% of energy capacity at the window end with a
        # ±5% band — prevents SOC ratcheting. e.g. "fixed_50" -> 50%,
        # "fixed_80" -> 80%.
        try:
            target_fraction = float(soc_terminal_strategy.split("_", 1)[1]) / 100.0
        except ValueError as exc:
            raise ValueError(
                f"Could not parse SOC target from soc_terminal_strategy="
                f"{soc_terminal_strategy!r}. Expected 'fixed_<percent>', e.g. 'fixed_80'."
            ) from exc
        if not 0.0 <= target_fraction <= 1.0:
            raise ValueError(
                f"soc_terminal_strategy={soc_terminal_strategy!r} implies a target of "
                f"{target_fraction:.0%}, which is outside [0%, 100%]."
            )
        midpoint = target_fraction * batt["E_capacity"]
        band = 0.05 * batt["E_capacity"]
        m.soc_terminal_lb = pyo.Constraint(
            expr=m.soc[max(HOURS)] >= max(batt["SOC_min"], midpoint - band)
        )
        m.soc_terminal_ub = pyo.Constraint(
            expr=m.soc[max(HOURS)] <= min(batt["SOC_max"], midpoint + band)
        )
    elif soc_terminal_strategy == "charge_bias":
        # No hard terminal constraint: the soft terminal-SOC reward in the
        # objective biases the end-of-window SOC toward fully charged while
        # still allowing deep discharge to serve load when needed.
        pass
    else:
        raise ValueError(
            f"Unknown soc_terminal_strategy: {soc_terminal_strategy!r}. "
            "Options: 'track_init', 'fixed_<percent>' (e.g. 'fixed_50', 'fixed_80'), "
            "'charge_bias'"
        )

    # 8. Solar / wind availability
    def solar_limit_rule(m, t):
        return m.p_solar[t] <= m.d_solar[t]

    m.solar_limit = pyo.Constraint(m.T, rule=solar_limit_rule)

    def wind_limit_rule(m, t):
        return m.p_wind[t] <= m.d_wind[t]

    m.wind_limit = pyo.Constraint(m.T, rule=wind_limit_rule)

    return m


# ── Solve ────────────────────────────────────────────────────────────────────


def solve_uc_model(m, tee=False):
    """Solve with HiGHS. Returns (results, status_ok)."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(m, tee=tee)

    status_ok = (
        results.solver.status == pyo.SolverStatus.ok
        and results.solver.termination_condition == pyo.TerminationCondition.optimal
    )

    if not status_ok:
        logger.warning(
            "UC solve did not find optimal solution: %s / %s",
            results.solver.status,
            results.solver.termination_condition,
        )

    return results, status_ok


# ── Extract results ──────────────────────────────────────────────────────────


def extract_uc_results(m):
    """
    Extract solved UC schedule into numpy arrays.

    Returns:
        dict with keys:
            u: (n_units, 24) commitment status
            v: (n_units, 24) startup indicators
            p_gas: (n_units, 24) gas power per unit (kW)
            p_solar: (24,) solar dispatch (kW)
            p_wind: (24,) wind dispatch (kW)
            p_charge: (24,) battery charge (kW)
            p_discharge: (24,) battery discharge (kW)
            p_short: (24,) unmet load (kW)
            soc: (24,) end-of-hour battery SOC (kWh)
            soc_init: float — initial SOC at solve time (kWh)
            total_cost: float — objective value ($)
    """
    n_units = len(m.I)
    T = len(m.T)

    u = np.zeros((n_units, T))
    v = np.zeros((n_units, T))
    p_gas = np.zeros((n_units, T))
    p_solar = np.zeros(T)
    p_wind = np.zeros(T)
    p_charge = np.zeros(T)
    p_discharge = np.zeros(T)
    p_short = np.zeros(T)
    soc = np.zeros(T)

    for t in range(T):
        p_solar[t] = pyo.value(m.p_solar[t])
        p_wind[t] = pyo.value(m.p_wind[t])
        p_charge[t] = pyo.value(m.p_ch[t])
        p_discharge[t] = pyo.value(m.p_dis[t])
        p_short[t] = pyo.value(m.p_short[t])
        soc[t] = pyo.value(m.soc[t])
        for idx, i in enumerate(m.I):
            u[idx, t] = round(pyo.value(m.u[i, t]))
            v[idx, t] = round(pyo.value(m.v[i, t]))
            p_gas[idx, t] = pyo.value(m.p_gas[i, t])

    return {
        "u": u,
        "v": v,
        "p_gas": p_gas,
        "p_solar": p_solar,
        "p_wind": p_wind,
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "p_short": p_short,
        "soc": soc,
        "soc_init": m.soc_init_value,
        "total_cost": pyo.value(m.cost),
    }
