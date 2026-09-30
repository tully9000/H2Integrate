"""Unit-commitment (UC) system-level controller for the ``single_case_uc`` plant.

This is the UC-optimal counterpart to ``single_case_slc``'s
:class:`DispatchableFirstControl`. Instead of the greedy renewables -> gas ->
battery priority, :class:`UCControl` plans gas commitment/output and battery
charge/discharge by solving a **receding-horizon unit-commitment MILP**
(``uc_model.py``, HiGHS via Pyomo): at each step a lookahead window (default
24 h) is optimized from the current battery SOC and gas commitment, but only
the first ``uc_control_step`` hours (default 1) are implemented before rolling
forward. With a one-hour control step this is a true hourly MPC, so the
end-of-window terminal SOC handling stays far from every applied decision and
barely distorts the implemented dispatch.

Dispatch construction per timestep (electricity):

1. **Fixed** techs (if any) always produce; their output is subtracted from
   the demand the UC must serve.
2. **Flexible** techs (solar, wind) are commanded at the UC-planned dispatch
   (``p_solar`` / ``p_wind``), i.e. curtailed to what the UC actually uses.
   Their *available* generation is passed to the UC as curtailable
   ``solar_profile`` / ``wind_profile``. To command curtailment without losing
   the availability signal (a single OpenMDAO variable carries both the
   command result and the read), availability is measured while commanding
   rated, then frozen after a short warm-up (see ``compute``).
3. **Dispatchable** techs (natural gas) receive the UC gas schedule directly as
   their set-point, split evenly if there is more than one. The plant topology
   combines gas + renewables into a single supply stream that feeds the
   battery's charge input, so gas can charge the battery per the UC plan.
4. **Storage** (battery) is commanded to the UC net-dispatch (discharge minus
   charge) via the base class :meth:`_dispatch_storage` helper, so the
   sub-controller command semantics match ``DispatchableFirstControl``.

Because the flexible set-points and the demand input do not depend on the
gas/battery set-points, the available-renewables signal is stable once the
Gauss-Seidel solver settles the renewables (after the first iteration). The UC
schedule is therefore cached and only re-solved when the demand or the
available-renewables arrays change.

The class is registered into ``supported_models`` in ``run_case.py`` (the SLC
framework does not scan ``system_level_control`` for ``model_location``).
"""

from __future__ import annotations

import time
import logging

import numpy as np
from mppoc_control.uc_model import build_uc_model, solve_uc_model, extract_uc_results

from h2integrate.control.control_strategies.system_level.system_level_control_base import (
    SystemLevelControlBase,
)


logger = logging.getLogger(__name__)

# 1 MMBtu/h expressed in kW_thermal (used to convert heat rate / gas price).
KW_TH_PER_MMBTU_HR = 1.0 / (3.412142 / 1000)  # ≈ 293.07 kW_th per MMBtu/h
KW_TO_MMBTU_HR = 3.412142 / 1000  # $/MMBtu -> $/kWh_th multiplier


class UCControl(SystemLevelControlBase):
    """System-level controller that dispatches via a rolling-horizon UC MILP."""

    def setup(self):
        super().setup()

        cp = (
            self.options["plant_config"]["system_level_control"].get("control_parameters", {}) or {}
        )

        # Gas and battery MILP parameters are built per-compute by
        # ``_build_params`` so they can track OpenMDAO inputs once sizing
        # becomes a design variable. Declared here for clarity.
        self._gas_unit_params = None
        self._battery_template = None
        self._soc_init = None

        # Receding-horizon parameters: solve a `horizon`-hour lookahead every
        # step, but only *implement* the first `control_step` hours before
        # rolling forward. With control_step=1 this is a true hourly MPC where
        # the terminal constraint is always `horizon` hours away from every
        # applied decision, so its effect on dispatch nearly vanishes.
        self._horizon = int(cp.get("uc_horizon", 24))
        self._control_step = int(cp.get("uc_control_step", 1))

        # Optional cap on the number of MILP-solved timesteps, for fast
        # pipeline testing only. Beyond the cap, gas covers the residual and
        # the battery is idle (same fallback as a failed solve). 0 = full year.
        self._max_solve_steps = int(cp.get("uc_max_solve_steps", 0)) or None

        # UC schedule cache (keyed on demand + available-renewables arrays).
        self._cache_key = None
        self._gas_schedule = np.zeros(self.n_timesteps)
        self._batt_schedule = np.zeros(self.n_timesteps)
        self._solar_schedule = np.zeros(self.n_timesteps)
        self._wind_schedule = np.zeros(self.n_timesteps)

        self.soc_store = np.zeros(self.n_timesteps)

        # Cumulative wall-clock time spent in the rolling-horizon UC MILP
        # solves (s), exposed as an OpenMDAO output so it is captured by the
        # SQL recorder. Accumulated across every (re)solve in
        # ``_solve_rolling_uc``; cache hits add nothing.
        self._total_solve_wall_s = 0.0
        self.add_output(
            "uc_solve_wall_s",
            val=0.0,
            units="s",
            desc="Cumulative wall-clock time spent in UC MILP solves",
        )

        self.add_input("battery_SOC", shape=self.n_timesteps, units="unitless")

        for f in self.flexible_techs:
            self.add_input(f"{f}_uncurtailed_electricity_out", shape=self.n_timesteps, units="kW")

        # Renewable-availability freeze state. The flexible-tech ``_out`` input
        # equals true availability only when the tech was commanded at its
        # rated set-point on the *previous* solver iteration (the performance
        # model clips its output to the command). We therefore command rated
        # during a short warm-up, capture availability once, and freeze it —
        # after which we can command curtailment (the UC-planned dispatch)
        # without corrupting the availability signal, since a single OpenMDAO
        # variable carries both the command result and the availability read.
        self._avail_frozen = False
        self._prev_rated = False
        self._avail_by_tech = None
        # Rated flexible-tech production seen when availability was frozen.
        # If it changes (renewable sizing as a design variable), the frozen
        # availability is stale and must be re-measured.
        self._rated_key = None

        self._avail_frozen_dict = {}
        self._prev_rated_dict = {}

        n_windows = int(self.n_timesteps / self.n_steps_per_compute)
        for i in range(n_windows):
            sim_range = self._get_compute_time_range([i * self.n_steps_per_compute])
            self._avail_frozen_dict.update({sim_range: False})
            self._prev_rated_dict.update({sim_range: False})

    # ------------------------------------------------------------------
    # MILP parameter construction
    # ------------------------------------------------------------------

    def _build_params(self, inputs):
        """Build the gas / battery MILP parameter dicts for this ``compute``.

        Called every ``compute`` rather than once in ``setup`` so that sizing
        can come from OpenMDAO inputs when it becomes a design variable. To
        promote a quantity to a design variable, ``add_input`` it in ``setup``
        and read it here instead of from config; the schedule cache key already
        covers these dicts, so the UC re-solves when they change.

        Args:
            inputs: the OpenMDAO inputs vector for the current ``compute``.
        """
        cp = (
            self.options["plant_config"]["system_level_control"].get("control_parameters", {}) or {}
        )

        # ── Gas unit parameters (single aggregate NGCC, n_units = 1) ──────
        heat_rate = cp.get("heat_rate_mmbtu_per_mwh", 6.3)  # MMBtu/MWh
        gas_price = cp.get("gas_price", 6.09)  # $/MMBtu
        p_max = cp.get("gas_capacity_mw", 500.0) * 1000.0  # kW

        # Constant heat rate -> no no-load fuel; incremental heat rate in
        # kW_thermal per kW_electric.
        b_inchr = heat_rate * KW_TH_PER_MMBTU_HR / 1000.0
        gas_price_per_kwh = gas_price * KW_TO_MMBTU_HR  # $/kWh_thermal

        self._gas_unit_params = {
            "A_nolod": 0.0,
            "B_inchr": b_inchr,
            "gas_price_per_kwh": gas_price_per_kwh,
            "startup_cost": cp.get("gas_startup_cost", 5000.0),
            "P_max": p_max,
            "P_min": cp.get("gas_min_load_fraction", 0.2) * p_max,
            "ramp_rate": cp.get("gas_ramp_fraction", 1.0) * p_max,
            "min_up_time": int(cp.get("gas_min_up_time_h", 1)),
            "min_down_time": int(cp.get("gas_min_down_time_h", 1)),
        }

        # ── Battery parameters (read from tech_config so UC matches plant) ─
        batt_cfg = self.options["tech_config"]["technologies"]["battery"]["model_inputs"]
        shared = batt_cfg["shared_parameters"]
        perf = batt_cfg["performance_parameters"]

        e_capacity = float(shared["max_capacity"])  # kWh
        power = float(shared["max_charge_rate"])  # kW
        rte = float(perf.get("round_trip_efficiency", 1.0))
        eta = float(np.sqrt(rte))

        self._battery_template = {
            "P_charge": power,
            "P_discharge": power,
            "E_capacity": e_capacity,
            "SOC_min": float(perf.get("min_soc_fraction", 0.1)) * e_capacity,
            "SOC_max": float(perf.get("max_soc_fraction", 0.9)) * e_capacity,
            "eta_charge": eta,
            "eta_discharge": eta,
            "soc_terminal_strategy": cp.get("soc_terminal_strategy", "fixed_50"),
        }
        # Optional explicit terminal-SOC reward ($/kWh) for the "charge_bias"
        # strategy; if omitted, uc_model defaults it to just below gas
        # marginal cost.
        if "soc_terminal_value" in cp:
            self._battery_template["soc_terminal_value"] = float(cp["soc_terminal_value"])
        self._soc_init = float(perf.get("init_soc_fraction", 0.5)) * e_capacity

    # ------------------------------------------------------------------
    # Receding-horizon UC solve
    # ------------------------------------------------------------------

    def _solve_rolling_uc(self, load, avail_solar, avail_wind, measured_SOC, simulation_range):
        """Solve the UC MILP in a receding horizon across the full year.

        At each step a ``horizon``-hour lookahead MILP is solved starting from
        the current battery SOC and gas commitment, but only the first
        ``control_step`` hours of the resulting schedule are *implemented*
        before rolling forward. With ``control_step = 1`` this is a true hourly
        model-predictive controller: the terminal SOC handling sits ``horizon``
        hours beyond every applied decision, so its distorting effect on the
        implemented dispatch nearly vanishes.

        Args:
            load: (n_timesteps,) electricity demand the UC must serve (kW),
                already net of any fixed-tech production.
            avail_solar: (n_timesteps,) available (curtailable) solar
                generation (kW).
            avail_wind: (n_timesteps,) available (curtailable) wind
                generation (kW).

        Returns:
            (gas_schedule, batt_schedule, solar_schedule, wind_schedule): each
            (n_timesteps,) arrays in kW. ``batt_schedule`` is net discharge
            (positive) / charge (negative); ``solar_schedule`` / ``wind_schedule``
            are the (possibly curtailed) UC-planned renewable dispatch.
        """
        n = self.n_timesteps
        horizon = self._horizon
        step = self._control_step
        gas = self._gas_schedule
        batt = self._batt_schedule
        solar = self._solar_schedule
        wind = self._wind_schedule

        if simulation_range.start == 0:
            soc_carry = self._soc_init
        else:
            soc_carry = self.soc_store[simulation_range.start - 1]

        commit_carry = [0]  # gas starts off
        p_max = self._gas_unit_params["P_max"]

        # Fast-test cap: only MILP-solve the first `cap` steps; the rest is
        # filled by the cheap residual fallback below.
        cap = self._max_solve_steps if self._max_solve_steps is not None else n
        n_solve = min(n, cap)

        logger.info(
            "UC rolling solve: %d steps (horizon=%dh, control_step=%dh)",
            n_solve,
            horizon,
            step,
        )
        t_start = time.monotonic()
        next_report = time.monotonic()

        t = simulation_range.start
        while t < simulation_range.stop:
            # if t > 200:
            #     end = min(t + horizon, n)
            #     impl = min(step, end - t)
            #     t += impl
            #     continue

            # Lookahead window [t, end); shrinks naturally near the year's end.
            end = min(t + horizon, n)
            load_w = np.asarray(load[t:end], dtype=float)
            solar_w = np.asarray(avail_solar[t:end], dtype=float)
            wind_w = np.asarray(avail_wind[t:end], dtype=float)

            params = {
                "n_units": 1,
                "unit_params": [self._gas_unit_params],
                "load_profile": load_w,
                "solar_profile": solar_w,
                "wind_profile": wind_w,
                "battery": {**self._battery_template, "SOC_init": soc_carry},
                "initial_commitment": commit_carry,
            }

            m = build_uc_model(params)
            _, ok = solve_uc_model(m)

            # Number of hours to actually implement from this solve.
            impl = min(step, end - t)
            if ok:
                res = extract_uc_results(m)
                g = res["p_gas"].sum(axis=0)
                b = res["p_discharge"] - res["p_charge"]
                gas[t : t + impl] = g[:impl]
                batt[t : t + impl] = b[:impl]
                solar[t : t + impl] = res["p_solar"][:impl]
                wind[t : t + impl] = res["p_wind"][:impl]

                soc_carry = float(res["soc"][impl - 1])
                # soc_carry = measured_SOC[impl-1] * self._battery_template["E_capacity"]

                commit_carry = [int(round(res["u"][0, impl - 1]))]
                self.soc_store[t : t + impl] = res["soc"][:impl]

            else:
                # Fallback: gas covers the positive residual, battery idle,
                # renewables run uncurtailed (use all available).
                logger.warning(
                    "UC solve failed for window [%d:%d]; using residual fallback", t, end
                )
                resid = np.maximum(load_w[:impl] - solar_w[:impl] - wind_w[:impl], 0.0)
                gas[t : t + impl] = np.minimum(resid, p_max)
                batt[t : t + impl] = 0.0
                solar[t : t + impl] = solar_w[:impl]
                wind[t : t + impl] = wind_w[:impl]
                # leave soc_carry / commit_carry unchanged

            t += impl

            # Periodic progress report (roughly every 10 s of wall time).
            now = time.monotonic()
            if now >= next_report or t >= n_solve:
                elapsed = now - t_start
                rate = t / elapsed if elapsed > 0 else 0.0
                eta = (n_solve - t) / rate if rate > 0 else float("nan")
                logger.info(
                    "UC progress: %d/%d steps (%.0f%%), %.1f steps/s, elapsed %.0fs, ETA %.0fs",
                    t,
                    n_solve,
                    100.0 * t / n_solve,
                    rate,
                    elapsed,
                    eta,
                )
                next_report = now + 10.0

        # Fill any uncomputed tail (fast-test cap) with the residual fallback.
        if t < n:
            resid = np.maximum(load[t:] - avail_solar[t:] - avail_wind[t:], 0.0)
            gas[t:] = np.minimum(resid, p_max)
            solar[t:] = avail_solar[t:]
            wind[t:] = avail_wind[t:]

        self._total_solve_wall_s += time.monotonic() - t_start

        return (
            gas[simulation_range],
            batt[simulation_range],
            solar[simulation_range],
            wind[simulation_range],
        )

    # ------------------------------------------------------------------
    # OpenMDAO compute
    # ------------------------------------------------------------------

    def compute(self, inputs, outputs):
        simulation_range = self._get_compute_time_range(inputs["timestep_index"])

        commodity = self.commodity
        demand = inputs[self.demand_input_name].copy()

        if simulation_range.start == 0:
            # Rebuild MILP parameters from config/inputs before anything reads them
            # (_warmup_dispatch and _solve_rolling_uc both need _gas_unit_params).
            self._build_params(inputs)

        # ── Step 1: fixed techs always produce; net them out of demand ────
        load = demand.copy()
        for fixed_tech in self.fixed_techs:
            if commodity in self._get_commodity_for_tech(fixed_tech):
                load = self._subtract_fixed(fixed_tech, load, commodity, inputs)

        # Flexible (renewable) techs serving this commodity, and whether each
        # is wind-like (classified by name; everything else is solar-like).
        flex_techs = [
            f for f in self.flexible_techs if commodity in self._get_commodity_for_tech(f)
        ]
        is_wind = {f: "wind" in f.lower() for f in flex_techs}

        # Read the current (uncurtailed) availability from each flexible tech's
        # ``_out``. This is only the true availability when the tech was
        # commanded at its rated set-point on the previous iteration; see the
        # freeze logic below.
        avail_read = {f: inputs[f"{f}_uncurtailed_{commodity}_out"].copy() for f in flex_techs}
        # avail_read = {f: inputs[f"{f}_{commodity}_out"].copy() for f in flex_techs}

        # ── Invalidate frozen availability if rated capacity changed ──────
        # Availability is a *measured* quantity (the perf model's output under
        # a rated command), so it cannot be rescaled analytically when capacity
        # changes — a new rated command has to be re-measured. Without this,
        # renewable sizing as a design variable would solve every iteration
        # against the first iteration's availability, silently and with no
        # cache miss (load and avail_* would both be unchanged).
        rated_key = tuple(
            np.asarray(inputs[f"{f}_rated_{commodity}_production"]).tobytes() for f in flex_techs
        )
        if rated_key != self._rated_key:
            if self._avail_frozen:
                logger.info("Rated flexible capacity changed; re-measuring availability")
            self._avail_frozen = False
            self._prev_rated = False
            self._rated_key = rated_key

        # ── Establish frozen renewable availability ───────────────────────
        # if not self._avail_frozen:
        if not self._avail_frozen_dict[simulation_range]:
            # Command every flexible tech at rated so next iteration's ``_out``
            # reports true availability.
            for f in flex_techs:
                outputs[f"{f}_{commodity}_set_point"] = inputs[
                    f"{f}_rated_{commodity}_production"
                ] * np.ones(self.n_timesteps)
            # Flexible techs producing a *different* commodity: hold at rated.
            for flexible_tech in self.flexible_techs:
                for tech_commodity in self._get_commodity_for_tech(flexible_tech):
                    if tech_commodity != commodity and (
                        f"{flexible_tech}_rated_{tech_commodity}_production" in inputs
                    ):
                        outputs[f"{flexible_tech}_{tech_commodity}_set_point"] = inputs[
                            f"{flexible_tech}_rated_{tech_commodity}_production"
                        ] * np.ones(self.n_timesteps)

            if self._prev_rated_dict[simulation_range]:
                # ``avail_read`` now reflects a full rated command → trust and
                # freeze it.
                if not self._avail_by_tech:
                    self._avail_by_tech = avail_read
                else:
                    for k in self._avail_by_tech.keys():
                        # self._avail_by_tech[k][simulation_range] = avail_read[k][simulation_range]
                        self._avail_by_tech[k] = avail_read[k]
                self._avail_frozen_dict[simulation_range] = True

            self._prev_rated_dict[simulation_range] = True

            if not self._avail_frozen_dict[simulation_range]:
                # Warm-up: availability not yet trusted. Command a safe residual
                # gas dispatch and idle battery so the solver has a consistent
                # state; renewables already commanded at rated above.
                self._warmup_dispatch(load, avail_read, commodity, inputs, outputs)
                return
            # Availability was just frozen; fall through to the real solve,
            # overwriting the rated renewable set-points with the curtailed
            # UC-planned dispatch.

        # ── Availability is frozen and valid from here ────────────────────
        avail_by_tech = self._avail_by_tech
        avail_solar = sum(
            (avail_by_tech[f] for f in flex_techs if not is_wind[f]),
            np.zeros(self.n_timesteps),
        )
        avail_wind = sum(
            (avail_by_tech[f] for f in flex_techs if is_wind[f]),
            np.zeros(self.n_timesteps),
        )

        # ── Solve (or reuse cached) rolling-horizon UC ────────────────────
        # Key on everything the MILP consumes, not just the profiles, so the
        # cache stays correct once gas/battery sizing become OpenMDAO inputs
        # rather than static config.
        key = (
            load.tobytes(),
            avail_solar.tobytes(),
            avail_wind.tobytes(),
            repr(sorted(self._gas_unit_params.items())),
            repr(sorted(self._battery_template.items())),
            self._soc_init,
            simulation_range.start,
        )
        if key != self._cache_key:
            (
                self._gas_schedule[simulation_range],
                self._batt_schedule[simulation_range],
                self._solar_schedule[simulation_range],
                self._wind_schedule[simulation_range],
            ) = self._solve_rolling_uc(
                load, avail_solar, avail_wind, inputs["battery_SOC"], simulation_range
            )
            self._cache_key = key
        else:
            logger.debug("UC schedule cache hit; reusing previous solve")

        # ── Step 2: flexible techs (solar, wind) at the UC dispatch ───────
        # The UC returns aggregate solar/wind dispatch (possibly curtailed);
        # split it back across the individual techs in proportion to each
        # tech's availability so renewables are curtailed exactly as planned.
        for f in flex_techs:
            total = avail_wind if is_wind[f] else avail_solar
            sched = self._wind_schedule if is_wind[f] else self._solar_schedule
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(total > 0, avail_by_tech[f] / total, 0.0)
            outputs[f"{f}_{commodity}_set_point"][simulation_range] = (
                share[simulation_range] * sched[simulation_range]
            )

        # ── Step 3: dispatchable techs (gas) get the UC schedule ──────────
        dispatchables = [
            d for d in self.dispatchable_techs if commodity in self._get_commodity_for_tech(d)
        ]
        n_dispatchable = max(len(dispatchables), 1)
        for dispatchable_tech in dispatchables:
            outputs[f"{dispatchable_tech}_{commodity}_set_point"][simulation_range] = (
                self._gas_schedule[simulation_range] / n_dispatchable
            )

        # ── Step 4: storage (battery) gets the UC net dispatch ────────────
        # Positive = discharge, negative = charge. _dispatch_storage translates
        # this into the sub-controller command semantics.
        storages = [s for s in self.storage_techs if commodity in self._get_commodity_for_tech(s)]
        n_storage = max(len(storages), 1)
        for storage_tech in storages:
            self._dispatch_storage(
                storage_tech, self._batt_schedule / n_storage, commodity, inputs, outputs
            )

        # Expose cumulative UC solve wall time for the SQL recorder.
        outputs["uc_solve_wall_s"] = self._total_solve_wall_s

    def _warmup_dispatch(self, load, avail_read, commodity, inputs, outputs):
        """Provisional dispatch used while renewable availability is not yet
        trusted: gas covers the positive residual (capped at capacity) and the
        battery is idle. Keeps the Gauss-Seidel loop consistent for the one or
        two iterations before availability is frozen."""
        avail = sum(avail_read.values(), np.zeros(self.n_timesteps))
        p_max = self._gas_unit_params["P_max"]
        resid = np.minimum(np.maximum(load - avail, 0.0), p_max)

        dispatchables = [
            d for d in self.dispatchable_techs if commodity in self._get_commodity_for_tech(d)
        ]
        n_dispatchable = max(len(dispatchables), 1)
        for dispatchable_tech in dispatchables:
            outputs[f"{dispatchable_tech}_{commodity}_set_point"] = resid / n_dispatchable

        storages = [s for s in self.storage_techs if commodity in self._get_commodity_for_tech(s)]
        n_storage = max(len(storages), 1)
        idle = np.zeros(self.n_timesteps)
        for storage_tech in storages:
            self._dispatch_storage(storage_tech, idle / n_storage, commodity, inputs, outputs)
