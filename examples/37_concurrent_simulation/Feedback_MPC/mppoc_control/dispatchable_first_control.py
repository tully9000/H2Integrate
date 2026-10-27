"""System-level controller with a gas-first, battery-as-reserve priority.

This controller differs from H2Integrate's built-in
:class:`~h2integrate.control.control_strategies.system_level.demand_following_control.DemandFollowingControl`
only in the order of steps 3 and 4: the base class dispatches storage before
dispatchable (i.e. the battery is drained on every small renewable shortfall
before gas fires), while this controller chains gas onto the renewable
residual and then chains the battery onto the gas residual, using the battery
only as a last-resort reserve. It provides a non-optimizing baseline to
compare against :class:`uc_control.UCControl`.

Dispatch priority per timestep (per commodity):

1. **Fixed** techs — always produce at rated capacity, subtracted from demand.
2. **Flexible** techs — set to rated production, subtracted from demand.
3. **Dispatchable** techs — cover only the *positive* residual after
   flexibles, split evenly across dispatchables. Each is commanded
   ``max(demand, 0) / n_dispatchable`` and the *actual* per-tech output
   (clamped internally to rated capacity) is then subtracted from the
   still-signed residual demand.
4. **Storage** techs — receive the remaining signed residual (positive =
   discharge request, negative = charge from surplus), split evenly, using
   the base class's :py:meth:`_dispatch_storage` helper so sub-controller
   semantics are preserved.

The SLC framework does not scan ``system_level_control`` for
``model_location``, so this class must be registered into
``supported_models`` in the runscript before ``H2IntegrateModel(...)`` is
instantiated (see ``run_case.py``).
"""

from __future__ import annotations

import numpy as np

from h2integrate.control.control_strategies.system_level.system_level_control_base import (
    SystemLevelControlBase,
)


class DispatchableFirstControl(SystemLevelControlBase):
    """Demand-following SLC with dispatchable prioritized over storage.

    See module docstring for the full dispatch order. This is a like-for-like
    port of the pre-SLC ``single_case`` behavior (renewables → gas → battery)
    onto the new system-level controller framework.
    """

    def compute(self, inputs, outputs):
        commodity = self.commodity
        demand = inputs[self.demand_input_name].copy()

        # --- Step 1: fixed techs always produce ---------------------------
        for fixed_tech in self.fixed_techs:
            if commodity in self._get_commodity_for_tech(fixed_tech):
                demand = self._subtract_fixed(fixed_tech, demand, commodity, inputs)

        # --- Step 2: flexible techs run at rated production ---------------
        for flexible_tech in self.flexible_techs:
            for tech_commodity in self._get_commodity_for_tech(flexible_tech):
                if tech_commodity == commodity:
                    demand = self._subtract_flexible(
                        flexible_tech, demand, commodity, inputs, outputs
                    )
                elif f"{flexible_tech}_rated_{tech_commodity}_production" in inputs:
                    # Side commodities from a multi-commodity flexible tech
                    # (matches DemandFollowingControl's handling).
                    outputs[f"{flexible_tech}_{tech_commodity}_set_point"] = inputs[
                        f"{flexible_tech}_rated_{tech_commodity}_production"
                    ] * np.ones(self.n_timesteps)

        # --- Step 3: dispatchable techs BEFORE storage --------------------
        # Cover only the positive residual. Any negative demand (renewable
        # surplus) is preserved on `demand` so it flows into storage as a
        # charge command in step 4.
        positive_demand = np.maximum(demand, 0.0)
        n_dispatchable = len(
            [d for d in self.dispatchable_techs if commodity in self._get_commodity_for_tech(d)]
        )
        for dispatchable_tech in self.dispatchable_techs:
            if commodity not in self._get_commodity_for_tech(dispatchable_tech):
                continue
            outputs[f"{dispatchable_tech}_{commodity}_set_point"] = positive_demand / n_dispatchable
            # Subtract *actual* output (clamped to rated capacity inside the
            # tech). The nonlinear solver converges this feedback loop.
            demand = demand - inputs[f"{dispatchable_tech}_{commodity}_out"]

        # --- Step 4: storage picks up whatever residual remains -----------
        # Positive residual = dispatchable at cap, unmet demand → discharge.
        # Negative residual = renewable surplus → charge.
        n_storage = len(
            [s for s in self.storage_techs if commodity in self._get_commodity_for_tech(s)]
        )
        for storage_tech in self.storage_techs:
            if commodity in self._get_commodity_for_tech(storage_tech):
                demand = self._dispatch_storage(
                    storage_tech, demand / n_storage, commodity, inputs, outputs
                )
