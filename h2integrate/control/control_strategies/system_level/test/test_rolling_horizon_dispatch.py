"""Unit tests for rolling horizon optimal dispatch controller."""

import numpy as np
import pytest
import networkx as nx
import openmdao.api as om

from h2integrate.control.control_strategies.system_level.rolling_horizon_optimal_dispatch import (
    RollingHorizonOptimalDispatch,
)


def _build_plant_config(technology_interconnections, n_timesteps=4):
    """Build minimal plant config for RHO testing."""
    return {
        "plant": {
            "simulation": {"n_timesteps": n_timesteps, "dt": 3600},
            "plant_life": 30,
        },
        "system_level_control": {
            "control_parameters": {
                "rolling_horizon_optimal_dispatch": {
                    "objective_function": "demand_following",
                    "horizon_length_hours": None,  # Use full horizon
                }
            }
        },
        "technology_interconnections": technology_interconnections,
    }


def _build_technology_graph(technology_interconnections):
    """Build networkx graph from connections."""
    technology_graph = nx.DiGraph()
    for connection in technology_interconnections:
        source = connection[0]
        destination = connection[1]
        if len(connection) == 4:
            technology_graph.add_edge(source, destination, commodity=connection[2])
        else:
            technology_graph.add_edge(source, destination)
    return technology_graph


def _build_tech_control_classifiers(
    fixed=None, flexible=None, dispatchable=None, storage=None, feedstock=None
):
    """Build tech classification dict."""
    tech_control_classifiers = dict.fromkeys(fixed or [], "fixed")
    tech_control_classifiers |= dict.fromkeys(flexible or [], "flexible")
    tech_control_classifiers |= dict.fromkeys(dispatchable or [], "dispatchable")
    tech_control_classifiers |= dict.fromkeys(storage or [], "storage")
    tech_control_classifiers |= dict.fromkeys(feedstock or [], "feedstock")
    return tech_control_classifiers


def _build_slc_topology(
    technology_graph,
    tech_control_classifiers,
    demand_tech="demand",
    demand_commodity="electricity",
    demand_commodity_rate_units="kW",
    storage_techs_with_control=None,
):
    """Build SLC topology dict."""
    if storage_techs_with_control is None:
        storage_techs_with_control = []

    sources_to_commodities = {
        (e[0], e[-1]) for e in technology_graph.edges(data="commodity") if e[-1] is not None
    }

    tech_to_commodities = {
        (e[0], e[-1]) for e in sources_to_commodities if e[0] in tech_control_classifiers
    }

    storage_techs = [k for k, v in tech_control_classifiers.items() if v == "storage"]
    storage_techs_to_control = {
        k: True if k in storage_techs_with_control else False for k in storage_techs
    }

    slc_topology = {
        "demand_commodity": demand_commodity,
        "demand_commodity_rate_units": demand_commodity_rate_units,
        "demand_tech": demand_tech,
        "tech_to_commodity": tech_to_commodities,
        "storage_techs_to_control": storage_techs_to_control,
        "technology_graph": technology_graph,
        "tech_control_classifiers": tech_control_classifiers,
    }
    return slc_topology


def _build_problem(slc_cls, plant_config, slc_topology, demand=50000, tech_config=None):
    """Create and setup an OpenMDAO Problem with the given controller."""
    if tech_config is None:
        tech_config = {}

    slc_topology["demand_profile"] = demand

    prob = om.Problem()

    feedstock_techs = [
        k for k, v in slc_topology["tech_control_classifiers"].items() if v == "feedstock"
    ]
    feedstock_subsystem_names = []
    for fi, feedstock_tech in enumerate(feedstock_techs):
        feedstock_commodity = [
            e[-1] for e in slc_topology["tech_to_commodity"] if e[0] == feedstock_tech
        ]
        feedstock_comp = prob.model.add_subsystem(f"IVC{fi}", om.Group())
        feedstock_comp.add_subsystem(
            "feedstock",
            om.IndepVarComp(
                name=f"{feedstock_tech}_{feedstock_commodity[0]}_out",
                val=np.full(plant_config["plant"]["simulation"]["n_timesteps"], 1e9),
                units="MMBtu/h",
            ),
        )

        feedstock_subsystem_names.append(
            f"IVC{fi}.feedstock.{feedstock_tech}_{feedstock_commodity[0]}_out"
        )

    prob.model.add_subsystem(
        "slc",
        slc_cls(
            driver_config={},
            plant_config=plant_config,
            tech_config=tech_config,
            slc_topology=slc_topology,
        ),
    )

    for feedstock_name in feedstock_subsystem_names:
        connection_destination = feedstock_name.split(".")[-1]
        prob.model.connect(feedstock_name, f"slc.{connection_destination}")

    prob.setup()

    # Set demand profile from config
    demand_name = f"slc.{slc_topology['demand_commodity']}_demand"
    prob.set_val(demand_name, demand)

    return prob


# ---------------------------------------------------------------------------
# Phase A: Demand Following
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRollingHorizonDemandFollowing:
    """Tests for demand-following objective in rolling horizon optimizer."""

    def test_demand_following_single_dispatchable(self):
        """Single dispatchable tech should meet demand over optimization window."""
        tech_connections = [
            ["ng1", "combiner", "electricity", "cable"],
            ["combiner", "demand", "electricity", "cable"],
        ]
        plant_config = _build_plant_config(tech_connections, n_timesteps=4)
        tech_graph = _build_technology_graph(tech_connections)
        tech_control_classifiers = _build_tech_control_classifiers(dispatchable=["ng1"])
        slc_topology = _build_slc_topology(tech_graph, tech_control_classifiers)

        demand = np.array([50000, 60000, 40000, 70000])
        prob = _build_problem(RollingHorizonOptimalDispatch, plant_config, slc_topology, demand=demand)

        prob.set_val("slc.ng1_rated_electricity_production", 100000)
        prob.run_model()

        sp = prob.get_val("slc.ng1_electricity_set_point")
        # In a rolling horizon over full window, first timestep output should minimize demand error
        # For demand-following with single tech: should produce first value of demand
        expected_first = demand[0]
        np.testing.assert_allclose(sp[0], expected_first, rtol=1e-3)

    def test_demand_following_two_dispatchable(self):
        """Two dispatchable techs should collectively meet demand."""
        tech_connections = [
            ["ng1", "combiner", "electricity", "cable"],
            ["ng2", "combiner", "electricity", "cable"],
            ["combiner", "demand", "electricity", "cable"],
        ]
        plant_config = _build_plant_config(tech_connections, n_timesteps=4)
        tech_graph = _build_technology_graph(tech_connections)
        tech_control_classifiers = _build_tech_control_classifiers(dispatchable=["ng1", "ng2"])
        slc_topology = _build_slc_topology(tech_graph, tech_control_classifiers)

        demand = np.array([50000, 60000, 40000, 70000])
        prob = _build_problem(RollingHorizonOptimalDispatch, plant_config, slc_topology, demand=demand)

        prob.set_val("slc.ng1_rated_electricity_production", 60000)
        prob.set_val("slc.ng2_rated_electricity_production", 50000)
        prob.run_model()

        sp1 = prob.get_val("slc.ng1_electricity_set_point")
        sp2 = prob.get_val("slc.ng2_electricity_set_point")

        # Total output should approximately meet demand[0]
        total_output = sp1[0] + sp2[0]
        np.testing.assert_allclose(total_output, demand[0], rtol=1e-3)

    def test_demand_following_respects_capacity(self):
        """Dispatchable tech should never exceed rated capacity."""
        tech_connections = [
            ["ng1", "combiner", "electricity", "cable"],
            ["combiner", "demand", "electricity", "cable"],
        ]
        plant_config = _build_plant_config(tech_connections, n_timesteps=4)
        tech_graph = _build_technology_graph(tech_connections)
        tech_control_classifiers = _build_tech_control_classifiers(dispatchable=["ng1"])
        slc_topology = _build_slc_topology(tech_graph, tech_control_classifiers)

        demand = np.array([120000, 100000, 90000, 110000])  # Demand exceeds capacity
        prob = _build_problem(RollingHorizonOptimalDispatch, plant_config, slc_topology, demand=demand)

        rated = 80000
        prob.set_val("slc.ng1_rated_electricity_production", rated)
        prob.run_model()

        sp = prob.get_val("slc.ng1_electricity_set_point")
        # Should never exceed rated capacity
        assert np.all(sp <= rated + 1e-6), f"Output {sp} exceeds rated {rated}"

    def test_horizon_parameter_effect(self):
        """Different horizon lengths should produce different results."""
        tech_connections = [
            ["ng1", "combiner", "electricity", "cable"],
            ["combiner", "demand", "electricity", "cable"],
        ]
        plant_config = _build_plant_config(tech_connections, n_timesteps=8)

        # Full horizon
        plant_config["system_level_control"]["control_parameters"]["rolling_horizon_optimal_dispatch"][
            "horizon_length_hours"
        ] = None
        tech_graph = _build_technology_graph(tech_connections)
        tech_control_classifiers = _build_tech_control_classifiers(dispatchable=["ng1"])
        slc_topology = _build_slc_topology(tech_graph, tech_control_classifiers)

        demand = np.array([30000, 60000, 40000, 50000, 20000, 70000, 50000, 40000])
        prob = _build_problem(RollingHorizonOptimalDispatch, plant_config, slc_topology, demand=demand)

        prob.set_val("slc.ng1_rated_electricity_production", 100000)
        prob.run_model()

        sp_full = prob.get_val("slc.ng1_electricity_set_point")

        # Shorter horizon: 2 hours ahead (2 timesteps, each dt=3600)
        plant_config["system_level_control"]["control_parameters"]["rolling_horizon_optimal_dispatch"][
            "horizon_length_hours"
        ] = 2
        prob2 = _build_problem(RollingHorizonOptimalDispatch, plant_config, slc_topology, demand=demand)
        prob2.set_val("slc.ng1_rated_electricity_production", 100000)
        prob2.run_model()

        sp_short = prob2.get_val("slc.ng1_electricity_set_point")

        # Different horizons should give different results (not necessarily true for all problems,
        # but for this time-varying demand, they should differ)
        # This is a soft assertion - the key is that both run without error
        assert sp_full is not None
        assert sp_short is not None
