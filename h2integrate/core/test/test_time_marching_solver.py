from h2integrate.core.time_marching_solver import TimeMarchingPlantSolver


def test_time_marching_solver_declares_maxiter_option():
    solver = TimeMarchingPlantSolver(
        plant_config={"plant": {"simulation": {"n_timesteps": 1, "n_timesteps_window": 1}}}
    )

    assert solver.options["maxiter"] == 20
