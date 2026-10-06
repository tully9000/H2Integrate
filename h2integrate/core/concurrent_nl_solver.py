import tqdm
import numpy as np
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS


class ConcurrentPlantNLSolver(NonlinearRunOnce):
    """
    Custom nonlinear solver to manage running the plant group in a loop.

    """

    def __init__(self, plant_config):
        super().__init__()
        self.plant_config = plant_config

    def solve(self):
        # Should only be used when system is the plant group
        system = self._system()

        # Find subsystems that take timestep_index as an input
        # Should only be performance models
        timestep_keys = [k for k in system._inputs.keys() if k.endswith("timestep_index")]

        n_timesteps = self.plant_config["plant"]["simulation"]["n_timesteps"]
        n_steps_per_compute = self.plant_config["plant"]["simulation"]["n_steps_per_compute"]

        # Make time stepping loop
        sim_starts = np.arange(0, n_timesteps, n_steps_per_compute)

        final_timestep_index = sim_starts[-1]

        # Subsystems whose compute() can be skipped on intermediate timesteps
        # (cost/finance models; see SkippableComputeMixin). Found by option
        # rather than by type to avoid coupling this solver to specific
        # baseclasses.
        skippable_subsystems = [
            s
            for s in system.system_iter(include_self=False, recurse=True)
            if "skip_compute" in getattr(s, "options", {})
        ]

        # Skip those subsystems' calculations in most of the simulation periods.
        for s in skippable_subsystems:
            s.options["skip_compute"] = True

        with Recording("NLRunOnce", 0, self) as rec:
            for ss in sim_starts:
                # Update timestep_index in all subsystems
                for tk in timestep_keys:
                    system._inputs[tk] = ss

                if ss == final_timestep_index:
                    # Allow skippable subsystems to compute once, on the final
                    # simulation period.
                    for s in skippable_subsystems:
                        s.options["skip_compute"] = False

                # Run one GS iteration on the plant group
                self._gs_iter()

            rec.abs = 0.0
            rec.rel = 0.0


class ConcurrentPlantNLBGSSolver(NonlinearBlockGS):
    """
    Custom nonlinear solver to manage running the plant group in a loop.

    """

    def __init__(self, plant_config):
        super().__init__()
        self.plant_config = plant_config
        self.options["iprint"] = 0
        # self.options["maxiter"] = 50
        # self.options["iprint"] = 0

        # import openmdao.api as om
        # recorder = om.SqliteRecorder("solver_recording.sql")
        # self.add_recorder(recorder)
        # self.recording_options["record_abs_error"] = True

    def solve(self):
        # Should only be used when system is the plant group
        system = self._system()

        # Find subsystems that take timestep_index as an input
        # Should only be performance models
        timestep_keys = [k for k in system._inputs.keys() if k.endswith("timestep_index")]

        # Find subsystems that take skip_compute as a discrete_input
        # Should only be cost models
        skip_compute_keys = [
            k for k in system._discrete_inputs.keys() if k.endswith("skip_compute")
        ]

        n_timesteps = self.plant_config["plant"]["simulation"]["n_timesteps"]
        n_steps_per_compute = self.plant_config["plant"]["simulation"]["n_steps_per_compute"]

        # Make time stepping loop
        sim_starts = np.arange(0, n_timesteps, n_steps_per_compute)

        final_timestep_index = sim_starts[-1]

        # Set skip_compute to True for relevant subsystems. This will skip
        # unnecessary computation in most of the simulation periods.
        for sk in skip_compute_keys:
            system._discrete_inputs[sk] = True

        with Recording("NLRunOnce", 0, self):
            for i in tqdm.tqdm(range(len(sim_starts))):
                # for ss in sim_starts:

                ss = sim_starts[i]

                # if (ss > 200) and (ss < 8730):
                #     continue

                # Update timestep_index in all subsystems
                for tk in timestep_keys:
                    system._inputs[tk] = ss

                if ss == final_timestep_index:
                    # Set skip_compute to False for the final simulation period so
                    # that the relevant calculations will be computed just once.
                    for sk in skip_compute_keys:
                        system._discrete_inputs[sk] = False

                try:
                    self._solve()

                    if self._iter_count > 30 and False:
                        import openmdao.api as om

                        cr = om.CaseReader(
                            "/Users/ztully/Documents/software/H2Integrate/examples/37_concurrent_simulation/model_predictive_control/run_pyomo_optimized_dispatch_out/solver_recording.sql"
                        )

                        input_vals = {}

                        case0 = cr.get_case(cr.list_cases()[0])
                        input_vals = {k: [] for k in case0.inputs.keys()}
                        output_vals = {k: [] for k in case0.outputs.keys()}

                        window_start_idx = 0

                        for i, cn in enumerate(cr.list_cases()):
                            if cn.endswith("|1"):
                                window_start_idx = i

                        for i, case_name in enumerate(cr.list_cases()):
                            if i < window_start_idx:
                                continue
                            case_i = cr.get_case(case_name)
                            for k in input_vals.keys():
                                input_vals[k].append(case_i.inputs[k])
                            for k in output_vals.keys():
                                output_vals[k].append(case_i.outputs[k])

                        import matplotlib.pyplot as plt

                        for _i, k in enumerate(input_vals.keys()):
                            fig, ax = plt.subplots(1, 1, layout="constrained")
                            ax.set_title(k)

                            inp = np.stack(input_vals[k])

                            ax.plot(inp)
                            if np.allclose(inp[0, :], inp[-1, :]):
                                plt.close(fig)

                        # []

                        for _i, k in enumerate(output_vals.keys()):
                            fig, ax = plt.subplots(1, 1, layout="constrained")
                            ax.set_title(k)

                            outp = np.stack(output_vals[k])

                            ax.plot(outp)
                            if np.allclose(outp[0, :], outp[-1, :]):
                                plt.close(fig)

                            # []

                        pass

                except Exception as err:
                    if self.options["debug_print"]:
                        self._print_exc_debug_info()
                    raise err
