import numpy as np
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS


class TimeMarchingPlantSolver(NonlinearBlockGS):
    """
    Custom nonlinear solver to manage running the plant group in a loop.

    """

    def __init__(self, plant_config):
        super().__init__()
        self.plant_config = plant_config
        # iprint is hardcoded to 0 for now to suppress excessive printouts from every time marching
        # step.
        # TODO Set allow the use solver options from config files?
        self.options['atol'] = 1e-6
        self.options['rtol'] = 1e-5
        self.options["iprint"] = 0
        self.options["maxiter"] = 30

        # import openmdao.api as om
        # # Enable case recording to capture iteration-by-iteration data
        # case_recorder = om.SqliteRecorder('rho_convergence_study.sql')
        # # case_recorder.options['record_residuals'] = True
        # self.add_recorder(case_recorder)


    def solve(self):
        # Should only be used when system is the plant group
        system = self._system()

        # Find subsystems that take timestep_index as an input
        # Should only be performance models
        timestep_keys = [k for k in system._inputs.keys() if k.endswith("timestep_index")]

        n_timesteps = self.plant_config["plant"]["simulation"]["n_timesteps"]
        n_timesteps_window = self.plant_config["plant"]["simulation"]["n_timesteps_window"]

        # Make time stepping loop
        sim_starts = np.arange(0, n_timesteps, n_timesteps_window)

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

        try:
            with Recording("NLRunOnce", 0, self):
                for ss in sim_starts:
                    # Update timestep_index in all subsystems
                    for tk in timestep_keys:
                        system._inputs[tk] = ss

                    if ss == final_timestep_index:
                        # Allow skippable subsystems to compute once, on the final
                        # simulation period.
                        for s in skippable_subsystems:
                            s.options["skip_compute"] = False

                    try:
                        self._solve()

                    except Exception as err:
                        if self.options["debug_print"]:
                            self._print_exc_debug_info()
                        raise err

                    if self._iter_count > 9: 
                        # self._record_debug_excessive_iterations()
                        pass


        finally:
            # A failed step must not leave cost and finance models skipped on the next run.
            for s in skippable_subsystems:
                s.options["skip_compute"] = False


    def _record_debug_excessive_iterations(self):
        import openmdao.api as om
        import matplotlib.pyplot as plt

        # # Enable case recording to capture iteration-by-iteration data
        # case_recorder = om.SqliteRecorder('rho_convergence_study.sql')
        # self.add_recorder(case_recorder)

        # self._solve()

        # This records every iteration of every solver
        # Later you can read with:
        cr = om.CaseReader(self.get_outputs_dir() / 'rho_convergence_study.sql')
        cases = cr.list_cases()  # All recorded cases

        # Get only the last call to solve's cases
        cases_trunc = []
        for c in reversed(cases):
            cases_trunc.append(c)
            if c.endswith("|1"):
                break

        cases = list(reversed(cases_trunc))

        all_inputs = []
        all_outputs = []
        all_residuals = []

        for case_id in cases:
            case = cr.get_case(case_id)
            inputs = case.inputs
            outputs = case.outputs
            residuals = case.residuals
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            all_residuals.append(residuals)



        def find_unconverged(io):

            keys = io[0].keys()
            unconverged_keys = []
            for k in keys:
                vals = np.asarray([i[k] for i in io])

                if not np.all(np.diff(vals[-2:, :], axis=0)):
                    # No difference
                    pass
                else:
                    # A difference
                    unconverged_keys.append(k)
                    pass
            return unconverged_keys

        in_unc = find_unconverged(all_inputs)
        out_unc = find_unconverged(all_outputs)


        def plot_unconverged(io, keys):
            fig, ax = plt.subplots(1, 1, layout="constrained")
            first = io[0]

            for key in keys:
                data = np.asarray([i[key] for i in io])
                normalized = data / np.max(data)

                if first[key].shape[0] == 1:
                    ax.plot(normalized, marker = ".", label=key)

                else:

                    for j in range(first[key].shape[0]):

                        if j > 2: 
                            break
                        if np.all(np.diff(normalized[-2:, j])):

                            ax.plot(normalized[:, j], marker=".", label=key)

            ax.legend()
            # ax.set_ylim([0, 1.05])


        plot_unconverged(all_inputs, in_unc)
        plot_unconverged(all_outputs, out_unc)


        pass