import pstats
import cProfile


class Profiler:
    def __init__(self, run_profile=True):
        self.run_profile = run_profile

    def __enter__(self):
        if self.run_profile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run_profile:
            self.profiler.disable()

            self.stats = pstats.Stats(self.profiler).sort_stats("cumulative")
            self.stats.print_stats(50)
