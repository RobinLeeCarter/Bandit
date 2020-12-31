import numpy as np

from problems import stationary_problem


class NonStationaryProblem(stationary_problem.StationaryProblem):
    is_stationary: bool = False

    def __init__(self, center: float = 0.0):
        super().__init__(center)

    # noinspection PyUnusedLocal
    def do_time_step(self, t: int = 0):
        self.mean += self.rng.normal(loc=0, scale=0.01, size=self.k)
        self.optimum_action: int = int(np.argmax(self.mean))
        self.optimum_return: float = float(np.max(self.mean))
