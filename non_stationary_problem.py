import numpy as np

import problem


class NonStationaryProblem(problem.Problem):
    def __init__(self):
        super().__init__()

    # noinspection PyUnusedLocal
    def do_iteration(self, iteration: int = 0):
        self.mean += self.rng.normal(loc=0, scale=0.01, size=self.k)
        self.optimum_action: int = int(np.argmax(self.mean))
        self.optimum_return: float = float(np.max(self.mean))
