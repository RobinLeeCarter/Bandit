import numpy as np


class StationaryProblem:
    is_stationary: bool = True
    k: int = 10
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, center: float = 0.0):
        self.center = center
        self.mean: np.ndarray = np.zeros(shape=self.k, dtype=float)
        self.variance: np.ndarray = np.ones(shape=self.k, dtype=float)
        self.std_dev: np.ndarray = np.ones(shape=self.k, dtype=float)
        self.build()
        self.std_dev = np.sqrt(self.variance)
        self.optimum_action: int = int(np.argmax(self.mean))
        self.optimum_return: float = float(np.max(self.mean))

    def build(self):
        self.mean = self.rng.normal(loc=self.center, size=self.k)

    # noinspection PyUnusedLocal
    def do_time_step(self, t: int = 0):
        pass

    def get_return(self, action: int) -> float:
        mean: float = self.mean[action]
        std_dev: float = self.std_dev[action]
        r: float = float(self.rng.normal(mean, std_dev, size=1))
        return r
