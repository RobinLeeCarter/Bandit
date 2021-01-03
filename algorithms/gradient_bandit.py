import numpy as np

from algorithms import algorithm


class GradientBandit(algorithm.Algorithm):
    def __init__(self, name: str, time_steps: int = 1000, alpha: float = 0.1, baseline_enabled: bool = True):
        super().__init__(name, time_steps)
        self.alpha = alpha
        self.baseline_enabled = baseline_enabled
        self._r_bar: float = 0.0
        self.baseline_alpha = 0.01
        self.N: np.ndarray = np.zeros(shape=0, dtype=float)
        self.H: np.ndarray = np.zeros(shape=0, dtype=float)
        self.PI: np.ndarray = np.zeros(shape=0, dtype=float)

    def initialize(self):
        self.N = np.zeros(shape=self.problem.k, dtype=float)
        self.H = np.zeros(shape=self.problem.k, dtype=float)
        self.PI = np.ones(shape=self.problem.k, dtype=float)/self.problem.k

    def _do_time_step(self):
        self._a = self.pick_action()
        self._r = self.problem.get_return(self._a)
        if self.baseline_enabled:
            # if self.t == 1:
            #     self._r_bar = self._r
            # else:
            if self.problem.non_stationary:
                self._r_bar += self.baseline_alpha * (self._r - self._r_bar)
            else:
                self._r_bar += (1/self.t) * (self._r - self._r_bar)

        # update preferences
        for a in range(self.problem.k):
            if a == self._a:
                self.H[a] += self.alpha * (self._r - self._r_bar) * (1 - self.PI[a])
            else:
                self.H[a] -= self.alpha * (self._r - self._r_bar) * (self.PI[a])

        # update policy
        self.PI = self.soft_max(self.H)

    def _set_step_size(self):
        self.step_size = (1 / self.N[self._a])

    def pick_action(self) -> int:
        a = self.get_pi_action()
        return a

    def get_pi_action(self) -> int:
        a = self.rng.choice(self.problem.k, p=self.PI)
        return a

    # safe soft-max must be 1-D array
    def soft_max(self, x: np.ndarray):
        # size = x.size
        max_x = np.max(x)
        safe_x = x - max_x
        exp_x = np.exp(safe_x)
        denominator = np.sum(exp_x)
        soft_max_x = exp_x / denominator
        return soft_max_x
