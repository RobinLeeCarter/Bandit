import numpy as np

from algorithms import greedy


class UCB(greedy.Greedy):
    def __init__(self, name: str, time_steps: int = 1000, c: float = 2.0):
        super().__init__(name, time_steps)
        self.c = c

    def initialize(self):
        self.N = np.zeros(shape=self.problem.k, dtype=float)
        self.Q = np.zeros(shape=self.problem.k, dtype=float)

    def pick_action(self) -> int:
        min_n = np.min(self.N)
        if min_n == 0:
            a = self.get_zero_action()
        else:
            a = self.get_ucb_action()
        return a

    def get_zero_action(self) -> int:
        zero_n = (self.N == 0)
        zero_a = np.flatnonzero(zero_n)
        a = self.rng.choice(zero_a)
        return a

    def get_ucb_action(self) -> int:
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        max_ucb = np.max(ucb)
        max_ucb_bool = (ucb == max_ucb)
        max_a = np.flatnonzero(max_ucb_bool)
        a = self.rng.choice(max_a)
        return a
