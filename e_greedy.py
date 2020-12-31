import numpy as np

import algorithm


class EGreedy(algorithm.Algorithm):
    def __init__(self, name: str, epsilon: float = 1.0, iterations: int = 1000):
        super().__init__(name, iterations)
        self.epsilon: float = epsilon   # explore rate
        self.step_size: float = 1           # step-size
        self.N: np.ndarray = np.zeros(shape=0, dtype=float)
        self.Q: np.ndarray = np.zeros(shape=0, dtype=float)

    def initialize(self):
        self.N = np.zeros(shape=self.problem.k, dtype=float)
        self.Q = np.zeros(shape=self.problem.k, dtype=float)

    def _do_iteration(self):
        self._a = self.pick_action()
        self._r = self.problem.get_return(self._a)
        self.N[self._a] += 1
        self._set_step_size()
        self.Q[self._a] += self.step_size * (self._r - self.Q[self._a])
        # self.Q[self._a] += (1 / self.N[self._a]) * (self._r - self.Q[self._a])
        # print(f"Action={self._a}\t" +
        #       f"return={self._r:.2f}" +
        #       f"\tN[a]={self.N[self._a]}" +
        #       f"\tQ[a]={self.Q[self._a]:.2f}")

    def _set_step_size(self):
        self.step_size = (1 / self.N[self._a])

    def pick_action(self) -> int:
        if self.rng.uniform() > self.epsilon:
            # greedy_action
            a = self.get_greedy_action()
            # print(f"greedy {a}")
        else:
            # random action
            a = self.get_random_action()
            # print(f"random {a}")
        return a

    def get_greedy_action(self) -> int:
        best_q = np.max(self.Q)
        best_q_bool = (self.Q == best_q)
        best_a = np.flatnonzero(best_q_bool)
        a = self.rng.choice(best_a)
        return a

    def get_random_action(self) -> int:
        a = self.rng.integers(low=0, high=self.problem.k)
        return a
