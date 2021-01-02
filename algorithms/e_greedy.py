# import numpy as np

from algorithms import greedy


class EGreedy(greedy.Greedy):
    def __init__(self, name: str = "no name", time_steps: int = 1000, epsilon: float = 0.0):
        super().__init__(name, time_steps)
        self.epsilon: float = epsilon   # explore rate

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

    def get_random_action(self) -> int:
        a = self.rng.integers(low=0, high=self.problem.k)
        return a
