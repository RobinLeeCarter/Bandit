# import numpy as np

from algorithms import e_greedy


class EGreedyAlpha(e_greedy.EGreedy):
    def __init__(self, name: str, time_steps: int = 1000,
                 epsilon: float = 0.0, alpha: float = 1.0, q1: float = 0, biased: bool = True):
        super().__init__(name, time_steps, epsilon)
        self.alpha = alpha
        self.q1 = q1
        self.biased = biased
        if not self.biased:
            self.trace_of_1: float = 0.0

    def initialize(self):
        super().initialize()
        if self.q1 != 0:
            self.Q += self.q1
        if not self.biased:
            self.trace_of_1 = 0.0

    def _set_step_size(self):
        if self.biased:
            self.step_size = self.alpha
        else:
            self.trace_of_1 += self.alpha * (1 - self.trace_of_1)
            self.step_size = self.alpha / self.trace_of_1
