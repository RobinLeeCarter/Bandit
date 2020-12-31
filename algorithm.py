from typing import Optional, Type
import abc

import numpy as np

import problem


class Algorithm(abc.ABC):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, name: str, iterations: int = 0):
        self.name = name
        self.problem: Optional[problem.Problem] = None
        self.epoch: int = 0
        self.iteration: int = 0
        self._iterations: int = iterations

        # action and return
        self._a: int = 0
        self._r: float = 0.0

        self.av_return: np.ndarray = np.zeros(shape=self._iterations, dtype=float)
        self.av_percent: np.ndarray = np.zeros(shape=self._iterations, dtype=float)

    def set_problem(self, problem_: problem.Problem, epoch: int):
        self.problem = problem_
        self.epoch = epoch
        self.initialize()

    @abc.abstractmethod
    def initialize(self):
        pass

    def do_iteration_and_record(self, iteration: int):
        self.iteration = iteration
        self._do_iteration()
        self.record_return()

    @abc.abstractmethod
    def _do_iteration(self):
        pass

    def record_return(self):
        self.av_return[self.iteration] += \
            (1 / (self.epoch + 1)) * (self._r - self.av_return[self.iteration])

        if self._a == self.problem.optimum_action:
            percent_optimal_action = 1.0
        else:
            percent_optimal_action = 0.0
        self.av_percent[self.iteration] += \
            (1 / (self.epoch + 1)) * (percent_optimal_action - self.av_percent[self.iteration])

        # percent_return = self._problem.mean[self._a] / self._problem.optimum_return
        # print(f"action = {self._a}" +
        #       f"\tmean = {self._problem.mean[self._a]:.2f}" +
        #       f"\toptimum={self._problem.optimum_return:.2f}" +
        #       f"\tpercent_return={percent_return:.2f}")
