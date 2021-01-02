from typing import Optional
import abc

import numpy as np

import problem


class Algorithm(abc.ABC):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, name: str = "no name", time_steps: int = 0):
        self.name = name
        self.problem: Optional[problem.Problem] = None
        self.epoch: int = 0
        self.t: int = 0
        self._time_steps: int = time_steps

        # action and return
        self._a: int = 0
        self._r: float = 0.0

        self.av_return: np.ndarray = np.zeros(shape=self._time_steps, dtype=float)
        self.av_percent: np.ndarray = np.zeros(shape=self._time_steps, dtype=float)

    def set_problem(self, problem_: problem.Problem, epoch: int):
        self.problem = problem_
        self.epoch = epoch
        self.initialize()

    @abc.abstractmethod
    def initialize(self):
        pass

    def do_time_step_and_record(self, t: int):
        self.t = t
        self._do_time_step()
        self.record_return()

    @abc.abstractmethod
    def _do_time_step(self):
        pass

    def record_return(self):
        self.av_return[self.t] += \
            (1 / (self.epoch + 1)) * (self._r - self.av_return[self.t])

        if self._a == self.problem.optimum_action:
            percent_optimal_action = 1.0
        else:
            percent_optimal_action = 0.0
        self.av_percent[self.t] += \
            (1 / (self.epoch + 1)) * (percent_optimal_action - self.av_percent[self.t])

        # percent_return = self._problem.mean[self._a] / self._problem.optimum_return
        # print(f"action = {self._a}" +
        #       f"\tmean = {self._problem.mean[self._a]:.2f}" +
        #       f"\toptimum={self._problem.optimum_return:.2f}" +
        #       f"\tpercent_return={percent_return:.2f}")

    def get_av_reward(self, final_steps: int = 0):
        if final_steps == 0:
            av_reward = float(np.mean(self.av_return))
        else:
            av_reward = float(np.mean(self.av_return[-final_steps:]))
        return av_reward
