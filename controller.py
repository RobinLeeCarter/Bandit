from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

import problem
import algorithms


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.epochs: int = 0
        self.time_steps: int = 0
        self.non_stationary = False
        self.problem_center = 0.0
        self.algorithms: List[algorithms.Algorithm] = []
        self.build()

    def build(self):
        # self.e_greedy_comparison()
        # self.sample_vs_alpha()
        # self.optimistic_vs_realistic()
        # self.optimistic_biased_vs_unbiased()
        # self.e_greedy_vs_ucb()
        self.gradient_bandit_comparison()
        self.run()
        self.graph()

    def e_greedy_comparison(self):
        self.epochs = 2000
        self.time_steps = 1000

        alg1 = algorithms.EGreedy(name="greedy", epsilon=0.0, time_steps=self.time_steps)
        alg2 = algorithms.EGreedy(name="ε=0.01", epsilon=0.01, time_steps=self.time_steps)
        alg3 = algorithms.EGreedy(name="ε=0.1", epsilon=0.1, time_steps=self.time_steps)
        self.algorithms = [alg1, alg2, alg3]

    def sample_vs_alpha(self):
        self.epochs = 2000
        self.time_steps = 10000
        self.non_stationary = True

        alg1 = algorithms.EGreedy(name="sample averages", time_steps=self.time_steps,
                                  epsilon=0.1)
        alg2 = algorithms.EGreedyAlpha(name="constant step-size", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1)
        self.algorithms = [alg1, alg2]

    def optimistic_vs_realistic(self):
        self.epochs = 2000
        self.time_steps = 1000

        alg1 = algorithms.EGreedyAlpha(name="optimistic greedy", time_steps=self.time_steps,
                                       epsilon=0.0, alpha=0.1, q1=5.0)
        alg2 = algorithms.EGreedyAlpha(name="realistic non-greedy", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1)
        self.algorithms = [alg1, alg2]

    def optimistic_biased_vs_unbiased(self):
        self.epochs = 2000
        self.time_steps = 1000

        alg1 = algorithms.EGreedyAlpha(name="optimistic non-greedy biased", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1, q1=5.0)
        alg2 = algorithms.EGreedyAlpha(name="optimistic non-greedy unbiased", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1, q1=5.0, biased=False)
        self.algorithms = [alg1, alg2]

    def e_greedy_vs_ucb(self):
        self.epochs = 2000
        self.time_steps = 1000
        alg1 = algorithms.EGreedy(name="e-greedy", time_steps=self.time_steps, epsilon=0.1)
        alg2 = algorithms.UCB(name="UCB", time_steps=self.time_steps, c=2.0)
        self.algorithms = [alg1, alg2]

    def gradient_bandit_comparison(self):
        self.epochs = 2000
        self.time_steps = 1000
        self.problem_center = 4.0

        gb = algorithms.GradientBandit
        alg1 = gb(name="alpha=0.1", time_steps=self.time_steps, alpha=0.1)
        alg2 = gb(name="alpha=0.1 no baseline", time_steps=self.time_steps, alpha=0.1, baseline_enabled=False)
        alg3 = gb(name="alpha=0.4", time_steps=self.time_steps, alpha=0.4)
        alg4 = gb(name="alpha=0.4 no baseline", time_steps=self.time_steps, alpha=0.4, baseline_enabled=False)
        self.algorithms = [alg1, alg2, alg3, alg4]

    def run(self):
        for epoch in range(self.epochs):
            if self.verbose and epoch % 100 == 0:
                print(f"epoch = {epoch}")

            problem_ = problem.Problem(center=self.problem_center, non_stationary=self.non_stationary)

            for alg in self.algorithms:
                alg.set_problem(problem_, epoch)

            for t in range(1, self.time_steps):
                problem_.do_time_step(t)
                for alg in self.algorithms:
                    alg.do_time_step_and_record(t)

    def graph(self):
        time_steps_x = np.arange(self.time_steps)
        for alg in self.algorithms:
            plt.plot(time_steps_x, alg.av_return, label=alg.name)
            plt.legend()
        plt.show()

        for alg in self.algorithms:
            plt.plot(time_steps_x, alg.av_percent, label=alg.name)
            plt.legend()
        plt.show()
