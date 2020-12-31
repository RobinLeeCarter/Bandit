import numpy as np
import matplotlib.pyplot as plt

import problems
import algorithms


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # self.epochs = 2000
        # self.time_steps = 1000
        # alg1 = algorithms.EGreedyAlpha(name="alg1", epsilon=0.0, time_steps=self.time_steps)
        # alg2 = algorithms.EGreedyAlpha(name="alg2", epsilon=0.01, time_steps=self.time_steps)
        # alg3 = algorithms.EGreedyAlpha(name="alg3", epsilon=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2, alg3]

        # self.epochs = 2000
        # self.time_steps = 10000
        # alg1 = algorithms.EGreedy(name="sample averages",
        #                         epsilon=0.1, time_steps=self.time_steps)
        # alg2 = algorithms.EGreedyAlpha(name="constant step-size",
        #                                    epsilon=0.1, alpha=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2]

        # self.epochs = 200
        # self.time_steps = 1000
        # alg1 = algorithms.EGreedyAlpha(name="optimistic greedy",
        #                                    epsilon=0.0, alpha=0.1, time_steps=self.time_steps, q1=5.0)
        # alg2 = algorithms.EGreedyAlpha(name="realistic non-greedy",
        #                                    epsilon=0.1, alpha=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2]

        # self.epochs = 2000
        # self.time_steps = 1000
        # alg1 = algorithms.EGreedyAlpha(name="optimistic non-greedy biased",
        #                                epsilon=0.1, alpha=0.1, time_steps=self.time_steps, q1=5.0)
        # alg2 = algorithms.EGreedyAlpha(name="optimistic non-greedy unbiased",
        #                                epsilon=0.1, alpha=0.1, time_steps=self.time_steps, q1=5.0, biased=False)
        # self.algorithms = [alg1, alg2]

        # self.epochs = 2000
        # self.time_steps = 1000
        # alg1 = algorithms.EGreedy(name="e-greedy", epsilon=0.1, time_steps=self.time_steps)
        # alg2 = algorithms.UCB(name="UCB", time_steps=self.time_steps, c=2.0)
        # self.algorithms = [alg1, alg2]

        self.epochs = 2000
        self.time_steps = 1000
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

            problem_ = problems.StationaryProblem(center=4.0)
            # problem_ = problems.NonStationaryProblem()
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
