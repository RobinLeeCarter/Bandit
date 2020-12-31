import numpy as np
import matplotlib.pyplot as plt

# import e_greedy
import e_greedy_alpha
# import problem
import non_stationary_problem


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # self.epochs = 2000
        # self.time_steps = 1000
        # alg1 = e_greedy.EGreedyAlpha(epsilon=0.0, time_steps=self.time_steps)
        # alg2 = e_greedy.EGreedyAlpha(epsilon=0.01, time_steps=self.time_steps)
        # alg3 = e_greedy.EGreedyAlpha(epsilon=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2, alg3]

        # self.epochs = 2000
        # self.time_steps = 10000
        # alg1 = e_greedy.EGreedy(name="sample averages",
        #                         epsilon=0.1, time_steps=self.time_steps)
        # alg2 = e_greedy_alpha.EGreedyAlpha(name="constant step-size",
        #                                    epsilon=0.1, alpha=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2]

        # self.epochs = 200
        # self.time_steps = 1000
        # alg1 = e_greedy_alpha.EGreedyAlpha(name="optimistic greedy",
        #                                    epsilon=0.0, alpha=0.1, time_steps=self.time_steps, q1=5.0)
        # alg2 = e_greedy_alpha.EGreedyAlpha(name="realistic non-greedy",
        #                                    epsilon=0.1, alpha=0.1, time_steps=self.time_steps)
        # self.algorithms = [alg1, alg2]

        self.epochs = 2000
        self.time_steps = 1000
        alg1 = e_greedy_alpha.EGreedyAlpha(name="optimistic non-greedy biased",
                                           epsilon=0.1, alpha=0.1, time_steps=self.time_steps, q1=5.0)
        alg2 = e_greedy_alpha.EGreedyAlpha(name="optimistic non-greedy unbiased",
                                           epsilon=0.1, alpha=0.1, time_steps=self.time_steps, q1=5.0, biased=False)
        self.algorithms = [alg1, alg2]

    def run(self):
        for epoch in range(self.epochs):
            if self.verbose and epoch % 100 == 0:
                print(f"epoch = {epoch}")

            # problem_ = problem.Problem()
            problem_ = non_stationary_problem.NonStationaryProblem()
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
