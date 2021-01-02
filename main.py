import controller


def main():
    controller_ = controller.Controller(verbose=True)
    # controller_.e_greedy_comparison()
    # controller_.sample_vs_alpha()
    # controller_.optimistic_vs_realistic()
    # controller_.optimistic_biased_vs_unbiased()
    # controller_.e_greedy_vs_ucb()
    controller_.gradient_bandit_comparison()
    controller_.run()
    controller_.graph()


if __name__ == '__main__':
    main()
