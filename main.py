import controller


def main():
    controller_ = controller.Controller(verbose=True)
    controller_.e_greedy_comparison()
    # controller_.sample_vs_alpha()
    controller_.run()
    controller_.graph()


if __name__ == '__main__':
    main()
