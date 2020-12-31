import controller


def main():
    controller_ = controller.Controller(verbose=True)
    controller_.run()
    controller_.graph()


if __name__ == '__main__':
    main()
