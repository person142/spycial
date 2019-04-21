import mpmath


def main():
    with mpmath.workdps(50):
        for n in range(0, 51):
            mpmath.nprint(mpmath.expint(n, 1), 20)


if __name__ == '__main__':
    main()
