import mpmath


def zeta_even_integers(N):
    values = []
    for n in range(0, N + 1):
        zeta_2n = (
            (-1)**(n + 1)
            * mpmath.bernoulli(2 * n)
            * (2 * mpmath.pi)**(2 * n)
            / (2 * mpmath.factorial(2 * n))
        )
        values.append(zeta_2n)
    return values


def main():
    with mpmath.workdps(100):
        zeta_values = zeta_even_integers(27)
        for n in zeta_values:
            mpmath.nprint(n, 20)


if __name__ == '__main__':
    main()
