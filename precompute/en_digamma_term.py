"""Compute ψ(n) up to n = 50.

The values are used in a series expansion for the exponential
integral; see DLMF 8.19.8.

"""
import numpy as np
import mpmath


def digamma_at_integers(N):
    values = [np.nan]
    for n in range(1, N + 1):
        values.append(mpmath.digamma(n))
    return values


def main():
    with mpmath.workdps(50):
        values = digamma_at_integers(50)
        for value in values:
            mpmath.nprint(value, 20)


if __name__ == '__main__':
    main()
