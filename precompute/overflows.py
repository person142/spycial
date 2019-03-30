"""Compute the largest double precision number that doesn't cause
exp/cosh/sinh to overflow.

"""
import numpy as np

np.seterr(all='ignore')


def find_overflow(f, a, b):
    # Start with a binary search
    while True:
        mid = 0.5*(a + b)
        if f(mid) == np.inf:
            b = mid
        else:
            a = mid
        if abs(b - a) < 1e-12:
            break

    # Polish with a brute force search
    while True:
        b = np.nextafter(a, np.inf)
        res = np.exp(b)
        if res == np.inf:
            return a
        else:
            a = b


def main():
    a = find_overflow(np.exp, 709.7, 709.9)
    print("a = {:.20g}, np.exp(a) = {}".format(a, np.exp(a)))
    a = find_overflow(np.cosh, 710, 711)
    print("a = {:.20g}, np.cosh(a) = {}".format(a, np.cosh(a)))
    a = find_overflow(np.sinh, 710, 711)
    print("a = {:.20g}, np.sinh(a) = {}".format(a, np.sinh(a)))


if __name__ == '__main__':
    main()
