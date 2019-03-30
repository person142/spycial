"""Find the largest number before exp underflows."""
import numpy as np


def find_underflow():
    # Start with a binary search
    a, b = -800, -700
    while True:
        mid = 0.5*(a + b)
        if np.exp(mid) != 0:
            b = mid
        else:
            a = mid
        if abs(b - a) < 1e-12:
            break

    # Polish with a brute force search
    while True:
        a = np.nextafter(b, -np.inf)
        res = np.exp(a)
        if res == 0:
            return b
        else:
            b = a


def main():
    a = find_underflow()
    print(
        'a = {:.20g}, exp(a) = {}, exp(nextafter(a, -inf) = {}'
        .format(a, np.exp(a), np.exp(np.nextafter(a, -np.inf)))
    )


if __name__ == '__main__':
    main()
