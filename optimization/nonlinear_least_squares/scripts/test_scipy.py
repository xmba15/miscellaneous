#!/usr/bin/env python
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt


def _gen_data(a=-0.5, b=2.8, c=1.0, num_samples=100, seed=2021):
    np.random.seed(seed)
    x = np.array([1.0 * i / num_samples for i in range(num_samples)])
    y = np.exp(a * x * x + b * x + c) + np.random.normal(0, 1, num_samples)
    return x, y


def _func(coeffs, x):
    return np.exp(coeffs[0] * x * x + coeffs[1] * x + coeffs[2])


def _residual_func(coeffs, x, y):
    return _func(coeffs, x) - y


def main():
    x_train, y_train = _gen_data()

    coeffs0 = np.array([0.0, 0.0, 0.0])
    res_lsq = least_squares(_residual_func, coeffs0, args=(x_train, y_train))
    coeffs = res_lsq.x
    print("estimated coefficients: {}".format(coeffs))
    y_test = _func(coeffs, x_train)

    plt.scatter(x_train, y_train, c="red")
    plt.plot(x_train, y_test, "g^")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
