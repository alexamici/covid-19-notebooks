import math

import numpy as np
import scipy.optimize


def exponential(day, a, b):
    return b * a ** day


def linear(day, m, y0):
    return m * day + y0


def log_fit(x, y, data, func=linear, p0=(1.4, 1000.)):
    y = np.log(data[y])
    x = data[np.isfinite(y)][x]
    y = y[np.isfinite(y)]

    params, log_cov = scipy.optimize.curve_fit(func, x, y, p0=p0)
    exp_params = [math.exp(p) for p in params]

    return exp_params
