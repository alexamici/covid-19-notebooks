import math

import attr
import numpy as np
import scipy.optimize


def exponential(x, a, b):
    return a * b ** x


def linear(x, y0, m):
    return m * x + y0


@attr.attrs()
class ExponentialFit:
    r"""$y = a \cdot b ^ \frac{x - x_0}{\Delta x}$"""
    x_0 = attr.attrib()
    delta_x = attr.attrib()
    start_fit = attr.attrib()
    stop_fit = attr.attrib()
    a = attr.attrib()
    b = attr.attrib()

    @classmethod
    def from_frame(cls, x, y, data, start_fit=None, stop_fit=None, x_0=np.datetime64('2020-02-20'), delta_x=np.timedelta64(1, 'D'), p0=(-0.1, 1.)):
        # x_ 0 defaults to index the Situation Reports from WHO, SR1 was on 2020-01-21

        data_fit = data[start_fit:stop_fit]
        x_fit = (data_fit[x].values - x_0) / delta_x
        y_fit = data_fit[y].values

        (start_fit, stop_fit), (a, b), _ = log_fit(x_fit, y_fit, p0=p0)

        return cls(x_0, delta_x, start_fit, stop_fit, a, b)

    def predict(self, x):
        x_fit = (x - self.x_0) / self.delta_x
        return exponential(x_fit, self.a, self.b)


def log_fit(x, y, func=linear, p0=(-0.1, 1.)):
    lny = np.log(y)
    x_fit = x[np.isfinite(lny)]
    lny_fit = lny[np.isfinite(lny)]

    params, log_cov = scipy.optimize.curve_fit(func, x_fit, lny_fit, p0=p0)

    return [x_fit[0], x_fit[-1]], [math.exp(p) for p in params], log_cov
