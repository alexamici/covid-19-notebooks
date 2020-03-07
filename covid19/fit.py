import math

import attr
import numpy as np
import scipy.optimize


def exp2(t, t_0, dt):
    return 2 ** ((t - t_0) / dt)


def linear(t, t_0, dt):
    return (t - t_0) / dt


@attr.attrs()
class ExponentialFit:
    r"""$y = 2 ^ \frac{t - t_0}{\delta t}$"""
    t_0 = attr.attrib()
    dt = attr.attrib()
    start_fit = attr.attrib()
    stop_fit = attr.attrib()

    @classmethod
    def from_frame(
        cls,
        y,
        data,
        start_fit=None,
        stop_fit=None,
        p0=(np.datetime64("2020-02-12T00:00:00"), np.timedelta64(48 * 60 * 60, "s")),
    ):
        t_0_guess, dt_guess = p0

        data_fit = data[start_fit:stop_fit]

        x_norm = linear(data_fit.index.values, t_0_guess, dt_guess)
        log2_y = np.log2(data_fit[y].values)

        x_fit = x_norm[np.isfinite(log2_y)]
        log2_y_fit = log2_y[np.isfinite(log2_y)]

        (t_0_norm, dt_norm), _ = scipy.optimize.curve_fit(
            linear, x_fit, log2_y_fit, p0=(1.0, 1.0)
        )

        dt = dt_norm * dt_guess
        t_0 = t_0_guess + t_0_norm * dt_guess

        return cls(t_0, dt, start_fit, stop_fit)

    def predict(self, t):
        return 2 ** linear(t, self.t_0, self.dt)
