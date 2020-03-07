import attr
import numpy as np
import scipy.optimize


def exp2(t, t_0, T_d):
    return 2 ** ((t - t_0) / T_d)


def linear(t, t_0, T_d):
    return (t - t_0) / T_d


# good starting point for fitting most of the curves
P0 = (np.datetime64("2020-02-12", "s"), np.timedelta64(48 * 60 * 60, "s"))


@attr.attrs()
class ExponentialFit:
    r"""$f(t) = 2 ^ \frac{t - t_0}{T_d}$"""
    t_0 = attr.attrib()
    T_d = attr.attrib()
    start_fit = attr.attrib()
    stop_fit = attr.attrib()

    @classmethod
    def from_frame(cls, y, data, start_fit=None, stop_fit=None, p0=P0):
        t_0_guess, T_d_guess = p0

        data_fit = data[start_fit:stop_fit]

        x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
        log2_y = np.log2(data_fit[y].values)

        x_fit = x_norm[np.isfinite(log2_y)]
        log2_y_fit = log2_y[np.isfinite(log2_y)]

        (t_0_norm, T_d_norm), _ = scipy.optimize.curve_fit(linear, x_fit, log2_y_fit)

        T_d = T_d_norm * T_d_guess
        t_0 = t_0_guess + t_0_norm * T_d_guess

        return cls(t_0, T_d, start_fit, stop_fit)

    @property
    def T_d_days(self):
        return self.T_d / np.timedelta64(1, "D")

    def predict(self, t):
        return 2 ** linear(t, self.t_0, self.T_d)

    def __str__(self):
        return f"t_0='{self.t_0}', T_d_days={self.T_d_days:.2f}, start_fit={self.start_fit!r}, stop_fit={self.stop_fit}"
