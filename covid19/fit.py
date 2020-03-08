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
    start = attr.attrib()
    stop = attr.attrib()

    @classmethod
    def from_frame(cls, y, data, start=None, stop=None, p0=P0):
        t_0_guess, T_d_guess = p0

        data_fit = data[start:stop]

        x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
        log2_y = np.log2(data_fit[y].values)

        t_fit = data_fit.index.values[np.isfinite(log2_y)]
        x_fit = x_norm[np.isfinite(log2_y)]
        log2_y_fit = log2_y[np.isfinite(log2_y)]

        (t_0_norm, T_d_norm), _ = scipy.optimize.curve_fit(linear, x_fit, log2_y_fit)

        T_d = T_d_norm * T_d_guess
        t_0 = t_0_guess + t_0_norm * T_d_guess

        return cls(t_0, T_d, start=t_fit[0], stop=t_fit[-1])

    @property
    def T_d_days(self):
        return self.T_d / np.timedelta64(1, "D")

    def predict(self, t):
        if isinstance(t, str):
            t = np.datetime64(t)
        return 2 ** linear(t, self.t_0, self.T_d)

    def __str__(self):
        return f"t_0='{self.t_0}', T_d_days={self.T_d_days:.2f}, start={self.start}, stop={self.stop}"

    def shift(self, offset):
        if isinstance(offset, (float, int)):
            offset = np.timedelta64(int(offset * 24 * 60 * 60), "s")
        t_0 = self.t_0 + offset
        start = np.datetime64(self.start) + offset
        stop = np.datetime64(self.stop) + offset
        return self.__class__(t_0, self.T_d, start=start, stop=stop)

    def scale(self, scale):
        offset = -np.log2(scale) * self.T_d
        t_0 = self.t_0 + offset
        return self.__class__(t_0, self.T_d, start=self.start, stop=self.stop)
