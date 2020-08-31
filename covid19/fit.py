import math

import attr
import numpy as np
import scipy.optimize
import xarray as xr


DAY = np.timedelta64(24 * 3600, "s")


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
    r2 = attr.attrib()
    start = attr.attrib()
    stop = attr.attrib()

    @classmethod
    def from_frame(cls, data, start=None, stop=None, p0=P0, min_value=9):
        t_0_guess, T_d_guess = p0

        data_fit = data[start:stop]

        x_norm = linear(data_fit.index.values, t_0_guess, T_d_guess)
        log2_y = np.log2(data_fit.values)

        t_fit = data_fit.index.values[
            np.isfinite(log2_y) & (data_fit.values >= min_value)
        ]
        x_fit = x_norm[np.isfinite(log2_y) & (data_fit.values >= min_value)]
        log2_y_fit = log2_y[np.isfinite(log2_y) & (data_fit.values >= min_value)]

        # (t_0_norm, T_d_norm), covariance = scipy.optimize.curve_fit(linear, x_fit, log2_y_fit)
        try:
            m, y, r2, _, _ = scipy.stats.linregress(x_fit, log2_y_fit)
        except ValueError:
            m, y, r2 = np.nan, np.nan, 0.
        t_0_norm = -y / m
        T_d_norm = 1 / m

        T_d = T_d_norm * T_d_guess
        t_0 = t_0_guess + t_0_norm * T_d_guess

        return cls(t_0, T_d, r2=r2, start=t_fit[0], stop=t_fit[-1])

    @classmethod
    def from_xarray(cls, data, start=None, stop=None, p0=P0, min_value=9, x="time", valid_ratio=0):
        assert isinstance(data, xr.DataArray)
        t_0_guess, T_d_guess = p0

        data_fit = data.sel(**{x: slice(start, stop)})

        x_norm = linear(data_fit.coords[x].values, t_0_guess, T_d_guess)
        log2_y = np.log2(data_fit.values)

        t_fit = data_fit.coords[x].values[
            np.isfinite(log2_y) & (data_fit.values >= min_value)
        ]
        fit_length = t_fit[-1] - t_fit[0] if t_fit.size > 0 else np.timedelta64(0)
        x_fit = x_norm[np.isfinite(log2_y) & (data_fit.values >= min_value)]
        log2_y_fit = log2_y[np.isfinite(log2_y) & (data_fit.values >= min_value)]

        # (t_0_norm, T_d_norm), covariance = scipy.optimize.curve_fit(linear, x_fit, log2_y_fit)
        try:
            m, y, r2, _, _ = scipy.stats.linregress(x_fit, log2_y_fit)
        except ValueError:
            m, y, r2 = np.nan, np.nan, 0.
            t_fit = [start, stop]
        t_0_norm = -y / m
        T_d_norm = 1 / m

        T_d = T_d_norm * T_d_guess
        t_0 = t_0_guess + t_0_norm * T_d_guess

        if abs(T_d / DAY) * valid_ratio > abs(fit_length / DAY):
            t_0, T_d, r2 = np.datetime64('Not a Time'), np.timedelta64('Not a Time'),  0.

        return cls(t_0, T_d, r2=r2, start=t_fit[0], stop=t_fit[-1])

    @property
    def T_d_days(self):
        return self.T_d / np.timedelta64(1, "D")

    def predict(self, t):
        if isinstance(t, str):
            t = np.datetime64(t)
        return 2 ** linear(t, self.t_0, self.T_d)

    def __str__(self):
        return f"T_d={self.T_d_days:.2f} t_0='{str(self.t_0)[:10]}' r^2={self.r2:.3f} start='{str(self.start)[:10]}' stop='{str(self.stop)[:10]}'"

    def shift(self, offset):
        if isinstance(offset, (float, int)):
            offset = np.timedelta64(int(offset * 24 * 60 * 60), "s")
        t_0 = self.t_0 + offset
        start = np.datetime64(self.start) + offset
        stop = np.datetime64(self.stop) + offset
        return self.__class__(t_0, self.T_d, r2=self.r2, start=start, stop=stop)

    def scale(self, scale):
        offset = -np.log2(scale) * self.T_d
        t_0 = self.t_0 + offset
        return self.__class__(
            t_0, self.T_d, r2=self.r2, start=self.start, stop=self.stop
        )


def find_best_fits_size(y, data, size):
    fits = []
    for start, stop in zip(data.index, data.index[size - 1 :]):
        fits.append(ExponentialFit.from_frame(y, data, start, stop))
    return sorted(fits, key=lambda x: -x.r2)


def find_best_fits(y, data, min_r2=0.99, min_size=3, max_size=16):
    fits = {}
    for size in range(max_size, min_size - 1, -1):
        res = [f for f in find_best_fits_size(y, data, size) if f.r2 >= min_r2]
        if res:
            fits[size] = res
    return fits


def fit_exponential_segments(data, breaks=(None, None), break_length=DAY, valid_ratio=0, **kwargs):
    starts = [np.datetime64(b, "s") if b is not None else b for b in breaks]
    stops = [s - break_length if s is not None else s for s in starts[1:]]
    exponential_segments = []
    for start, stop in zip(starts, stops):
        try:
            if isinstance(data, xr.DataArray):
                fit = ExponentialFit.from_xarray(data, start=start, stop=stop, valid_ratio=valid_ratio, **kwargs)
            else:
                fit = ExponentialFit.from_frame(data, start=start, stop=stop, **kwargs)
            if np.isfinite(fit.T_d):
                exponential_segments.append(fit)
        except ValueError:
            print(f"skipping start={start} stop={stop}")
    return exponential_segments


def fit_exponential_outbreaks(sources, outbreaks):
    exponential_outbreaks = []
    for o in outbreaks:
        outbreak = o.copy()
        for source in sources:
            if outbreak["location"] in source.location:
                data = source.sel(location=outbreak["location"])
                fit = ExponentialFit.from_xarray(
                    data, outbreak["start"], outbreak["stop"]
                )
                outbreak["fit"] = fit
                if math.isnan(outbreak["lat"]):
                    outbreak["lat"] = float(data.lat.values)
                if math.isnan(outbreak["lon"]):
                    outbreak["lon"] = float(data.lon.values)
                exponential_outbreaks.append(outbreak)
                break
    return exponential_outbreaks
