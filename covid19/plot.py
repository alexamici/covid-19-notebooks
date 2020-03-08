import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = itertools.cycle(sns.color_palette())


def plot_fit(ax, fit, label=None, extrapolate=(None, None), color=None):
    plot_kwargs = {"color": color or next(PALETTE)}
    extrapolate_start = (
        fit.start if extrapolate[0] is None else np.datetime64(extrapolate[0])
    )
    extrapolate_stop = (
        fit.stop if extrapolate[1] is None else np.datetime64(extrapolate[1])
    )

    x_predict = pd.date_range(extrapolate_start, extrapolate_stop, freq="D")
    y_predict = fit.predict(x_predict)
    ax.plot(x_predict, y_predict, ":", **plot_kwargs)

    x_fit = pd.date_range(fit.start, fit.stop, freq="D").values
    y_fit = fit.predict(x_fit)
    if label:
        label = f"estimated {label} $T_d={fit.T_d_days:.2f}$ days, $t_0=${str(fit.t_0)[:10]}"
    ax.plot(x_fit, y_fit, ".-", label=label, **plot_kwargs)

    ax.set(xlim=(extrapolate_start, extrapolate_stop))


def plot_data(ax, data, start=None, stop=None, label=None, color=None, date_interval=2):
    plot_kwargs = {"color": color or next(PALETTE), "s": 80}

    sns.scatterplot(ax=ax, data=data[start:stop], label=label, **plot_kwargs)
    if start is not None:
        sns.scatterplot(data=data[data.index < start], marker="x", **plot_kwargs)
    if stop is not None:
        sns.scatterplot(data=data[data.index > stop], marker="x", **plot_kwargs)

    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=date_interval))
    ax.xaxis.set_tick_params(rotation=20)


def plot(ax, data, fit, label=None, extrapolate=(None, None), color=None, **kwargs):
    color = color or next(PALETTE)
    plot_fit(ax, fit, label=label, extrapolate=extrapolate, color=color)
    plot_data(ax, data, fit.start, fit.stop, label=label, color=color, **kwargs)
