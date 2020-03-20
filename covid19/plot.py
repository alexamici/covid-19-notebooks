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
        label = f"{label} - $T_d={fit.T_d_days:.1f}$ giorni, $r^2={fit.r2:.3f}$"
    ax.plot(x_fit, y_fit, ".-", label=label, **plot_kwargs)

    ax.set(xlim=(extrapolate_start, extrapolate_stop))


def plot_data(
    ax, data, start=None, stop=None, label=None, color=None, date_interval=2, kind='scatter', **kwargs
):
    plot_kwargs = {"color": color or next(PALETTE)}
    plot_kwargs.update(kwargs)

    if kind=='scatter':
        sns.scatterplot(ax=ax, data=data[start:stop], label=label, s=80, **plot_kwargs)
    else:
        sns.lineplot(ax=ax, data=data[start:stop], label=label, **plot_kwargs)
    if start is not None:
        sns.scatterplot(data=data[data.index < start], marker="o", s=80, **plot_kwargs)
    if stop is not None:
        sns.scatterplot(data=data[data.index > stop], marker="o", s=80, facecolors='none', edgecolor=color, **plot_kwargs)

    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=date_interval))
    ax.xaxis.set_tick_params(rotation=20)


def plot(ax, data, fit, label=None, extrapolate=(None, None), color=None, **kwargs):
    color = color or next(PALETTE)
    plot_fit(ax, fit, label=label, extrapolate=extrapolate, color=color)
    plot_data(ax, data, fit.start, fit.stop, color=color, **kwargs)


ITALY_EVENTS = [
    # {'x': '2020-02-19', 'label': 'First alarm'},
    {'x': '2020-02-24', 'label': 'Chiusura scuole al nord'},
    {'x': '2020-03-01', 'label': 'Lockdown parziale al nord'},
    {'x': '2020-03-05', 'label': 'Chiusura scuole in Italia'},
    {'x': '2020-03-08', 'label': 'Lockdown al nord'},
    {'x': '2020-03-10', 'label': 'Lockdown in Italia'},
]


def add_events(ax, events=ITALY_EVENTS, offset=0, **kwargs):
    for event in events:
        label = '{x} + {offset} {label}'.format(offset=offset, **event)
        ax.axvline(x=np.datetime64(event['x']) + np.timedelta64(offset * 24 * 60 * 60, 's'), label=label, **kwargs)
