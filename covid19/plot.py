import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


DAY = np.timedelta64(24 * 3600, "s")
PALETTE = itertools.cycle(sns.color_palette())
LOG2 = math.log(2)


def plot_fit(ax, fit, label=None, extrapolate=(-2, +2), color=None, **kwargs):
    extrapolate_start, extrapolate_stop = extrapolate

    if isinstance(extrapolate_start, int):
        extrapolate_start = fit.start + extrapolate_start * DAY
    elif isinstance(extrapolate_start, str):
        extrapolate_start = np.datetime64(extrapolate_start)

    if isinstance(extrapolate_stop, int):
        extrapolate_stop = fit.stop + extrapolate_stop * DAY
    elif isinstance(extrapolate_stop, str):
        extrapolate_stop = np.datetime64(extrapolate_stop)

    x_predict = pd.date_range(extrapolate_start, extrapolate_stop, freq="D")
    y_predict = fit.predict(x_predict)

    plot_kwargs = {"color": color or next(PALETTE), **kwargs}

    ax.plot(x_predict, y_predict, ":", **plot_kwargs)

    x_fit = pd.date_range(fit.start, fit.stop, freq="D").values
    y_fit = fit.predict(x_fit)
    if label:
        # label = f"{label} - $T_d={fit.T_d_days:.1f}$ giorni, $r^2={fit.r2:.3f}$"
        label = f"$T_d={fit.T_d_days:.1f}$ days - {label}"
    ax.plot(x_fit, y_fit, ".-", label=label, **plot_kwargs)


def plot_data(
    ax,
    data,
    start=None,
    stop=None,
    label=None,
    color=None,
    date_interval=7,
    marker="o",
    markersize=3,
    delay=None,
    ratio=None,
    show_left=False,
    show_right=False,
    drop_negative=True,
    x="time",
    **kwargs,
):
    plot_kwargs = {
        "color": color or next(PALETTE),
        "markersize": markersize,
        "marker": marker,
    }
    plot_kwargs.update(kwargs)

    data_to_plot = data
    if delay is not None:
        if isinstance(delay, (int, float)):
            delay = delay * np.timedelta64(24 * 3600, "s")
        data_to_plot = data_to_plot.assign_coords(
            {x: (x, data_to_plot.coords[x] + delay)}
        )
    if ratio is not None:
        data_to_plot = data_to_plot / ratio
    if drop_negative:
        data_to_plot = data_to_plot[data_to_plot > 0]

    # if kind == "scatter":
    data_to_plot.sel(**{x: slice(start, stop)}).plot(ax=ax, label=label, **plot_kwargs)
    # else:
    #    sns.lineplot(ax=ax, data=data_to_plot, label=label, **plot_kwargs)
    if show_left and start is not None:
        sns.scatterplot(
            data=data_to_plot[data_to_plot.index < start],
            marker="o",
            s=60,
            **plot_kwargs,
        )
    if show_right and stop is not None:
        sns.scatterplot(
            data=data_to_plot[data_to_plot.index > stop],
            marker="o",
            s=80,
            facecolors="none",
            edgecolor=color,
            **plot_kwargs,
        )

    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=date_interval))
    ax.xaxis.set_tick_params()


def plot(
    ax,
    data,
    fit=None,
    label=None,
    extrapolate=(-2, 2),
    color=None,
    add_diff=False,
    **kwargs,
):
    color = color or next(PALETTE)
    if fit is not None:
        plot_fit(ax, fit, label=label, extrapolate=extrapolate, color=color)
    plot_data(ax, data, fit.start, fit.stop, color=color, **kwargs)
    if add_diff:
        diff = data[fit.start : fit.stop].diff(1) * fit.T_d_days / LOG2
        plot_data(ax, diff, color=color, alpha=0.4)


ITALY_EVENTS = [
    # {'x': '2020-02-19', 'label': 'First alarm'},
    {"x": "2020-02-24", "label": "Chiusura scuole al nord"},
    {"x": "2020-03-01", "label": "Lockdown parziale al nord"},
    {"x": "2020-03-05", "label": "Chiusura scuole in Italia"},
    {"x": "2020-03-08", "label": "Lockdown al nord"},
    {"x": "2020-03-10", "label": "Lockdown parziale in Italia"},
    {"x": "2020-03-12", "label": "Lockdown in Italia"},
]


def add_events(ax, events=ITALY_EVENTS, offset=0, **kwargs):
    for event in events:
        label = "{x} + {offset} {label}".format(offset=offset, **event)
        ax.axvline(
            x=np.datetime64(event["x"]) + np.timedelta64(offset * 24 * 60 * 60, "s"),
            label=label,
            **kwargs,
        )


def subplots(*args, **kwargs):
    f, ax = plt.subplots(*args, **kwargs)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.grid(color="lightgrey", linewidth=0.5)

    ax.xaxis.grid(color="lightgrey", linewidth=0.5)

    return f, ax


def plot_xarray(
    data,
    foreground_hue=None,
    x="time",
    ax=None,
    hue="age_class",
    window=1,
    foreground_interval=(None, None),
    ylim=(0, None),
    alpha=0.8,
    **kwargs,
):
    if ax is None:
        _, ax = subplots()

    data_stop = data[x].values.max()

    if window != 1:
        data = data.rolling({x: window}, center=True).mean()

    if isinstance(foreground_hue, str):
        foreground_hue = [foreground_hue]

    foreground_data = data.sel({x: slice(*foreground_interval)})

    for h, color in zip(
        data[hue].values, itertools.cycle(sns.color_palette())
    ):
        label = None if foreground_hue is None or h in foreground_hue else h
        ax.plot(
            data[x],
            data.sel({hue: h}),
            linewidth=2.5,
            alpha=alpha / 4,
            color=color,
            label=label,
        )
        if foreground_hue is None or h in foreground_hue:
            ax.plot(
                foreground_data[x],
                foreground_data.sel({hue: h}),
                linewidth=2.5,
                alpha=alpha,
                color=color,
                label=h,
            )

    if foreground_interval[1] is not None:
        ax.set(ylim=ylim)
        ylim = ax.get_ylim()
        ax.fill(
            [np.datetime64(foreground_interval[1])] * 2 + [data_stop] * 2,
            [ylim[0], ylim[1], ylim[1], ylim[0]],
            "grey",
            alpha=0.1,
        )
        ax.set(ylim=ylim)

    ax.set(**kwargs)

    return ax


def scatter_xarray(x, y, hue="location", time="time", ax=None, window=1, **kwargs):
    if ax is None:
        _, ax = subplots()

    if window != 1:
        x = x.rolling({time: window}, center=True).mean().dropna(time)
        y = y.rolling({time: window}, center=True).mean().dropna(time)

    x, y = xr.align(x, y)

    for h, color in zip(x[hue].values, itertools.cycle(sns.color_palette())):
        xx = x.sel(**{hue: h}).values
        yy = y.sel(**{hue: h}).values
        ax.plot(xx, yy, "-", color=color, alpha=0.3, linewidth=2)
        ax.plot(xx[-1:], yy[-1:], "o", color=color, label=h, **kwargs)
        ax.annotate(h, (xx[-1:] * 1.02, yy[-1:] * 0.85), color=color)

    return ax
