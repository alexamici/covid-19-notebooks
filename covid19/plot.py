import itertools
import math

import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


DAY = np.timedelta64(24 * 3600, "s")
PALETTE = itertools.cycle(sns.color_palette())
LOG2 = math.log(2)


def myLogFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = "{{:.{:1d}f}}".format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def plot_fit(ax, fit, label=None, extrapolate=(-2, +2), delay=None, ratio=None, color=None, **kwargs):
    extrapolate_start, extrapolate_stop = extrapolate

    if delay is not None:
        fit = fit.shift(delay)
    if ratio is not None:
        fit = fit.scale(1 / ratio)

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
        label = f"$T_d={fit.T_d_days:.1f}$ giorni - {label}"
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
    markersize=2,
    delay=None,
    ratio=None,
    show_left=False,
    show_right=False,
    drop_negative=True,
    x="time",
    linestyle="-",
    annotate=False,
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
        data_to_plot = data_to_plot[data_to_plot >= 0]

    # if kind == "scatter":
    data_to_plot_interval = data_to_plot.sel(**{x: slice(start, stop)})
    data_to_plot_interval.plot(ax=ax, label=label, linestyle=linestyle, **plot_kwargs)
    if annotate:
        x = data_to_plot_interval[-1][x].values + np.timedelta64(4, 'D')
        value = data_to_plot_interval[-1].values
        label = f'{value:#.2g}' if value <= 10 else f'{value:.0f}'
        y = value * .95
        ax.annotate(label, (x, y), color=color, path_effects=[
                patheffects.Stroke(linewidth=4, foreground='white'),
                patheffects.Normal(),
        ])
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


def subplots(*args, tick_right=True, **kwargs):
    f, ax = plt.subplots(*args, **kwargs)

    if tick_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.yaxis.grid(color="lightgrey", linewidth=0.5)
    ax.xaxis.grid(color="lightgrey", linewidth=0.5)
    ax.xaxis.set_tick_params(labelsize=14)

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
    date_interval=30,
    linewidth=2.5,
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

    for h, color in zip(data[hue].values, itertools.cycle(sns.color_palette())):
        label = None if foreground_hue is None or h in foreground_hue else h
        ax.plot(
            data[x],
            data.sel({hue: h}),
            linewidth=linewidth,
            alpha=alpha / 4,
            color=color,
            label=label,
        )
        if foreground_hue is None or h in foreground_hue:
            ax.plot(
                foreground_data[x],
                foreground_data.sel({hue: h}),
                linewidth=linewidth,
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

    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=date_interval))
    ax.set(**kwargs)

    return ax


def scatter_xarray(
    x, y, hue="location", time="time", ax=None, window=1, xlim=None, ylim=None, **kwargs
):
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
        xp = xx[-1:] * 1.05
        yp = yy[-1:]
        if (xlim is None or xlim[0] < xp < xlim[1]) and (
            ylim is None or ylim[0] < yp < ylim[1]
        ):
            ax.annotate(h, (xp, yp), path_effects=[
                patheffects.Stroke(linewidth=4, foreground='white'),
                patheffects.Normal(),
            ])

    return ax


def animate_scatter(x, y, *, time="time", freq='6h', tail=28, **kwargs):
    time_interp = pd.date_range(x[time].values[0], x[time].values[-1], freq=freq)
    x_interp = x.interp({time: time_interp})
    y_interp = y.interp({time: time_interp})

    fig, ax = subplots()

    def animate_ax(i):
        ax.clear()
        ax.set(**kwargs)

        ax.yaxis.grid(color="lightgrey", linewidth=0.5)
        ax.xaxis.grid(color="lightgrey", linewidth=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(myLogFormat))
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(myLogFormat))

        ax.set_title(str(time_interp[i])[:10])

        x_plot = x_interp.isel({time: slice(max(0, i - tail + 1), i + 1)})
        y_plot = y_interp.isel({time: slice(max(0, i - tail + 1), i + 1)})
        scatter_xarray(x_plot, y_plot, ax=ax, xlim=kwargs.get('xlim', None), ylim=kwargs.get('ylim', None))

    return animation.FuncAnimation(fig, animate_ax, frames=time_interp.size, repeat=True)
