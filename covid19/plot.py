import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = itertools.cycle(sns.color_palette())


def plot(data, fit, label=None, extrapolate=(None, None), color=None):
    extrapolate_start = (
        data.index[0] if extrapolate[0] is None else np.datetime64(extrapolate[0])
    )
    extrapolate_stop = (
        data.index[-1] if extrapolate[1] is None else np.datetime64(extrapolate[1])
    )
    kwargs = {"color": color or next(PALETTE)}

    x_predict = pd.date_range(extrapolate_start, extrapolate_stop, periods=2)
    y_predict = fit.predict(x_predict)
    plt.plot(x_predict, y_predict, ":", **kwargs)

    x_fit = data[fit.start : fit.stop].index.values
    y_fit = fit.predict(x_fit)
    label_fit = (
        f"estimated {label} $T_d={fit.T_d_days:.2f}$ days, $t_0=${str(fit.t_0)[:10]}"
    )
    plt.plot(x_fit, y_fit, ".-", label=label_fit, **kwargs)

    kwargs["s"] = 80

    ax = sns.scatterplot(data=data[fit.start : fit.stop], label=label, **kwargs)
    if fit.start is not None:
        sns.scatterplot(data=data[data.index < fit.start], marker="x", **kwargs)
    if fit.stop is not None:
        sns.scatterplot(data=data[data.index > fit.stop], marker="x", **kwargs)

    ax.set(xlim=(extrapolate_start, extrapolate_stop))
    ax.set(yscale="log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_tick_params(rotation=20)

    return ax
