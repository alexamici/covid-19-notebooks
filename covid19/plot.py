import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot(data, fit, label=None, extrapolate=(None, None)):
    extrapolate_start = (
        data.index[0] if extrapolate[0] is None else np.datetime64(extrapolate[0])
    )
    extrapolate_stop = (
        data.index[-1] if extrapolate[1] is None else np.datetime64(extrapolate[1])
    )

    x_predict = pd.date_range(extrapolate_start, extrapolate_stop)
    y_predict = fit.predict(x_predict)
    label_fit = f"estimated {label} $T_d={fit.T_d_days:.2f}$ days, $t_0=${str(fit.t_0)[:10]}"
    plt.plot(x_predict, y_predict, ":", label=label_fit)

    ax = sns.scatterplot(data=data, label=label)

    ax.set(xlim=(extrapolate_start, extrapolate_stop))

    return ax
