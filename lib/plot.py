from typing import Tuple, Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_empirical_distribution(samples, plot_bounds: Iterable[float], *, ax: Axes = None, plot_kws=None,
                                label: str = None):
    if ax is None:
        ax = plt.gca()
    if plot_kws is None:
        plot_kws = {}

    lb, ub = plot_bounds
    N = len(samples)

    sorted_samples = np.sort(samples)
    ax.plot([lb, sorted_samples[0]], [0, 0], label=label, **plot_kws)
    for i in range(N - 1):
        level = (i + 1) / N
        ax.plot(sorted_samples[i:i + 2], [level, level], **plot_kws)
    ax.plot([sorted_samples[-1], ub], [1, 1], **plot_kws)
