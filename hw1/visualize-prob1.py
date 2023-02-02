import numpy as np
from matplotlib import pyplot as plt

import matplotlib as mpl
import seaborn as sns

from lib.estimator import gaussian_cdf, cdf_kernel_estimator
from lib.plot import plot_empirical_distribution

mpl.use('TkAgg')

palette = sns.color_palette()
fig = plt.figure(figsize=(7, 5), dpi=148)
plot_bounds = -4, 4
zs = np.linspace(plot_bounds[0], plot_bounds[1], 101)  # we use a fixed grid for display

# Plot the theoretical CDF
plt.plot(zs, gaussian_cdf(zs), ls=':', label='Theoretical', c=palette[0])

# Draw 100 independent samples from normal distribution
N = 100
samples = np.random.randn(N)

# The empirical distribution is discontinuous
plot_empirical_distribution(samples, plot_bounds, plot_kws={'color': palette[1]})

# Try different kernel estimators
asymptotic_orders = 2, 5
for i, order in enumerate(asymptotic_orders):
    plt.plot(zs, cdf_kernel_estimator(zs, samples, gaussian_cdf, np.power(N, -1 / order)),
             label=f'Smoothed (order={order})', c=palette[2 + i])

plt.gca().set(xlabel='x', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best')
fig.savefig('prob1.eps')
plt.show()
