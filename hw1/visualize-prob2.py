import numpy as np
from matplotlib import pyplot as plt

import matplotlib as mpl
import seaborn as sns
import pandas as pd
from pandas import Series

from lib.estimator import gaussian_cdf, cdf_kernel_estimator_1d
from lib.plot import plot_empirical_distribution

mpl.use('TkAgg')

# Read in the dataframe
df = pd.read_stata('WAGE1.DTA')
wage_samples: Series = df['wage']
N = len(wage_samples)

palette = sns.color_palette()
fig = plt.figure(figsize=(7, 5), dpi=148)
plot_bounds = wage_samples.min(), wage_samples.max()
zs = np.linspace(plot_bounds[0], plot_bounds[1], 101)

# Plot empirical CDF by different number of samples
sub_sample = wage_samples.sample(10).values
plot_empirical_distribution(sub_sample, plot_bounds, plot_kws={'color': palette[0], 'ls': '-'}, label='10')
plt.plot(zs, cdf_kernel_estimator_1d(zs, sub_sample, gaussian_cdf, np.power(10, -1 / 2)), ls=':', c=palette[0],
         label='10, smoothed')

sub_sample = wage_samples.sample(100).values
plot_empirical_distribution(sub_sample, plot_bounds, plot_kws={'color': palette[1], 'ls': '-'}, label='100')
plt.plot(zs, cdf_kernel_estimator_1d(zs, sub_sample, gaussian_cdf, np.power(100, -1 / 2)), ls=':', c=palette[1],
         label='100, smoothed')

plot_empirical_distribution(wage_samples.values, plot_bounds, plot_kws={'color': palette[2], 'ls': '-'}, label='full')
plt.plot(zs, cdf_kernel_estimator_1d(zs, wage_samples.values, gaussian_cdf, np.power(N, -1 / 2)), ls=':', c=palette[2],
         label='full, smoothed')

plt.gca().set(xlabel='wage', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best', title='sample size')
fig.savefig('prob2.eps')
plt.show()
