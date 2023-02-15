import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.model_selection import KFold

from lib.estimator import pdf_kernel_estimator, uniform_pdf, gaussian_pdf, epanechnikov_pdf

mpl.use('TkAgg')

# Read in the dataframe
df = pd.read_stata('WAGE1.DTA')
wage_samples: Series = df['wage'].values
N = len(wage_samples)
sigma_hat = np.std(wage_samples, ddof=1)

palette = sns.color_palette()
plot_bounds = wage_samples.min(), wage_samples.max()

# Test the uniform kernel
fig, _ = plt.subplots(figsize=(6, 3), dpi=148)
zs = np.linspace(plot_bounds[0], plot_bounds[1], 1001)
for l in np.arange(1, 6) * 0.5:
    h = l * sigma_hat * np.power(N, -1 / 5)
    plt.plot(zs, pdf_kernel_estimator(zs, wage_samples, uniform_pdf, h), label=f'{l:.1f}')
plt.gca().set(xlabel='wage', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best', title='$\lambda$')
fig.savefig('prob3-1.eps')

# Test the gaussian kernel
fig, _ = plt.subplots(figsize=(6, 3), dpi=148)
for l in np.arange(1, 6) * 0.5:
    h = l * sigma_hat * np.power(N, -1 / 5)
    plt.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h), label=f'{l:.1f}')
plt.gca().set(xlabel='wage', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best', title='$\lambda$')
fig.savefig('prob3-2.eps')

# Test the epanechnikov kernel
fig, _ = plt.subplots(figsize=(6, 3), dpi=148)
for l in np.arange(1, 6) * 0.5:
    h = l * sigma_hat * np.power(N, -1 / 5)
    plt.plot(zs, pdf_kernel_estimator(zs, wage_samples, epanechnikov_pdf, h), label=f'{l:.1f}')
plt.gca().set(xlabel='wage', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best', title='$\lambda$')
fig.savefig('prob3-3.eps')

# Test the gaussian kernel, but with different scaling
fig, _ = plt.subplots(figsize=(6, 3), dpi=148)
for ir in np.arange(2, 8):
    h = 1.06 * sigma_hat * np.power(N, -1 / ir)
    plt.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h), label=f'1/{ir:d}')
plt.gca().set(xlabel='wage', ylabel='CDF', xlim=plot_bounds)
plt.legend(loc='best', title='$r$')
fig.savefig('prob3-4.eps')

# The leave-one-out CV approach
neg_log_likelihood = {}
# hs = np.power(10, np.arange(-1.9, 0.4, 0.1))
ls = np.arange(0.3, 1.31, 0.1)
hs = ls * sigma_hat * np.power(N, -1 / 5)
for h in hs:
    z_score_mat = (wage_samples[:, None] - wage_samples[None, :]) / h
    gaussian_mat = gaussian_pdf(z_score_mat)
    density_estimate = (np.sum(gaussian_mat, axis=0) - np.diag(gaussian_mat)) / ((N - 1) * h)
    neg_log_likelihood[h] = np.mean(-np.log(np.clip(density_estimate, a_min=1e-6, a_max=None)))

# The two-fold approach
neg_log_likelihood_2fold = {}
# hs = np.power(10, np.arange(-1.9, 0.4, 0.1))
for h in hs:
    z_score_mat = (wage_samples[:, None] - wage_samples[None, :]) / h
    fold_train, fold_val = next(KFold(n_splits=2, shuffle=True).split(wage_samples))
    gaussian_mat = gaussian_pdf(z_score_mat[fold_train, :][:, fold_val])
    density_estimate = np.average(gaussian_mat, axis=0) / h
    neg_log_likelihood_2fold[h] = np.mean(-np.log(np.clip(density_estimate, a_min=1e-6, a_max=None)))

seq_nll = [neg_log_likelihood[h] for h in hs]
seq_nll_2fold = [neg_log_likelihood_2fold[h] for h in hs]
h_best = hs[np.argmin(seq_nll)]
l_best = ls[np.argmin(seq_nll)]
h_best_2fold = hs[np.argmin(seq_nll_2fold)]
l_best_2fold = ls[np.argmin(seq_nll_2fold)]
h_test_1 = hs[2]
h_test_2 = hs[-3]
fig, (ax_cv, ax_compare) = plt.subplots(2, 1, figsize=(7, 5), dpi=148, layout='constrained')
ax_cv.plot(hs, seq_nll, label='Leave one out', c=palette[0], ls='-')
ax_cv.plot(hs, seq_nll_2fold, label='2 fold', c=palette[1], ls='-')
ax_cv.set(xlabel=r'$h=\lambda \, \widehat{\sigma} \, N^{-1/5}$', ylabel='Negative log likelihood', xscale='log',
          title='Cross validation')
ax_cv.legend(loc='best')
# ax_cv.scatter([h_test_1, h_best, h_best_2fold, h_test_2],
#               [seq_nll[2], np.min(seq_nll), np.min(seq_nll_2fold), seq_nll[-3]], marker='^', c=palette[3:5])
ax_cv.scatter([h_best, h_best_2fold], [np.min(seq_nll), np.min(seq_nll_2fold)], marker='^', c=palette[3:5])

zs = np.linspace(plot_bounds[0], plot_bounds[1], 1001)
# ax_compare.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h=4e-2), label=f'{h_test_1:.1e}', alpha=0.5,
#                 c=palette[2])
ax_compare.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h=h_best), label=f'{l_best:.1f} (loo best)',
                c=palette[3])
ax_compare.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h=h_best_2fold),
                label=f'{l_best_2fold:.1f} (2 fold best)', c=palette[4])
# ax_compare.plot(zs, pdf_kernel_estimator(zs, wage_samples, gaussian_pdf, h=3e0), label=f'{h_test_2:.1e}', alpha=0.5,
# c=palette[5])
ax_compare.set(xlabel='wage', ylabel='density', title='Kernel estimate')
ax_compare.legend(loc='best', title=r'$\lambda$')
fig.savefig('prob3-5.eps')

plt.show()
