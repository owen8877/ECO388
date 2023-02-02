import numpy as np
from scipy.special import erf


def gaussian_cdf(z):
    return (1 + erf(z / np.sqrt(2))) / 2


def gaussian_pdf(z):
    return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)


def uniform_pdf(z):
    return np.abs(z) < 0.5


def epanechnikov_pdf(z):
    return 0.75 * (1 - z ** 2) * (np.abs(z) <= 1)


def cdf_kernel_estimator(grids, samples, method, h):
    return kernel_estimator(grids, samples, method, h, True)


def pdf_kernel_estimator(grids, samples, method, h):
    return kernel_estimator(grids, samples, method, h, False)


def kernel_estimator(grids, samples, method, h, is_cdf):
    z_score = (grids[None, :] - samples[:, None]) / h
    return np.mean(method(z_score), axis=0) / (1 if is_cdf else h)
