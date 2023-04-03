# The first approach, i.e. a direct kernel regression
from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from tqdm.auto import trange, tqdm
from tqdm.contrib import tenumerate

from common import COL
from lib.estimator import gaussian_pdf


def _regression_helper(tX: DataFrame, vX: DataFrame, ty: Series, fill_in_choice: float, h: float) -> Series:
    tX_fill = tX.replace(100, fill_in_choice)
    vX_fill = vX.replace(100, fill_in_choice)

    weight_matrix = np.empty((len(tX), len(vX)))
    CHUNK_SIZE = 100
    for i in trange(int(np.ceil(len(tX) / CHUNK_SIZE)), position=1, disable=True):
        arr_start = i * CHUNK_SIZE
        arr_end = min((i + 1) * CHUNK_SIZE, len(tX))
        diff = (tX_fill.values[arr_start:arr_end, None, :] - vX_fill.values[None, :, :]) / h
        weight_matrix[arr_start:arr_end, :] = np.sum(np.nan_to_num(gaussian_pdf(diff), copy=False), axis=2)
    weight_matrix += 1e-12
    vy_pred = Series(data=(weight_matrix.T @ ty) / np.sum(weight_matrix, axis=0), index=vX.index)
    return vy_pred


def direct_kernel_regression_estimate(
        train_df: DataFrame, test_X: DataFrame, fill_in_choice: float = np.nan) -> Tuple[DataFrame, Series]:
    # Validation that builds the regression hyperparameters
    cv_loss = dict()
    h_trial_space = np.power(10, np.linspace(-0.2, 0.8, 10))
    for h in tqdm(h_trial_space):
        for i, (train_index, val_index) in tenumerate(KFold(n_splits=5).split(train_df), disable=True):
            tdf, vdf = train_df.iloc[train_index], train_df.iloc[val_index]
            tX, vX = tdf[COL.WAPs], vdf[COL.WAPs]
            error = dict()
            for field in COL.LON, COL.LAT, COL.FLR, COL.BID:
                ty, vy = tdf[field], vdf[field]
                vy_pred = _regression_helper(tX, vX, ty, fill_in_choice, h)
                if field in {COL.FLR, COL.BID}:
                    vy_pred = np.round(vy_pred).astype(int)
                error[field] = np.sqrt(((vy_pred - vy) ** 2).mean())

            cv_loss[h] = np.sqrt(error[COL.LON] ** 2 + error[COL.LAT] ** 2) + 4 * error[COL.FLR] + 50 * error[COL.BID]
    cv_loss = Series(data=cv_loss, name='cv loss')
    cv_loss.index.name = 'h'

    h_best: float = cv_loss.idxmin()
    print(f'Best h is {h_best:.3f} and minimal cv loss is {cv_loss.min():.3f}.')

    # Predict the test data!
    tX = train_df[COL.WAPs]
    prediction = dict()
    for field in COL.LON, COL.LAT, COL.FLR, COL.BID:
        ty = train_df[field]
        y_pred = _regression_helper(tX, test_X, ty, fill_in_choice, h_best)
        if field in {COL.FLR, COL.BID}:
            y_pred = np.round(y_pred).astype(int)
        prediction[field] = y_pred
    return DataFrame(data=prediction, index=test_X.index), cv_loss
