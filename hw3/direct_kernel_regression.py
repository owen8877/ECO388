# The first approach, i.e. a direct kernel regression
from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from tqdm.auto import trange, tqdm
from tqdm.contrib import tenumerate

from common import COL, Estimator
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
        log_diff = np.log(gaussian_pdf(diff) + 1e-12)
        try:
            nm = np.nanmean(log_diff, axis=2)
        except RuntimeWarning:
            pass
        w = np.nan_to_num(np.exp(nm))

        all_nan = np.isnan(diff).all(axis=2)
        w = w * (1 - all_nan)
        weight_matrix[arr_start:arr_end, :] = w
    weight_matrix += 1e-12
    vy_pred = Series(data=(weight_matrix.T @ ty) / np.sum(weight_matrix, axis=0), index=vX.index)
    return vy_pred


def direct_kernel_regression_estimate(
        train_df: DataFrame, test_X: DataFrame, fill_in_choice: float = np.nan) -> Tuple[DataFrame, dict]:
    # Validation that builds the regression hyperparameters
    cv_loss = dict()
    h_trial_space = np.linspace(10, 30, 10)
    for h in tqdm(h_trial_space):
        cv_loss[h] = []
        for i, (train_index, val_index) in tenumerate(KFold(n_splits=3).split(train_df), disable=True):
            tdf, vdf = train_df.iloc[train_index], train_df.iloc[val_index]
            tX, vX = tdf[COL.WAPs], vdf[COL.WAPs]
            error = dict()
            for field in COL.LON, COL.LAT, COL.FLR, COL.BID:
                ty, vy = tdf[field], vdf[field]
                vy_pred = _regression_helper(tX, vX, ty, fill_in_choice, h)
                if field in {COL.FLR, COL.BID}:
                    vy_pred = np.round(vy_pred).astype(int)
                error[field] = (vy_pred - vy) ** 2
                if field in {COL.FLR, COL.BID}:
                    error[field] = error[field] > 0

            cv_loss[h].append(np.mean(
                np.sqrt(error[COL.LON] + error[COL.LAT]) + 4 * np.sqrt(error[COL.FLR]) + 50 * np.sqrt(error[COL.BID])))

    cv_loss = Series(data={h: np.mean(cv_loss[h]) for h in cv_loss}, name='cv loss')
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
    return DataFrame(data=prediction, index=test_X.index), {'cv_loss': cv_loss}


class DirectKernelRegression(Estimator):
    def __init__(self, train_df: DataFrame, subsample: float = 0.03):
        super().__init__(train_df, 'dkr')
        self.train_building_classifier()
        self.subsample = subsample

    def get_wap_locations(self) -> DataFrame:
        sub_df = self.train_df.sample(frac=self.subsample)
        weight = np.power(sub_df[COL.WAPs].replace(100, np.nan) / 104.0 + 1.0, 0.3).fillna(0).values + 1e-12
        user_coor = sub_df[COL.COOR3].values
        df = DataFrame(data=(weight.T @ user_coor) / weight.sum(axis=0)[:, None], columns=COL.COOR3)
        df[COL.BID] = self.clf.predict(self.scaler.transform(df[COL.COOR2].values))
        return df

    def estimate(self, test_X: DataFrame, fill_in_choice: float = np.nan) -> Tuple[DataFrame, dict]:
        return direct_kernel_regression_estimate(self.train_df.sample(frac=self.subsample), test_X,
                                                 fill_in_choice=fill_in_choice)

    def analyze_and_dump(self, test_y, pred_y):
        super().analyze_and_dump(test_y, pred_y)
