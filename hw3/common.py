from datetime import datetime
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.svm import LinearSVC

from lib import ensure_dir


class COL:
    WAPs = [f'WAP{i + 1:03d}' for i in range(520)]
    LON = 'LONGITUDE'
    LAT = 'LATITUDE'
    FLR = 'FLOOR'
    BID = 'BUILDINGID'
    COOR2 = [LON, LAT]
    COOR3 = [LON, LAT, FLR]


class MyStandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    My standard scaler where only one variance is fit.
    """

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.sqrt(np.sum(np.var(X, axis=0)))
        return self

    def transform(self, X):
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X *= self.scale_
        X += self.mean_
        return X


class Estimator:
    def __init__(self, train_df: DataFrame, name: str):
        self.name = name
        self.train_df = train_df

        self.scaler = MyStandardScaler()  # used to normalize GPS coordinates
        X = self.train_df[COL.COOR2].values
        self.transformed_coordinates = DataFrame(data=self.scaler.fit_transform(X), columns=COL.COOR2,
                                                 index=train_df.index)
        self.transformed_coordinates[COL.FLR] = train_df[COL.FLR]

        self.clf = LinearSVC()  # used to infer building id

    def train_building_classifier(self):
        y = self.train_df[COL.BID].values
        self.clf.fit(self.transformed_coordinates[COL.COOR2].values, y)
        print(f'Building classifier score: {self.clf.score(self.transformed_coordinates[COL.COOR2].values, y)}')

    def train(self, **kwargs):
        raise NotImplemented

    def interpret(self):
        raise NotImplemented

    def get_wap_locations(self) -> DataFrame:
        raise NotImplemented

    def estimate(self, test_X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        raise NotImplemented

    def analyze_and_dump(self, test_y, pred_y):
        loss_score, fig_prediction, fig_distance_error, fig_building_mismatch, fig_floor_mismatch = estimate_loss(
            test_y, pred_y)
        dt = datetime.now()
        path = f'dump/{self.name}_{dt.strftime("%Y%m%d_%H%M%S")}'
        ensure_dir(path)

        np.save(f'{path}/score_{loss_score:.3f}.npy', [])
        fig_prediction.savefig(f'{path}/fig1.pdf')
        fig_distance_error.savefig(f'{path}/fig2.pdf')
        fig_building_mismatch.savefig(f'{path}/fig3.pdf')
        fig_floor_mismatch.savefig(f'{path}/fig4.pdf')


def estimate_loss(ground_truth: DataFrame, prediction: DataFrame) -> Tuple[float, Figure, Figure, Figure, Figure]:
    coor_error = np.sqrt(np.sum((ground_truth[COL.COOR2] - prediction[COL.COOR2]) ** 2, axis=1))
    floor_error = (ground_truth[COL.FLR] != prediction[COL.FLR]).astype(int) * 4
    building_error = (ground_truth[COL.BID] != prediction[COL.BID]).astype(int) * 50
    loss_score = np.mean(coor_error) + np.mean(floor_error) + np.mean(building_error)

    building_colors = sns.hls_palette(3)
    floor_markers = ['o', '^', 'v', '*', 'd']

    fig_prediction = plt.figure(1)
    for i, color in enumerate(building_colors):
        for j, marker in enumerate(floor_markers):
            mask = (prediction[COL.FLR] == j) & (prediction[COL.BID] == i)
            sub_df = prediction.loc[mask]
            plt.scatter(sub_df[COL.LON], sub_df[COL.LAT], color=color, marker=marker)

    fig_distance_error = plt.figure(2)
    distance_error = np.sqrt(((ground_truth[[COL.LON, COL.LAT]] - prediction[[COL.LON, COL.LAT]]) ** 2).sum(axis=1))
    max_error = distance_error.max()
    p = plt.scatter(ground_truth[COL.LON], ground_truth[COL.LAT], c=distance_error / max_error,
                    alpha=distance_error / max_error, cmap='BuGn')
    plt.colorbar(p)

    fig_building_mismatch = plt.figure(3)
    for i, color in enumerate(building_colors):
        for j, marker in enumerate(floor_markers):
            selection = (ground_truth[COL.FLR] == j) & (ground_truth[COL.BID] == i)
            mask = prediction[selection][COL.BID] != i
            mismatch = mask.index[np.where(mask)[0]]
            match = mask.index[np.where(~mask)[0]]
            plt.scatter(ground_truth.iloc[mismatch][COL.LON], ground_truth.iloc[mismatch][COL.LAT], color=color,
                        marker=marker)
            plt.scatter(ground_truth.iloc[match][COL.LON], ground_truth.iloc[match][COL.LAT], color='k', alpha=0.01)

    fig_floor_mismatch = plt.figure(4)
    for i, color in enumerate(building_colors):
        for j, marker in enumerate(floor_markers):
            selection = (ground_truth[COL.FLR] == j) & (ground_truth[COL.BID] == i)
            mask = prediction[selection][COL.FLR] != j
            mismatch = mask.index[np.where(mask)[0]]
            match = mask.index[np.where(~mask)[0]]
            plt.scatter(ground_truth.iloc[mismatch][COL.LON], ground_truth.iloc[mismatch][COL.LAT], color=color,
                        marker=marker)
            plt.scatter(ground_truth.iloc[match][COL.LON], ground_truth.iloc[match][COL.LAT], color='k', alpha=0.01)

    return loss_score, fig_prediction, fig_distance_error, fig_building_mismatch, fig_floor_mismatch
