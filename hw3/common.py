from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame


class COL:
    WAPs = [f'WAP{i + 1:03d}' for i in range(520)]
    LON = 'LONGITUDE'
    LAT = 'LATITUDE'
    FLR = 'FLOOR'
    BID = 'BUILDINGID'


def estimate_loss(ground_truth: DataFrame, prediction: DataFrame) -> Tuple[float, Figure, Figure, Figure, Figure]:
    coor_true_np, coor_pred_np = ground_truth.values, prediction.values
    coor_error = np.sqrt(np.sum((coor_true_np[:, :2] - coor_pred_np[:, :2]) ** 2, axis=1))
    floor_error = (coor_true_np[:, 2] != coor_pred_np[:, 2]).astype(int) * 4
    building_error = (coor_true_np[:, 3] != coor_pred_np[:, 3]).astype(int) * 50
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
