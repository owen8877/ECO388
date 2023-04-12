from datetime import datetime
from itertools import chain
from typing import Tuple
from unittest import TestCase

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import RandomSampler, BatchSampler
from tqdm.auto import trange, tqdm

from hw3.common import COL, Estimator
from hw3.direct_kernel_regression import DirectKernelRegression
from lib import ensure_dir


def info_tensor(user_coor, wap_coor):
    diff = (user_coor[:, :2] - wap_coor[:, :2])
    horizontal_distance2 = (diff ** 2).sum(dim=1)
    vertical_distance = torch.abs(user_coor[:, 2] - wap_coor[:, 2])
    vertical_count = torch.abs(torch.floor(user_coor[:, 2]) - torch.floor(wap_coor[:, 2]))
    return torch.stack((horizontal_distance2, vertical_distance, vertical_count), dim=1).float()


def expand_coor(coor, index):
    return torch.index_select(coor, 0, index)


def scale_RSSI(raw_RSSI):
    return 1.0 + raw_RSSI / 104.0


def inverse_scaled_RSSI(scaled_RSSI):
    return 104.0 * (scaled_RSSI - 1.0)


def prepare_train_data(WAPs: DataFrame, coor: DataFrame):
    user_coor = torch.from_numpy(coor.values)
    query_row_np, query_column_np = np.where(WAPs != 100)
    user_row = torch.from_numpy(query_row_np)
    wap_column = torch.from_numpy(query_column_np)
    raw_RSSI = torch.tensor([WAPs.iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
    scaled_RSSI = scale_RSSI(raw_RSSI)[:, None]
    weight = scaled_RSSI ** 2 + 0.01

    return user_coor, user_row, wap_column, scaled_RSSI, weight / weight.mean()


def prepare_test_data(df: Series, nan_penalty: float):
    if np.isclose(nan_penalty, 0):
        query_row_np, query_column_np = np.where(df[COL.WAPs] != 100)
        user_row = torch.from_numpy(query_row_np)
        wap_column = torch.from_numpy(query_column_np)
        raw_RSSI = torch.tensor([df.iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
        scaled_RSSI = scale_RSSI(raw_RSSI)[:, None]
    else:
        query_row_np, query_column_np = np.where(df[COL.WAPs] < 101)
        user_row = torch.from_numpy(query_row_np)
        wap_column = torch.from_numpy(query_column_np)
        raw_RSSI = torch.tensor(
            [df.replace(100, -104).iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
        scaled_RSSI = scale_RSSI(raw_RSSI)[:, None]

    weight = scaled_RSSI ** 2 + 0.01
    return user_row, wap_column, scaled_RSSI, weight / weight.mean()


def wap_init(df: DataFrame) -> Tensor:
    de = DirectKernelRegression(df, subsample=0.05)
    estimated_location = de.get_wap_locations().values[:, :3]
    estimated_location[:, :2] = np.nan_to_num(de.scaler.transform(estimated_location[:, :2]))
    estimated_location[:, 2] = np.nan_to_num(estimated_location[:, 2], nan=2.5)
    return torch.from_numpy(estimated_location)


class GMMApproach(Estimator):
    def __init__(self, train_df: DataFrame, save_path: str = None, batch_size: int = 100):
        super().__init__(train_df, 'gmma_tree')
        self.tree = RandomForestRegressor(n_estimators=100)
        if save_path is None:
            self.train_building_classifier()
            self.wap_coor = wap_init(self.train_df).requires_grad_(True)
        else:
            self.wap_coor = torch.load(f'{save_path}/wap_coor.pt').requires_grad_(True)
            self.net.load_state_dict(torch.load(f'{save_path}/net.pt'))
            self.clf = joblib.load(f'{save_path}/clf.joblib')
            # self.scaler = joblib.load(f'{save_path}/scaler.joblib')
        parameters = chain([self.wap_coor], self.net.parameters())
        self.optimizer = Adam(parameters, lr=1e-2)

        all_indices = np.arange(len(self.train_df))
        sampler = BatchSampler(RandomSampler(all_indices), batch_size=batch_size, drop_last=False)
        self.real_data = [prepare_train_data(self.train_df.iloc[sample, :][COL.WAPs],
                                             self.transformed_coordinates.iloc[sample, :][COL.COOR3])
                          for sample in tqdm(sampler, desc='Preparing data...')]

    def save(self):
        dt = datetime.now()
        path = f'param/{self.name}_{dt.strftime("%Y%m%d_%H%M%S")}'
        ensure_dir(path)
        torch.save(self.net.state_dict(), f'{path}/net.pt')
        torch.save(self.wap_coor, f'{path}/wap_coor.pt')
        joblib.dump(self.clf, f'{path}/clf.joblib')
        joblib.dump(self.scaler, f'{path}/scaler.joblib')

    def train(self, n_epoch: int = 2, n_mini_inner: int = 1):
        epoch_pbar = trange(n_epoch, position=0)
        train_val_loss = []
        for epoch in epoch_pbar:
            train_losses = []
            train_indices, val_indices = train_test_split(np.arange(len(self.real_data)))

            sample_pbar = tqdm(RandomSampler(train_indices), position=1, leave=False)
            for package in sample_pbar:
                user_coor, user_row, wap_column, scaled_RSSI, train_weight = self.real_data[package]
                user_coor_expanded = expand_coor(user_coor, user_row)

                for _ in range(n_mini_inner):
                    self.optimizer.zero_grad()
                    wap_coor_expanded = expand_coor(self.wap_coor, wap_column)
                    predicted_RSSI = self.net(info_tensor(user_coor_expanded, wap_coor_expanded))
                    loss = torch.mean(torch.abs(predicted_RSSI - scaled_RSSI) * train_weight)
                    loss.backward()
                    self.optimizer.step()

                train_losses.append(loss.item())
                sample_pbar.set_description(f'Loss: {np.mean(train_losses):.3e}')

            val_losses = []
            for package in tqdm(val_indices, position=1, leave=False):
                user_coor, user_row, wap_column, scaled_RSSI, train_weight = self.real_data[package]
                user_coor_expanded = expand_coor(user_coor, user_row)
                with torch.no_grad():
                    wap_coor_expanded = expand_coor(self.wap_coor, wap_column)
                    predicted_RSSI = self.net(info_tensor(user_coor_expanded, wap_coor_expanded))
                    loss = torch.mean(torch.abs(predicted_RSSI - scaled_RSSI) * train_weight)
                val_losses.append(loss.item())
            epoch_pbar.set_description(f'Train loss: {np.mean(train_losses):.3e}, val loss: {np.mean(val_losses):.3e}')
            train_val_loss.append((train_losses, val_losses))
        return train_val_loss

    def get_wap_locations(self) -> DataFrame:
        wap_coor = self.wap_coor.detach()
        df = DataFrame(data=wap_coor, columns=COL.COOR3)
        df[COL.BID] = self.clf.predict(df[[COL.LON, COL.LAT]].values)
        df[COL.COOR2] = self.scaler.inverse_transform(df[COL.COOR2])
        return df

    def interpret(self):
        sample = np.random.randint(0, len(self.train_df), int(0.05 * len(self.train_df)))
        user_coor, user_row, wap_column, scaled_RSSI, _ = prepare_train_data(
            self.train_df[COL.WAPs].iloc[sample, :], self.transformed_coordinates[COL.COOR3].iloc[sample, :])
        RSSI = inverse_scaled_RSSI(scaled_RSSI)
        user_coor_expanded = expand_coor(user_coor, user_row)
        wap_coor_expanded = expand_coor(self.wap_coor, wap_column)
        info = info_tensor(user_coor_expanded, wap_coor_expanded)
        pred_raw = self.net(info)
        predicted_RSSI = inverse_scaled_RSSI(pred_raw)
        # distance_grid = torch.linspace(0, info.max().item(), 100)
        # signal_loss_pred = self.polynomial(distance_grid)
        distance = info * self.scaler.scale_

        scaled_RSSI, pred_raw, predicted_RSSI, RSSI, info, distance = [
            x.detach().numpy().squeeze() for x in
            (scaled_RSSI, pred_raw, predicted_RSSI, RSSI, info, distance)]
        df = DataFrame(
            {'scaled_RSSI': scaled_RSSI, 'pred_raw': pred_raw, 'predicted_RSSI': predicted_RSSI, 'RSSI': RSSI})

        plt.figure(1)
        plt.scatter(pred_raw, scaled_RSSI)
        # g = sns.jointplot(data=df, x='info', y='scaled_RSSI', kind='scatter', s=3)
        # sns.kdeplot(data=df, x='info', y='scaled_RSSI', ax=g.ax_joint)
        # g.ax_joint.plot(distance_grid, signal_loss_pred, '--')
        # g.ax_joint.set(ylim=[g.ax_joint.get_ylim()[0], 5])

        # plt.figure(2)
        # g = sns.jointplot(data=df, x='distance', y='RSSI', kind='kde')

    def estimate(
            self, test_X: DataFrame, n_test_itr: int = 100, batch_size: int = 500,
            nan_penalty: float = 0.0,
    ) -> Tuple[DataFrame, dict]:
        coords = np.zeros((len(test_X), 3))
        wap_coor = self.wap_coor.detach()

        all_indices = np.arange(len(test_X))
        sampler = BatchSampler(RandomSampler(all_indices), batch_size=batch_size, drop_last=False)

        out_pbar = tqdm(sampler, position=0)
        for sample in out_pbar:
            user_row, wap_column, scaled_RSSI, test_weight = prepare_test_data(test_X.iloc[sample, :], nan_penalty)
            user_coor = torch.randn(len(sample), 3).requires_grad_(True)
            wap_coor_expanded = expand_coor(wap_coor, wap_column)
            optimizer = Adam([user_coor], lr=1e-2)
            inner_pbar = trange(n_test_itr, position=1, leave=False)
            for _ in inner_pbar:
                optimizer.zero_grad()
                user_coor_expanded = expand_coor(user_coor, user_row)
                predicted_RSSI = self.net(info_tensor(user_coor_expanded, wap_coor_expanded))
                loss = torch.mean(torch.abs(predicted_RSSI - scaled_RSSI) * test_weight)
                loss.backward()
                optimizer.step()
                if _ % 10 == 0:
                    inner_pbar.set_description(f'Loss: {loss.item():.3e}')
            out_pbar.set_description(f'Loss: {loss.item():.3e}')
            coords[sample, :] = user_coor.detach().numpy()
        df = DataFrame(data=coords, columns=COL.COOR3)
        df[COL.FLR] = df[COL.FLR].astype(int)
        df[COL.BID] = self.clf.predict(df[[COL.LON, COL.LAT]].values)
        df[COL.COOR2] = self.scaler.inverse_transform(df[COL.COOR2])
        return df, dict()


class Test(TestCase):
    def test_shape(self):
        df = pd.read_csv('UJIndoorLoc/trainingData.csv')
        test_df = pd.read_csv('UJIndoorLoc/validationData.csv')

        gmma = GMMApproach(df.sample(frac=0.01), batch_size=10)
        gmma.train()
        gmma.interpret()
        gmma.estimate(test_df.sample(frac=0.1)[COL.WAPs], batch_size=10)
