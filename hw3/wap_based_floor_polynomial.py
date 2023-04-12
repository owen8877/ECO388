from datetime import datetime
from itertools import chain
from typing import Tuple

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import RandomSampler, BatchSampler
from tqdm.auto import trange, tqdm

from hw3.common import COL, Estimator
from hw3.direct_kernel_regression import DirectKernelRegression
from lib import ensure_dir


class Polynomial(nn.Module):
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(degree + 1))

    def forward(self, x):
        x_i, z_i = [1], [0]
        for i in range(self.degree + 1):
            c = self.coeff[i] if i == 0 else torch.exp(self.coeff[i])
            z_i.append(z_i[-1] + x_i[-1] * c)
            if i < self.degree:
                x_i.append(x_i[-1] * x)
        return z_i[-1]


def distance_calculation(user_coor_expanded, wap_coor_expanded, floor_weight):
    adjust_weight = torch.concat([torch.tensor([1, 1]), floor_weight])
    diff = (user_coor_expanded - wap_coor_expanded) * adjust_weight[None, :]
    return torch.sqrt((diff ** 2).sum(dim=1))[:, None]


def expand_coor(coor, index):
    return torch.index_select(coor, 0, index)


def RSSI_to_signal_loss(raw_RSSI):
    return np.power(1 + raw_RSSI / 104.1, -0.5)
    # return -raw_RSSI / 104.0


def signal_loss_to_RSSI(signal_loss):
    return (np.power(signal_loss, -2.0) - 1) * 104.1


def prepare_train_data(WAPs: DataFrame, coor: DataFrame):
    count = (WAPs != 100).sum(axis=1)
    user_coor = torch.from_numpy(coor.values)
    query_row_np, query_column_np = np.where(WAPs != 100)
    user_row = torch.from_numpy(query_row_np)
    wap_column = torch.from_numpy(query_column_np)
    raw_RSSI = torch.tensor([WAPs.iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
    signal_loss = RSSI_to_signal_loss(raw_RSSI)[:, None]
    count_weight = torch.from_numpy((1 / count).iloc[query_row_np].values)[:, None]
    signal_weight = 1 / signal_loss

    train_weight = count_weight * signal_weight
    # train_weight = np.power(1 + signal_loss, -2.0)
    # signal_loss = raw_RSSI[:, None]
    # train_weight = signal_loss * 0 + 1.0

    return user_coor, user_row, wap_column, signal_loss, train_weight / torch.mean(train_weight)


def prepare_test_data(df: Series, nan_penalty: float):
    if np.isclose(nan_penalty, 0):
        query_row_np, query_column_np = np.where(df[COL.WAPs] != 100)
        count = (df[COL.WAPs] != 100).sum(axis=1)
        user_row = torch.from_numpy(query_row_np)
        wap_column = torch.from_numpy(query_column_np)
        raw_RSSI = torch.tensor([df.iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
        signal_loss = RSSI_to_signal_loss(raw_RSSI)[:, None]
        count_weight = torch.from_numpy((1 / count).iloc[query_row_np].values)[:, None]
        signal_weight = 1 / signal_loss
    else:
        missing_data = df[COL.WAPs] == 100
        query_row_np, query_column_np = np.where(df[COL.WAPs] < 101)
        user_row = torch.from_numpy(query_row_np)
        wap_column = torch.from_numpy(query_column_np)
        raw_RSSI = torch.tensor(
            [df.replace(100, -104).iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
        signal_loss = RSSI_to_signal_loss(raw_RSSI)[:, None]
        count_weight = torch.tensor(
            [(missing_data * nan_penalty + (1 - missing_data)).iloc[row, column] for row, column in
             zip(query_row_np, query_column_np)])[:, None]
        signal_weight = 1 / signal_loss

    train_weight = count_weight * signal_weight
    # train_weight = np.power(1 + signal_loss, -2.0)
    # signal_loss = raw_RSSI[:, None]
    # train_weight = signal_loss * 0 + 1.0

    return user_row, wap_column, signal_loss, train_weight / torch.mean(train_weight)


def wap_init(df: DataFrame) -> Tensor:
    de = DirectKernelRegression(df, subsample=0.05)
    estimated_location = de.get_wap_locations().values[:, :3]
    estimated_location[:, :2] = np.nan_to_num(de.scaler.transform(estimated_location[:, :2]))
    estimated_location[:, 2] = np.nan_to_num(estimated_location[:, 2], nan=2.5)
    return torch.from_numpy(estimated_location)


class FloorPolynomialApproach(Estimator):
    def __init__(self, train_df: DataFrame, save_path: str = None, polynomial_degree: int = 3):
        super().__init__(train_df, 'fpa')
        self.polynomial = Polynomial(polynomial_degree)
        if save_path is None:
            self.train_building_classifier()
            self.wap_coor = wap_init(self.train_df).requires_grad_(True)
            # self.wap_coor = torch.zeros(len(COL.WAPs), 3).requires_grad_(True)
            self.floor_weight = torch.tensor([4.0 / 50.0], requires_grad=True)
        else:
            self.wap_coor = torch.load(f'{save_path}/wap_coor.pt').requires_grad_(True)
            self.floor_weight = torch.load(f'{save_path}/floor_weight.pt').requires_grad_(True)
            self.polynomial.load_state_dict(torch.load(f'{save_path}/poly.pt'))
            self.clf = joblib.load(f'{save_path}/clf.joblib')
            # self.scaler = joblib.load(f'{save_path}/scaler.joblib')
        parameters = chain([self.wap_coor, self.floor_weight], self.polynomial.parameters())
        self.optimizer = Adam(parameters, lr=1e-2)

    def save(self):
        dt = datetime.now()
        path = f'param/{self.name}_{dt.strftime("%Y%m%d_%H%M%S")}'
        ensure_dir(path)
        torch.save(self.polynomial.state_dict(), f'{path}/poly.pt')
        torch.save(self.wap_coor, f'{path}/wap_coor.pt')
        torch.save(self.floor_weight, f'{path}/floor_weight.pt')
        joblib.dump(self.clf, f'{path}/clf.joblib')
        joblib.dump(self.scaler, f'{path}/scaler.joblib')

    def train(self, n_epoch: int = 2, n_mini_inner: int = 1, batch_size: int = 500):
        all_indices = np.arange(len(self.train_df))
        sampler = BatchSampler(RandomSampler(all_indices), batch_size=batch_size, drop_last=False)

        epoch_pbar = trange(n_epoch, position=0)
        for epoch in epoch_pbar:
            sample_pbar = tqdm(sampler, position=1)
            losses = []
            for sample in sample_pbar:
                user_coor, user_row, wap_column, signal_loss, train_weight = prepare_train_data(
                    self.train_df.iloc[sample, :][COL.WAPs], self.transformed_coordinates.iloc[sample, :][COL.COOR3])
                user_coor_expanded = expand_coor(user_coor, user_row)

                for _ in range(n_mini_inner):
                    self.optimizer.zero_grad()
                    wap_coor_expanded = expand_coor(self.wap_coor, wap_column)
                    signal_loss_pred = self.polynomial(
                        distance_calculation(user_coor_expanded, wap_coor_expanded, self.floor_weight))
                    # if _ == n_mini_inner - 1:
                    #     return signal_loss_pred, signal_loss, train_weight
                    loss = torch.mean((signal_loss_pred - signal_loss) ** 2 * train_weight)
                    loss.backward()
                    self.optimizer.step()

                losses.append(loss.item())
                sample_pbar.set_description(f'Loss: {np.mean(losses):.3f}')
            epoch_pbar.set_description(f'Last loss: {np.mean(losses):.3f}')

    def get_wap_locations(self) -> DataFrame:
        wap_coor = self.wap_coor.detach()
        df = DataFrame(data=wap_coor, columns=COL.COOR3)
        df[COL.BID] = self.clf.predict(df[[COL.LON, COL.LAT]].values)
        df[COL.COOR2] = self.scaler.inverse_transform(df[COL.COOR2])
        return df

    def interpret(self):
        sample = np.random.randint(0, len(self.train_df), int(0.02 * len(self.train_df)))
        user_coor, user_row, wap_column, signal_loss, _ = prepare_train_data(
            self.train_df[COL.WAPs].iloc[sample, :], self.transformed_coordinates[COL.COOR3].iloc[sample, :])
        RSSI = signal_loss_to_RSSI(signal_loss)
        user_coor_expanded = expand_coor(user_coor, user_row)
        wap_coor_expanded = expand_coor(self.wap_coor, wap_column)
        distance_scaled = distance_calculation(user_coor_expanded, wap_coor_expanded, self.floor_weight)
        distance_grid = torch.linspace(0, distance_scaled.max().item(), 100)
        signal_loss_pred = self.polynomial(distance_grid)
        distance = distance_scaled * self.scaler.scale_

        signal_loss, signal_loss_pred, RSSI, distance_grid, distance_scaled, distance = [
            x.detach().numpy().squeeze() for x in (
                signal_loss, signal_loss_pred, RSSI, distance_grid, distance_scaled, distance)]
        df = DataFrame(
            {'signal_loss': signal_loss, 'RSSI': RSSI, 'distance_scaled': distance_scaled, 'distance': distance})

        plt.figure(1)
        g1 = sns.jointplot(data=df, x='distance_scaled', y='signal_loss', kind='scatter', s=3)
        sns.kdeplot(data=df, x='distance_scaled', y='signal_loss', ax=g1.ax_joint)
        g1.ax_joint.plot(distance_grid, signal_loss_pred, '--')
        g1.ax_joint.set(ylim=[g1.ax_joint.get_ylim()[0], 5])

        plt.figure(2)
        g2 = sns.jointplot(data=df, x='distance', y='RSSI', kind='kde')

        return g1, g2

    def estimate(
            self, test_X: DataFrame, n_test_itr: int = 100, batch_size: int = 500,
            nan_penalty: float = 0.0,
    ) -> Tuple[DataFrame, dict]:
        coords = np.zeros((len(test_X), 3))
        wap_coor = self.wap_coor.detach()
        floor_weight = self.floor_weight.detach()

        all_indices = np.arange(len(test_X))
        sampler = BatchSampler(RandomSampler(all_indices), batch_size=batch_size, drop_last=False)

        out_pbar = tqdm(sampler, position=0)
        for sample in out_pbar:
            user_row, wap_column, signal_loss, test_weight = prepare_test_data(test_X.iloc[sample, :], nan_penalty)
            user_coor = torch.randn(len(sample), 3).requires_grad_(True)
            wap_coor_expanded = expand_coor(wap_coor, wap_column)
            optimizer = Adam([user_coor], lr=1e-1)
            inner_pbar = trange(n_test_itr, position=1, leave=False)
            for _ in inner_pbar:
                optimizer.zero_grad()
                user_coor_expanded = expand_coor(user_coor, user_row)
                signal_loss_pred = self.polynomial(
                    distance_calculation(user_coor_expanded, wap_coor_expanded, floor_weight))
                loss = torch.mean((signal_loss_pred - signal_loss) ** 2 * test_weight)
                loss.backward()
                optimizer.step()
                if _ % 100 == 0:
                    inner_pbar.set_description(f'Loss: {loss.item():.3f}')
            out_pbar.set_description(f'Loss: {loss.item():.3f}')
            coords[sample, :] = user_coor.detach().numpy()
        df = DataFrame(data=coords, columns=COL.COOR3)
        df[COL.FLR] = df[COL.FLR].astype(int)
        df[COL.BID] = self.clf.predict(df[[COL.LON, COL.LAT]].values)
        df[COL.COOR2] = self.scaler.inverse_transform(df[COL.COOR2])
        return df, dict()
