from itertools import chain
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import Sampler, RandomSampler, BatchSampler
from tqdm.auto import trange, tqdm

from hw3.common import COL

COOR = [COL.LON, COL.LAT, COL.FLR]


class Polynomial(nn.Module):
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(degree + 1))

    def forward(self, x):
        x_i, z_i = [1], [0]
        for i in range(self.degree + 1):
            z_i.append(z_i[-1] + x_i[-1] * self.coeff[i])
            if i < self.degree:
                x_i.append(x_i[-1] * x)
        return z_i[-1]


def distance_calculation(user_coor_expanded, wap_coor_expanded, floor_weight):
    adjust_weight = torch.concat([torch.tensor([1, 1]), floor_weight])
    diff = (user_coor_expanded - wap_coor_expanded) * adjust_weight[None, :]
    return torch.sqrt((diff ** 2).sum(dim=1))[:, None]


def expand_coor(coor, index):
    return torch.index_select(coor, 0, index)


def normalize_RSSI(raw_RSSI):
    return 1.0 / np.power(1 + raw_RSSI / 104.1, 0.2)


def prepare_train_data(df: DataFrame):
    user_coor = torch.from_numpy(df[COOR].values)
    query_row_np, query_column_np = np.where(df[COL.WAPs] != 100)
    user_row = torch.from_numpy(query_row_np)
    wap_column = torch.from_numpy(query_column_np)
    raw_RSSI = torch.tensor([df.iloc[row, column] for row, column in zip(query_row_np, query_column_np)])
    RSSI = normalize_RSSI(raw_RSSI)[:, None]
    train_weight = 1 / RSSI

    return user_coor, user_row, wap_column, RSSI, train_weight / torch.mean(train_weight)


def prepare_test_data(s: Series, wap_coor: Tensor):
    query_column_np, = np.where(s[COL.WAPs] != 100)
    wap_column = torch.from_numpy(query_column_np)
    tuser_coor_init = torch.index_select(wap_coor, 0, wap_column).mean(dim=0).clone().detach().requires_grad_(True)
    raw_RSSI = torch.from_numpy(s.iloc[query_column_np].values)
    RSSI = normalize_RSSI(raw_RSSI)[:, None]
    train_weight = 1 / RSSI

    return tuser_coor_init, wap_column, RSSI, train_weight / torch.mean(train_weight)


def wap_init(df: DataFrame) -> Tensor:
    sub_df = df.sample(frac=0.03)
    weight = np.power(sub_df[COL.WAPs].replace(100, np.nan) / 104.0 + 1.0, 0.3).fillna(0).values + 1e-12
    user_coor = sub_df[COOR].values
    return torch.from_numpy((weight.T @ user_coor) / weight.sum(axis=0)[:, None])


def wap_based_estimate(
        train_df: DataFrame, test_X: DataFrame,
        n_epoch: int = 2, n_mini_inner: int = 1, n_test_itr: int = 100,
) -> Tuple[DataFrame, Series]:
    all_indices = np.arange(len(train_df))
    sampler = BatchSampler(RandomSampler(all_indices), batch_size=100, drop_last=False)

    wap_coor = wap_init(train_df).requires_grad_(True)
    floor_weight = torch.tensor([4.0], requires_grad=True)
    polynomial = Polynomial(degree=3)
    parameters = chain([wap_coor, floor_weight], polynomial.parameters())
    optimizer = Adam(parameters, lr=1e-2)

    epoch_pbar = trange(n_epoch, position=0)
    for epoch in epoch_pbar:
        sample_pbar = tqdm(sampler, position=1)
        losses = []
        for sample in sample_pbar:
            user_coor, user_row, wap_column, RSSI, train_weight = prepare_train_data(train_df.iloc[sample, :])
            # print(user_coor.shape)
            # print(train_weight.shape)
            user_coor_expanded = expand_coor(user_coor, user_row)

            for _ in range(n_mini_inner):
                optimizer.zero_grad()
                wap_coor_expanded = expand_coor(wap_coor, wap_column)
                # print(wap_coor_expanded.shape)
                RSSI_pred = polynomial(distance_calculation(user_coor_expanded, wap_coor_expanded, floor_weight) / 50.0)
                loss = torch.mean((RSSI_pred - RSSI) ** 2 * train_weight)
                # if torch.any(torch.isnan(loss)):
                # print(torch.any(torch.isnan(train_weight)))
                # print(torch.any(torch.isnan(RSSI)))
                # print(torch.any(torch.isnan(RSSI_pred)))
                # print(train_weight)
                # print(RSSI)
                # print(RSSI_pred)
                # raise
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            sample_pbar.set_description(f'Loss: {np.mean(losses):.3f}')
        epoch_pbar.set_description(f'Last loss: {np.mean(losses):.3f}')

    # Now move to the testing phase
    test_coords = []
    wap_coor_detached = wap_coor.detach()
    floor_weight_detached = floor_weight.detach()
    out_pbar = trange(len(test_X))
    for i in out_pbar:
        tuser_coor, twap_column, tRSSI, test_weight = prepare_test_data(test_X.iloc[i, :], wap_coor_detached)
        if len(twap_column) > 1:
            twap_coor_expanded = expand_coor(wap_coor_detached, twap_column)
            tuser_coor.requires_grad_(True)
            toptimizer = Adam([tuser_coor], lr=1e-2)
            for _ in range(n_test_itr):
                toptimizer.zero_grad()
                tRSSI_pred = polynomial(
                    distance_calculation(tuser_coor[None, :], twap_coor_expanded, floor_weight_detached) / 50.0)
                tloss = torch.mean((tRSSI_pred - tRSSI) ** 2 * test_weight)
                tloss.backward()
                toptimizer.step()
            out_pbar.set_description(f'Loss: {tloss.item():.3f}')
        test_coords.append(Series(tuser_coor.detach().numpy().squeeze()))
    return pd.concat(test_coords, axis=1).T
