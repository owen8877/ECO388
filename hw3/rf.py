from datetime import datetime
from typing import Tuple
from unittest import TestCase

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from torch.optim import Adam
from torch.utils.data import RandomSampler, BatchSampler
from tqdm.auto import trange, tqdm

from hw3.common import COL, Estimator
from lib import ensure_dir


class RFApproach(Estimator):
    def __init__(self, train_df: DataFrame, save_path: str = None, subsample_frac: float = 1e-2):
        super().__init__(train_df, 'tree')

        self.subsample_frac = subsample_frac
        feature_transformer = FeatureUnion(transformer_list=[
            ('feature', SimpleImputer()),
            ('indicators', MissingIndicator(missing_values=100))
        ])
        self.coor_tree = make_pipeline(feature_transformer,
                                       MultiOutputRegressor(RandomForestRegressor(n_estimators=100), n_jobs=-1))
        self.floor_tree = make_pipeline(feature_transformer,
                                        MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs=-1))
        if save_path is None:
            # self.train_building_classifier()
            pass
        else:
            # self.clf = joblib.load(f'{save_path}/clf.joblib')
            self.coor_tree = joblib.load(f'{save_path}/coor_tree.joblib')
            self.floor_tree = joblib.load(f'{save_path}/floor_tree.joblib')

    def save(self):
        dt = datetime.now()
        path = f'param/{self.name}_{dt.strftime("%Y%m%d_%H%M%S")}'
        ensure_dir(path)
        # joblib.dump(self.clf, f'{path}/clf.joblib')
        joblib.dump(self.coor_tree, f'{path}/coor_tree.joblib')
        joblib.dump(self.floor_tree, f'{path}/floor_tree.joblib')

    def train(self, n_epoch: int = 2, n_mini_inner: int = 1):
        df = self.train_df.sample(frac=self.subsample_frac)
        self.coor_tree.fit(df[COL.WAPs], df[COL.COOR2])
        self.floor_tree.fit(df[COL.WAPs], df[[COL.FLR, COL.BID]])

    def get_wap_locations(self) -> DataFrame:
        sub_df = self.train_df.sample(frac=self.subsample_frac)
        weight = np.power(sub_df[COL.WAPs].replace(100, np.nan) / 104.0 + 1.0, 0.3).fillna(0).values + 1e-12
        user_coor = sub_df[COL.COOR3].values
        df = DataFrame(data=(weight.T @ user_coor) / weight.sum(axis=0)[:, None], columns=COL.COOR3)
        df[COL.BID] = self.clf.predict(self.scaler.transform(df[COL.COOR2].values))
        return df

    def interpret(self):
        pass

    def estimate(
            self, test_X: DataFrame, n_test_itr: int = 100, batch_size: int = 500,
            nan_penalty: float = 0.0,
    ) -> Tuple[DataFrame, dict]:
        coor = self.coor_tree.predict(test_X)
        flr_bid = self.floor_tree.predict(test_X)
        return DataFrame({
            COL.COOR2[0]: coor[:, 0],
            COL.COOR2[1]: coor[:, 1],
            COL.FLR: flr_bid[:, 0],
            COL.BID: flr_bid[:, 1],
        }), dict()


class Test(TestCase):
    def test_shape(self):
        df = pd.read_csv('UJIndoorLoc/trainingData.csv')
        test_df = pd.read_csv('UJIndoorLoc/validationData.csv')

        rfa = RFApproach(df.sample(frac=0.01), subsample_frac=1e-2)
        rfa.train()
        rfa.interpret()
        rfa.estimate(test_df.sample(frac=0.1)[COL.WAPs], batch_size=10)
