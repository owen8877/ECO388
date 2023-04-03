from unittest import TestCase

import numpy as np
import pandas as pd

from hw3.common import COL, estimate_loss
from lib import ensure_dir


def wrapper(name, method):
    df = pd.read_csv('UJIndoorLoc/trainingData.csv')
    test_df = pd.read_csv('UJIndoorLoc/validationData.csv')

    prediction = method(df, test_df[COL.WAPs])

    loss_score, fig1, fig2, fig3, fig4 = estimate_loss(test_df[[COL.LON, COL.LAT, COL.FLR, COL.BID]], prediction)
    prefix = f'dump/{name}'
    ensure_dir(prefix)

    np.save(f'{prefix}/score_{loss_score:.3f}', 0)
    for i, fig in enumerate((fig1, fig2, fig3, fig4)):
        fig.savefig(f'{prefix}/fig{i + 1:d}.pdf')


class Test(TestCase):
    def test_direct_kernel_regression(self):
        from hw3.direct_kernel_regression import direct_kernel_regression_estimate

        wrapper('direct_kernel_regression',
                lambda train_df, test_X: direct_kernel_regression_estimate(train_df.sample(frac=0.03), test_X)[0])
