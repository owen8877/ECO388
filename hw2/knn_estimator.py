import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsRegressor

mpl.use('TkAgg')

df = pd.read_stata('../hw1/WAGE1.DTA')

(train, test), _ = KFold(n_splits=2, shuffle=True).split(df)
train_df = df.loc[train]
test_df = df.loc[test]


def is_categorical(df, column):
    return len(np.unique(df[column])) < 0.02 * len(df)


def knn_estimator(covariates, train_df, test_df, k):
    continuous_covariates = []
    categorical_covariates = []
    for covariate in covariates:
        if is_categorical(train_df, covariate):
            categorical_covariates.append(covariate)
        else:
            continuous_covariates.append(covariate)

    train_cat_comb = train_df[categorical_covariates].apply(tuple, axis=1)
    test_cat_comb = test_df[categorical_covariates].apply(tuple, axis=1)
    test_unique_cat_combination = test_cat_comb.unique()

    knn = KNeighborsRegressor(n_neighbors=k)
    preds = []
    for comb in test_unique_cat_combination:
        train_sub_df = train_df[train_cat_comb == comb]
        test_sub_df = test_df[test_cat_comb == comb]
        knn.fit(train_sub_df[continuous_covariates], train_sub_df['wage'])
        pred = Series(data=knn.predict(test_sub_df[continuous_covariates]), index=test_sub_df.index)
        preds.append(pred)
    return pd.concat(preds).sort_index()


covariates = {'educ', 'exper', 'female'}
lambdas = np.arange(0.3, 1.3, 0.1)
errors = []
for l in lambdas:
    k = int(l * np.sqrt(len(train_df)))
    test_wage_pred = knn_estimator(covariates, train_df, test_df, k)
    error = np.mean((test_wage_pred - test_df['wage']) ** 2)
    errors.append(error)
    print(f'error={error:.3f}, k={k:d}')
fig = plt.figure(figsize=(7, 3))
plt.plot(lambdas, errors)
plt.gca().set(xlabel=r'$\lambda$', ylabel='MISE')
plt.title(f'Using covariates: {covariates}')
fig.savefig('plots/knn-educ-exper-female.pdf')

covariates = {'educ', 'tenure', 'profocc', 'female'}
lambdas = np.arange(0.3, 1.3, 0.1)
errors = []
for l in lambdas:
    k = int(l * np.sqrt(len(train_df)))
    test_wage_pred = knn_estimator(covariates, train_df, test_df, k)
    error = np.mean((test_wage_pred - test_df['wage']) ** 2)
    errors.append(error)
    print(f'error={error:.3f}, k={k:d}')
fig = plt.figure(figsize=(7, 3))
plt.plot(lambdas, errors)
plt.gca().set(xlabel=r'$\lambda$', ylabel='MISE')
plt.title(f'Using covariates: {covariates}')
fig.savefig('plots/knn-educ-tenure-profocc-female.pdf')

plt.show()
