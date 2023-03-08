import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from lib.estimator import uniform_pdf, epanechnikov_pdf, gaussian_pdf

mpl.use('TkAgg')

df = pd.read_stata('../hw1/WAGE1.DTA')

(train, test), _ = KFold(n_splits=2).split(df)
train_df = df.loc[train]
test_df = df.loc[test]


def kernel_regression_estimator(train_df, test_df, covariates, target='wage'):
    n = len(train_df)
    if not isinstance(covariates, dict):
        kernel_mapping = {}
        for i, covariate in enumerate(covariates):
            if len(np.unique(train_df[covariate])) < 0.02 * n:
                kernel_mapping[covariate] = uniform_pdf
            else:
                h = 1.06 * np.std(train_df[covariate]) * np.power(n, -1 / 5)
                kernel_mapping[covariate] = lambda z: epanechnikov_pdf(z / h)
    else:
        kernel_mapping = covariates

    covariates = list(kernel_mapping.keys())

    train_y = train_df[target]
    train_Z = train_df[covariates]

    test_Z = test_df[covariates]

    weight = 1
    for covariate, kernel in kernel_mapping.items():
        weight *= kernel((test_Z[covariate].values[None, :] - train_Z[covariate].values[:, None]))

    test_y_pred = np.average(train_y.values[:, None] * weight, axis=0) / np.average(weight, axis=0)
    return test_y_pred


cat_count, cont_count = 0, 0
for column in set(df.columns) - {'wage', 'lwage', 'expersq', 'tenursq'}:
    if len(train_df[column].unique()) < len(train_df) * 0.02:
        cat_count += 1
    else:
        cont_count += 1

figcat, ax_cats = plt.subplots(2, cat_count // 2, figsize=(20, 10), sharey=True)
figcont, ax_conts = plt.subplots(cont_count, 1, figsize=(8, 12))
i_cat, i_cont = 0, 0
diffs = {}
for column in set(df.columns) - {'wage', 'lwage', 'expersq', 'tenursq'}:
    if len(train_df[column].unique()) < len(train_df) * 0.02:
        sns.violinplot(data=train_df, x=column, y='wage', orient='v', ax=ax_cats.flat[i_cat])
        selection = train_df[column] == 0
        diff = train_df.loc[selection, 'wage'].mean() - train_df.loc[~selection, 'wage'].mean()
        print(f'Difference between groups of {column}: {np.abs(diff):.3f}')
        diffs[column] = np.abs(diff)
        i_cat += 1
    else:
        sns.scatterplot(data=train_df, x=column, y='wage', ax=ax_conts[i_cont])
        i_cont += 1

figcat.savefig('plots/cat_violinplot.pdf')
figcont.savefig('plots/cont_violinplot.pdf')

# Given covariates - rule of thumb
cont1 = 'educ'
cont2 = 'exper'

h1 = 1.06 * np.std(train_df[cont1]) * np.power(len(train_df), -1 / 6)
h2 = 1.06 * np.std(train_df[cont2]) * np.power(len(train_df), -1 / 6)
test_y_pred = kernel_regression_estimator(train_df, test_df, {
    cont1: lambda z: gaussian_pdf(z / h1),
    cont2: lambda z: gaussian_pdf(z / h2),
    'female': uniform_pdf,
})
mise = np.mean((test_df['wage'] - test_y_pred) ** 2)

print(f'Using rule of thumb: {mise:.3f}')

# Given covariates - CV
cont1 = 'educ'
cont2 = 'exper'

h1s = np.linspace(0.2, 3, 7) * np.std(train_df[cont1]) * np.power(len(train_df), -1 / 6)
h2s = np.linspace(0.2, 3, 7) * np.std(train_df[cont2]) * np.power(len(train_df), -1 / 6)
mise = np.zeros((len(h1s), len(h2s)))
for i, h1 in enumerate(h1s):
    for j, h2 in enumerate(h2s):
        test_y_pred = kernel_regression_estimator(train_df, test_df, {
            cont1: lambda z: gaussian_pdf(z / h1),
            cont2: lambda z: gaussian_pdf(z / h2),
            'female': uniform_pdf,
        })
        mise[i, j] = np.mean((test_df['wage'] - test_y_pred) ** 2)

df = pd.DataFrame(data=mise, index=h1s, columns=h2s)
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, ax=ax, yticklabels=[f'{_:.2f}' for _ in df.index],
            xticklabels=[f'{_:.2f}' for _ in df.columns])
plt.xlabel(cont2)
plt.ylabel(cont1)
plt.title(f'Other covariates: female')
f.savefig(f'plots/female.pdf')

# New trials
cat_candidates = ['profocc', 'female', 'servocc', 'married', 'numdep']
cont1 = 'educ'
cont2 = 'tenure'

h1s = np.linspace(0.2, 3, 7) * np.std(train_df[cont1]) * np.power(len(train_df), -1 / 6)
h1_rot = 1.06 * np.std(train_df[cont1]) * np.power(len(train_df), -1 / 6)
h2s = np.linspace(0.2, 3, 7) * np.std(train_df[cont2]) * np.power(len(train_df), -1 / 6)
h2_rot = 1.06 * np.std(train_df[cont2]) * np.power(len(train_df), -1 / 6)
for cat_i in range(len(cat_candidates)):
    candidate_1 = cat_candidates[cat_i]

    for cat_j in range(cat_i + 1, len(cat_candidates)):
        candidate_2 = cat_candidates[cat_j]

        mise = np.zeros((len(h1s), len(h2s)))
        for i, h1 in enumerate(h1s):
            for j, h2 in enumerate(h2s):
                test_y_pred = kernel_regression_estimator(train_df, test_df, {
                    cont1: lambda z: gaussian_pdf(z / h1),
                    cont2: lambda z: gaussian_pdf(z / h2),
                    candidate_1: uniform_pdf,
                    candidate_2: uniform_pdf,
                })
                mise[i, j] = np.mean((test_df['wage'] - test_y_pred) ** 2)

        test_y_pred = kernel_regression_estimator(train_df, test_df, {
            cont1: lambda z: gaussian_pdf(z / h1_rot),
            cont2: lambda z: gaussian_pdf(z / h2_rot),
            candidate_1: uniform_pdf,
            candidate_2: uniform_pdf,
        })
        m = np.mean((test_df['wage'] - test_y_pred) ** 2)
        print(f'Using rule of thumb on {candidate_1}, {candidate_2}: {m:.3f}')

        df = pd.DataFrame(data=mise, index=h1s, columns=h2s)
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, ax=ax, yticklabels=[f'{_:.2f}' for _ in df.index],
                    xticklabels=[f'{_:.2f}' for _ in df.columns])
        plt.xlabel(cont2)
        plt.ylabel(cont1)
        plt.title(f'Other covariates: {candidate_1} and {candidate_2}')
        f.savefig(f'plots/{candidate_1},{candidate_2}.pdf')
