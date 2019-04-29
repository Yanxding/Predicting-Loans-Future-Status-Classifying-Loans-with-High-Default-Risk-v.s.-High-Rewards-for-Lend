'''
Project: Predicting Loan Status
Script: Train-Test Split & Oversampling
Auther: Yanxiang Ding
Date: March 2019

Data used: All Lending Club loan data
    (https://www.kaggle.com/wordsforthewise/lending-club)
'''


# ========== import packages ==========
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn import model_selection as ms
from sklearn import datasets, metrics, tree
from imblearn import pipeline as pl
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt5


# ========== train-test split ==========
x_new = pd.read_csv('selected_data.csv',sep='\t',index_col=0)
y = pd.read_csv('data_y_newnew.csv')

train_index = pd.read_csv('trainindex.csv')
test_index = pd.read_csv('testindex.csv')
train_index = [i-1 for i in list(train_index['index'])]
test_index = [i-1 for i in list(test_index['index'])]

train_x = x_new.iloc[train_index].reset_index(drop=True)
train_y = y.iloc[train_index].reset_index(drop=True)
test_x = x_new.iloc[test_index].reset_index(drop=True)
test_y = y.iloc[test_index].reset_index(drop=True)

train_x.to_csv('train_x.csv',sep='\t')
train_y.to_csv('train_y.csv',sep='\t')
test_x.to_csv('test_x.csv',sep='\t')
test_y.to_csv('test_y.csv',sep='\t')


# ========== SMOTE oversampling ==========
# resample only minority class in training set
# 1. select best resampling parameter
scorer = metrics.make_scorer(metrics.cohen_kappa_score)
smote = SMOTENC(categorical_features=list(range(70)), sampling_strategy = {3:50000, 4:50000, 5:50000}, random_state=42, n_jobs=-1)
cart = tree.DecisionTreeClassifier(random_state=42)
pipeline = pl.make_pipeline(smote, cart)

param_range = range(5, 11)
train_scores, test_scores = ms.validation_curve(pipeline, train_x,train_y['level_cor'], param_name='smotenc__k_neighbors', 
                                                param_range=param_range, cv=3, scoring=scorer, n_jobs=-1)

# 2. visualize parameter selection
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(param_range, test_scores_mean, label='SMOTE')
ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)
idx_max = np.argmax(test_scores_mean)
plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
            label=r'Cohen Kappa: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))

plt.title("Validation Curve with SMOTE-CART")
plt.xlabel("k_neighbors")
plt.ylabel("Cohen's kappa")

# 3. resampling training set
smote = SMOTENC(categorical_features=list(range(70)), sampling_strategy = {3:50000, 4:50000, 5:50000}, k_neighbors=7, n_jobs=1)
x_new,y_new = smote.fit_resample(train_x,train_y['level_cor'])
new = pd.DataFrame(x_new,columns=list(train_x.columns))
new['y'] = y_new
