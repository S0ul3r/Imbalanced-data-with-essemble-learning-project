from enum import unique
import enum
from lib2to3.pytree import Base
from msilib.schema import Class
import scipy
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn import datasets, random_projection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from scipy.spatial import distance
from scipy import stats
from scipy.spatial import distance
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector, SelectPercentile, SelectKBest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import rankdata
from scipy.stats import ranksums
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

scores = np.load('results/recall_results.npy').T
#print(scores.shape)
m_scores = np.mean(scores, axis = 1)
me_scores = [[m] for m in m_scores]
mean_scores = np.array(me_scores)
#print(mean_scores)

datasets = ['ecoli-0-1_vs_2-3-5', 'ecoli-0-1_vs_5', 'ecoli-0-1-4-6_vs_5', 'ecoli-0-3-4_vs_5', 'ecoli-0-4-6_vs_5',
            'ecoli-0-6-7_vs_5', 'glass-0-1-4-6_vs_2', 'glass-0-1-5_vs_2', 'glass-0-1-6_vs_2', 'glass-0-1-6_vs_5', 'glass-0-4_vs_5',
            'glass-0-6_vs_5']

ens_d = {
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'RandomForest': RandomForestClassifier()
}

n_datasets = len(datasets)
n_ens = len(ens_d)
n_splits = 5
n_repeats = 2

alpha = 0.05

t_stat = np.zeros(shape=(n_ens, n_ens), dtype=float)
p_val = np.zeros(shape=(n_ens, n_ens), dtype=float)
adv = np.zeros(shape=(n_ens, n_ens), dtype=float)
sig = np.zeros(shape=(n_ens, n_ens), dtype=float)
s_better = np.zeros(shape=(n_ens, n_ens), dtype=float)

for i in range(n_ens):
    for j in range(n_ens):
            t_stat[i,j], p_val[i,j] = ttest_rel(scores.T[i] , scores.T[j])

headers = list(ens_d.keys())
names_column = np.expand_dims(np.array(list(ens_d.keys())), axis=1)

adv[t_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

#print(mean_scores)

mean_table = tabulate(np.concatenate((names_column, mean_scores), axis = 1), floatfmt=".2f")

t_stat_table = tabulate(np.concatenate((names_column, t_stat), axis = 1), headers, floatfmt=".2f")
p_val_table = tabulate(np.concatenate((names_column, p_val), axis = 1), headers, floatfmt=".2f")
adv_table = tabulate(np.concatenate((names_column, adv), axis = 1), headers, floatfmt=".2f")
sig_table = tabulate(np.concatenate((names_column, sig), axis = 1), headers, floatfmt=".2f")
result_table = tabulate(np.concatenate((names_column, s_better), axis = 1), headers, floatfmt=".2f")

print(f"\navg recall:\n{mean_table}\n")
print(f"t-stat:\n{t_stat_table}\n\np-value:\n{p_val_table}\n\nadvantage:\n{adv_table}\n\nsignificance:\n{sig_table}\n\nstat-better:\n{result_table}\n")