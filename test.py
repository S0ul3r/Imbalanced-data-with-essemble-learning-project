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

scores = np.load('results/accuracy_results.npy').T
print(scores)

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

adv[t_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

print(f"t-stat:\n{t_stat}\np-value:\n{p_val}\nadvantage:\n{adv}\nsignificance:\n{sig}\nstat-better:\n{s_better}")