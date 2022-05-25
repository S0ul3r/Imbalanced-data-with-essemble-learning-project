import sklearn as sk
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.tree import DecisionTreeClassifier
from myForest import myForEns

# bagging;
# random forest; 
# adaboost;



datasets = ['ecoli-0-1_vs_2-3-5', 'ecoli-0-1_vs_5', 'ecoli-0-1-4-6_vs_5', 'ecoli-0-3-4_vs_5', 'ecoli-0-4-6_vs_5',
            'ecoli-0-6-7_vs_5', 'glass-0-1-4-6_vs_2', 'glass-0-1-5_vs_2', 'glass-0-1-6_vs_2', 'glass-0-1-6_vs_5', 'glass-0-4_vs_5',
            'glass-0-6_vs_5']
names = ["accuracy_results", "error_results", "precision_results", "recall_results", "f1_results", "gmean_results"]

ens_d = {
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200),
    'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200),
    'RandomForest': RandomForestClassifier(n_estimators=200),
    'myForest': myForEns(n_trees=200)
}

n_datasets = len(datasets)
n_clfs = len(ens_d)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1337)

a_scores, e_scores, p_scores, r_scores, f1_scores, gmean_scores = (np.zeros(shape=(n_datasets, n_splits * n_repeats, n_clfs), dtype=float) for _ in range(6))
scores = [a_scores, e_scores, p_scores, r_scores, f1_scores, gmean_scores]

for dataset_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt(f"datasets/{dataset}.csv", delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for ens_id, ens_name in enumerate(ens_d):
            ens = sk.clone(ens_d[ens_name])
            ens.fit(X[train], y[train])
            prediction = ens.predict(X[test])
            a_scores[dataset_id, fold_id, ens_id] = accuracy_score(y[test], prediction)
            e_scores[dataset_id, fold_id, ens_id] = 1 - accuracy_score(y[test], prediction)
            p_scores[dataset_id, fold_id, ens_id] = precision_score(y[test], prediction, zero_division=0)
            r_scores[dataset_id, fold_id, ens_id] = recall_score(y[test], prediction, zero_division=0)
            f1_scores[dataset_id, fold_id, ens_id] = f1_score(y[test], prediction, zero_division=0)
            gmean_scores[dataset_id, fold_id, ens_id] = geometric_mean_score(y[test], prediction)

for id, score in enumerate(scores):
    np.save(f'results/{names[id]}', score)