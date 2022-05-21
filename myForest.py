import numpy as np
import enum
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class myForEns(BaseEnsemble, ClassifierMixin):
    def __init__(self, n_trees = 100, max_features = 'sqrt', random_state = None):

        self.base_estimator = DecisionTreeClassifier()
        self.n_trees = n_trees
        self.random_state = random_state
        self.max_features = max_features

        np.random.seed(self.random_state)

    def fit(self, X, y):
        
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)

        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        self.samples = np.random.randint(0, self.n_samples, (self.n_trees, self.n_samples))

        self.subfeat = []
        if self.max_features == 'sqrt':
            for i in range(self.n_trees):
                self.subfeat.append(np.random.choice(self.n_features, round(np.sqrt(self.n_features)), replace=False))
        elif self.max_features == 'log2':
            for i in range(self.n_trees):
                self.subfeat.append(np.random.choice(self.n_features, round(np.log2(self.n_features)), replace=False))
        else:
            raise ValueError("max features not specified")
        self.subfeat = np.array(self.subfeat)
        #print(self.subfeat[0])
        self.bags = []

        for i in range(self.n_trees):
            self.bags.append(X[self.samples[i]])
        self.bags = np.array(self.bags)
        #print(self.bags[0, :, [1,2,3]].T)

        #print(self.bags.shape)
        #print(self.subfeat.shape)

        self.subset = np.zeros(shape=(self.n_trees, self.n_samples, self.subfeat.shape[1]), dtype=float)
        for i in range(self.n_trees):
            #print(self.bags[i, :, self.subfeat[i]].T)
            self.subset[i] = self.bags[i, :, self.subfeat[i]].T
        #print(self.subset[1])

        self.ensamble_ = []
        for i in range(self.n_trees):
            self.ensamble_.append(clone(self.base_estimator).fit(self.subset[i], y[self.samples[i]]))

        return self
    def predict(self, X):

        check_is_fitted(self, "classes_")

        X = check_array(X)

        pred_ = []
        
        #print(X.shape)
        for i, member_clf in enumerate(self.ensamble_):
                pred_.append(member_clf.predict(X[:, self.subfeat[i]]))
        
        pred_ = np.array(pred_)
        
        prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
        
        return self.classes_[prediction]

X, y = datasets.make_classification(n_samples=200, n_features=10, n_informative=10, n_redundant = 0, n_repeated=0, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1234)

myForestEns = myForEns(random_state=1234)

myForestEns.fit(X_train, y_train)
predict = myForestEns.predict(X_test)
score = accuracy_score(y_test, predict)
print(score)
