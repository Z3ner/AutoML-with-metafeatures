import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.datasets import make_regression


class AutoRegressor:

    def __init__(self, kernel=None, n_classifiers=3):

        self.regressors = []
        self.n_classifiers = n_classifiers

        for _ in range(n_classifiers):
            self.regressors += [GaussianProcessRegressor(kernel=kernel)]


    def train(self, dataset, fold = 0):

        dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)
        x = dataTrain[:, :-1]

        for i in range(self.n_classifiers):
            y = dataset.accuracies[::i]
            self.regressors[i].fit(x, y)

        return

    def predict(self, x):
        out = []
        for regressor in self.regressors:
            out += [regressor.predict(x)]

        return out