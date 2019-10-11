import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.datasets import make_regression
from Classifier import Classifier


class AutoClassifier(Classifier):

    def __init__(self, kernel=None, n_classifiers=3):
        self.model = GaussianProcessClassifier(kernel=kernel)
        self.n_classifiers = n_classifiers

    def train(self, dataset, fold=0):
        dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)
        x = dataTrain[:, :-1]
        y = dataTrain[:, -1]

        self.model.fit(x, y)

        return

    def predict(self, x):
        # out = self.model.predict_proba(x)
        return self.model.predict(x)


class AutoRegressor:

	def __init__(self, kernel=None, n_classifiers=3):

		self.regressors = []
		self.n_classifiers = n_classifiers

		for _ in range(n_classifiers):
			self.regressors += [GaussianProcessRegressor(kernel=kernel)]

	def train(self, dataset, fold=0):

		dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)
		x = dataTrain[:, :-1]
		accuracies = np.array(dataset.accuracies)
		accuracies = accuracies[dataset.folds[fold].trainIndexes]
		print(accuracies)
		for i in range(self.n_classifiers):

			y = accuracies[:,i]
			self.regressors[i].fit(x, y)

		return

	def predict(self, x):
		out = []
		for regressor in self.regressors:
			out += [regressor.predict(x)]

		return out


"""class AutoRegressor(Classifier):

	def __init__(self, kernel = None, label_id = 0):

		self.model = GaussianProcessRegressor(kernel=kernel)

		self.label_id = label_id

	def train(self, dataset, fold = 0):

		n_labels = dataset.n_labels
		dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)

		x = dataTrain[:,:-n_labels]
		y = dataTrain[:,-n_labels + self.label_id]

		self.model.fit(x, y)

		return

	def predict(self, x):

		return self.model.predict(x)"""
