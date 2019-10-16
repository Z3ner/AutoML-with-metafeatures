import numpy as np
import random

from Dataset import Dataset
from Dataset import Fold
from Metafeatures import metadataset_row

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


class Metadataset(Dataset):

	def __init__(self, train_data = None, classifiers = None):

		self.data = []
		self.accuracies = []
		self.winners = []
		self.dataset_names = train_data

		for path in train_data:
			dataset = Dataset(path)
			#dataset.data = self.normalization(dataset.data)
			dataset.createFolds(mode = "Cross", k = 10)

			label = []
			for classifier in classifiers:
				c = classifier()
				acc, _ = c.score(dataset)
				label += [acc]

			self.n_labels = len(label)
			self.accuracies += [label]
			self.winners += [np.argmax(np.asarray(label))]

			self.data += [metadataset_row(dataset.data[:-1]).tolist() + [np.argmax(np.asarray(label))]]

		self.train_size = len(self.data)


		self.size = len(self.data)
		self.data = np.asarray(self.data)
		p_train = self.train_size/self.size
		self.createFoldsMetadataset()



	def featureSelection(self, x_data, y_data):

		clf = ExtraTreesClassifier(n_estimators=50)

		clf = clf.fit(x_data, y_data)

		model = SelectFromModel(clf, prefit=True)

		new_x_data = model.transform(x_data)

		return new_x_data

	def normalization(self, data):

		mean = []
		std = []
		features = np.transpose(data)[:-1]
		labels = np.transpose(data)[-1]

		"""
		isNaN = np.isnan(features.any())
		print("@@@@@ Is INF?", np.where(np.isinf(features)))
		"""

		for feature in features:
			mean += [np.mean(feature)]
			std += [np.std(feature)]

		feature_norm = []
		idx = 0
		for feature in features:
			for x in feature:
				feature_norm += [(x - mean[idx])/std[idx]]
			idx += 1

		feature_norm = np.array(feature_norm)
		feature_norm = np.reshape(feature_norm, (len(features),len(data)))

		features = np.transpose(feature_norm)
		labels = np.reshape(labels,(len(data), 1))
		norm = np.concatenate((features, labels), axis=1)

		return norm


	def createFoldsMetadataset(self, mode = "Random", k = 9, p_train = 0.8, seed=None):

		random.seed(seed)
		self.folds=[]


		indexes = list(range(self.size))
		random.shuffle(indexes)
		k_folds = [indexes[i::k] for i in range(k)]
		for k_test in range(k):
			fold = Fold()
			fold.trainIndexes = [y for x in k_folds[:k_test] for y in x] + [y for x in k_folds[k_test+1:] for y in x]
			fold.testIndexes = k_folds[k_test]
			print(fold.trainIndexes)
			print(fold.testIndexes)
			self.folds += [fold]


		return self.folds


	def showInfo(self):

		print("\n##  Metadataset  Train##")
		i = 0
		for all_data in self.data[:self.train_size]:
			print("\n---------------------------------------------------------------------------")
			print("Path:", self.dataset_names[i])
			print("Metafeatures:", all_data[:-self.n_labels])
			print("Scores:", self.accuracies[i])
			print("Winner:", self.winners[i])
			print("---------------------------------------------------------------------------")
			i += 1