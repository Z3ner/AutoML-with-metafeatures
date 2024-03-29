from Dataset import Dataset
from Classifier import NaiveBayesClassifier
from Classifier import KNNClassifier
from Classifier import LogisticRegressionClassifier
from Classifier import LogisticRegression2Classifier
from Classifier import SVMClassifier
from Classifier import DecisionTreeClassifier
from Classifier import PerceptronClassifier
from Classifier import SGDClassifier
from Classifier import GaussianNBClassifier
from Classifier import KNN2Classifier
from Classifier import MLPClassifier
from Metadataset import Metadataset
from AutoML import AutoClassifier
from AutoML import AutoRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import os


data_files_train = os.listdir("data/final")
list_path_train = [os.path.join("data", os.path.join("final", file)) for file in data_files_train]

data_files_test = os.listdir("data/test")
list_path_test = [os.path.join("data", os.path.join("test", file)) for file in data_files_test]

classifiers = [LogisticRegression2Classifier, KNN2Classifier, MLPClassifier]
metadataset = Metadataset(train_data = list_path_train, classifiers = classifiers)
metadataset.showInfo()

kernel1 = DotProduct() + WhiteKernel()


auto_regressor = AutoRegressor(kernel = kernel1)

for i in range(len(metadataset.folds)):
	auto_regressor = AutoRegressor(kernel=kernel1)
	auto_regressor.train(metadataset, i)
	test_data = metadataset.collectData(metadataset.folds[i].testIndexes)
	accuracies = auto_regressor.predict(metadataset.getXTest(i))
	print("Labels:", test_data[:,-1])
	print("Accuracies:", accuracies)