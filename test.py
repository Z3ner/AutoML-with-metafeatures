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


data_files_train = os.listdir("data/train")
list_path_train = [os.path.join("data", os.path.join("train", file)) for file in data_files_train]

data_files_test = os.listdir("data/test")
list_path_test = [os.path.join("data", os.path.join("test", file)) for file in data_files_test]

classifiers = [LogisticRegression2Classifier, SVMClassifier, MLPClassifier]
metadataset = Metadataset(train_data = list_path_train, classifiers = classifiers)
metadataset.showInfo()

kernel1 = DotProduct() + WhiteKernel()


final_classifier = AutoClassifier(kernel = kernel1)
acc, _ = final_classifier.score(metadataset)
print("Final Accuracy =", acc)

"""
for i in range(len(metadataset.folds)):
	test_data=metadataset.collectData(metadataset.folds[i].testIndexes)
	logits = final_classifier.predict(metadataset.getXTest(i))
	print("Labels:", test_data[:,-1])
	print("Logits:", logits)
"""