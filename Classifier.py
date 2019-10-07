from abc import ABCMeta,abstractmethod
import numpy as np
import math


from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier as SGDClf
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as MLPClf

class Classifier:
  
	__metaclass__ = ABCMeta

	@abstractmethod
	def train(self, dataset, fold = 0):
		pass


	@abstractmethod
	def predict(self, x):
		pass


	def error(self,data,logits):   
		return sum(map(lambda x, y: 0 if x == y else 1, data[:,-1], logits))/float(len(data[:,-1]))


	def score(self, dataset):
		errors = []
		if len(dataset.folds) == 1:
			self.train(dataset)
			test_data=dataset.collectData(dataset.folds[0].testIndexes)
			logits = self.predict(dataset.getXTest())
			acc = (1 - self.error(test_data, logits))*100
			return acc, 0
		else:
			for fold in range(len(dataset.folds)):
				self.train(dataset, fold = fold)
				test_data=dataset.collectData(dataset.folds[fold].testIndexes)
				logits = self.predict(dataset.getXTest(fold = fold))
				errors += [self.error(test_data, logits)]

			errors = np.array(errors)
			acc = (1 - errors.mean())*100
			std = errors.std()*100
			return acc, std

	
       
  
##############################################################################

class NaiveBayesClassifier(Classifier):

	def __init__(self, laplace=True):
		self.tV = []
		self.tC = {}
		self.laplace = laplace

	def train(self, dataset, fold = 0):
		self.tC = {}
		self.tV = []
		self.dict = dataset.dict
		self.nominalFeatures = dataset.nominalFeatures

		dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)

		n_labels = len(self.dict[-1])
		i = 0
		num_rows = dataTrain.shape[0]
		if num_rows == 0:
			num_rows = 0.0001
		for k in self.dict[-1].keys():
			value = self.dict[-1][k]
			self.tC[k] = dataTrain[np.ix_(dataTrain[:,-1] == value, (0,))].shape[0]/(num_rows +0.0)

		while i < len(self.dict)-1 :

			if self.nominalFeatures[i]:
				a = np.zeros((len(self.dict[i]), n_labels))
				for row in dataTrain:
					a[int(row[i]), int(row[-1])] += 1
				if self.laplace and np.any(a==0):
					a+=1

			else:
				a = np.zeros((2, n_labels))
				for k in self.dict[-1].keys():
					a[0, int(self.dict[-1][k])] = np.mean(dataTrain[np.ix_(dataTrain[:, -1] == self.dict[-1][k], (i, ))])
					a[1, int(self.dict[-1][k])] = np.var(dataTrain[np.ix_(dataTrain[:, -1] == self.dict[-1][k], (i ,))])

			self.tV.append(a)
			i += 1


			a = np.zeros((len(self.dict[i]), n_labels))
			
	def predict(self, x):

		logits = []

		for row in x:
			ppost = {}
			for k in self.dict[-1].keys():
				v = self.dict[-1][k]
				a = 1
				i = 0
				while i < (len(row)):
					if self.nominalFeatures[i]:
						a *= (self.tV[i][int(row[i]), v] / sum(self.tV[i][:, v]))
					else:
						exp = math.exp(-(((row[i]-self.tV[i][0,v])**2)/(2.0*self.tV[i][1,v])))
						sqrt = math.sqrt(2*math.pi*self.tV[i][1,v])
						a *= (exp/sqrt)

					i += 1
				a = a*self.tC[k]
				ppost[k] = a

			logits += [self.dict[-1][max(ppost,key=ppost.get)]]

		return np.array(logits)

    
    
class KNNClassifier(Classifier):

	def __init__(self, k = 3 ,norm = True):
		self.norm = norm
		self.mean = []
		self.std = []
		self.k = k

	def train(self, dataset, fold = 0):
		
		if self.norm:
			self.dataTrain = self.normalization(dataset.collectData(dataset.folds[fold].trainIndexes))
		else:
			self.dataTrain = dataset.collectData(dataset.folds[fold].trainIndexes)
		return

	def predict(self, x):

		zeros = np.reshape(np.zeros(len(x)), (len(x), 1))
		x_data = np.concatenate([x, zeros], axis = 1)

		if self.norm:
			x_data = self.normalization(x_data)
		else:
			x_data = x
			
		res = []

		for row_x in x_data:
			list_error = []
			losses = []
			for rows_train in self.dataTrain:
				s_mse = 0
				for idx in range(len(rows_train) - 1):
					s_mse += (rows_train[idx] - row_x[idx])**2
					
				loss = math.sqrt(s_mse)
				list_error += [loss]
				losses += [(loss, rows_train[-1])]

			labels = np.array([])

			for _ in range(self.k):
				minimal = min(list_error)
				for i, error in enumerate(list_error):
					if error == minimal:
						pos=i
					
				error, k = losses[pos]

				losses.pop(pos)
				list_error.pop(pos)
				labels = np.append(labels, k)

			freqs = np.unique(labels, return_counts=True)

			for i, counts in enumerate(freqs[1]):
				if counts == np.max(freqs[1]):
					pos = i

			res += [freqs[0][pos]]
		return np.array(res)


	def normalization(self, data):

		features = np.transpose(data)[:-1]
		labels = np.transpose(data)[-1]

		for feature in features:
			self.mean += [np.mean(feature)]
			self.std += [np.std(feature)]

		feature_norm = []
		idx = 0

		for feature in features:
			for x in feature:
				feature_norm += [(x - self.mean[idx])/self.std[idx]]
			idx += 1

		feature_norm = np.array(feature_norm)
		feature_norm = np.reshape(feature_norm, (len(features),len(data)))

		features = np.transpose(feature_norm)
		labels = np.reshape(labels,(len(data), 1))
		norm = np.concatenate((features, labels), axis=1)

		return norm




class LogisticRegressionClassifier(Classifier):

	def __init__(self, c=1,epochs=100):
		self.epochs = epochs
		self.c = c
		self.w=[]

	def train(self, dataset, fold = 0):
 
		if len(self.w)!= len(dataset.dict):
			self. w = np.random.uniform(low=-0.5,high=0.5, size=(1,len(dataset.dict)))
		for i in range(self.epochs):
			for row in dataset.collectData(dataset.folds[fold].trainIndexes):
				aux = np.append([1],row[:-1])
				self.w = self.w - (self.c*(self.sigmoid(np.dot(self.w,aux))-row[-1]))*aux
			i=i+1

	def predict(self, x):

		logits = []
		for row in x:
			aux = np.append([1], row)
			logits.append(1 if self.sigmoid(np.dot(self.w,aux)) >= 0.5 else 0 )
		return np.array(logits)



	def sigmoid(self,p):
		try:
			aux=1.0/(1+math.exp(-p))
		except OverflowError:
			aux= 0.0 
		return aux

class LogisticRegression2Classifier(Classifier):

	def __init__(self, random_state=0, solver='lbfgs'):
		
		self.clf = LogisticRegression(random_state=random_state, solver=solver)

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)



class RandomForestClassifier(Classifier):

	def __init__(self, n_estimators=10):
		self.clf = RFClassifier(n_estimators=10)

	def train(self, dataset, fold = 0):

		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)


class SVMClassifier(Classifier):

	def __init__(self, c=1,epochs=100):
		
		self.clf = svm.SVC(gamma='scale')

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)



class DecisionTreeClassifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = DTClassifier(random_state=random_state)

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

class PerceptronClassifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = Perceptron(random_state=random_state)

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

class SGDClassifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = SGDClf(random_state=random_state)

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

class GaussianNBClassifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = GaussianNB()

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

class KNN2Classifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = KNeighborsClassifier(n_neighbors=3)

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

class MLPClassifier(Classifier):

	def __init__(self, random_state=0):
		
		self.clf = MLPClf(hidden_layer_sizes=(64, 32, ))

	def train(self, dataset, fold = 0):
 
		train_data = dataset.collectData(dataset.folds[fold].trainIndexes)
		self.clf.fit(train_data[:,:-1], train_data[:,-1])

	def predict(self, x):

		return self.clf.predict(x)

		