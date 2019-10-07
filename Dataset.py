import numpy as np
import random

class Fold:

	def __init__(self):
		self.trainIndexes=[]
		self.testIndexes=[]

#####################################################################################################

class Dataset:

	def __init__(self, filename, delim = ','):
		self.name = filename
		with open(filename,'r') as f:
			lines = f.readlines()
			
		self.nominalFeatures = []
		self.data = []
		self.dict = []
		self.size = int(lines[0])
		self.featureTypes = lines[2].rstrip().split(",")
		

		for item in self.featureTypes:
			if item=='Continuous':
				self.nominalFeatures.append(False)
			elif item=='Nominal':
				self.nominalFeatures.append(True)
			else:
				print("ERROR: Dataset",filename,"init FAILED")
				print("\tUnknown feature type")
				print(item)
				return
			

		for line in lines[3:]:
			samples = []
			for i in range(len(self.featureTypes)):
				samples += [line.rstrip().split(delim)[i]]
			self.data += [samples]

		sets=[]
		for i in range(len(self.featureTypes)):
			feature_set = set()
			for j in range(int(self.size)):
				feature_set.add(self.data[j][i])
			sets += [sorted(feature_set)]


		for i, featureSet in enumerate(sets):
			dictionary = {}
			if self.nominalFeatures[i] == True:
				for i, key in enumerate(featureSet):
					dictionary.update({key:i})
			self.dict += [dictionary]

		for elem in self.data:
			for i, dictionary in enumerate(self.dict):
				for item in dictionary.items():
					if item[0] == elem[i]:
						elem[i]=item[1]
				elem[i]=float(elem[i])

		self.rawData = self.data
		self.data = np.array(self.data)
		self.createFolds()


	def collectData(self,idx):
		return self.data[idx]

	def getXTest(self,fold = 0):
		return self.collectData(self.folds[fold].testIndexes)[:,:-1]

	def createFolds(self, mode = "Random", k = 5, p_train = 0.8, seed=None):

		random.seed(seed)
		self.folds=[]

		if mode == "Simple":

			fold = Fold()
			indexes = list(range(self.size))
			split_idx =  int(self.size*p_train)
			print("Simple Pivote:", split_idx)
			fold.trainIndexes = indexes[:split_idx]
			fold.testIndexes = indexes[split_idx:]
			self.folds = [fold]

		elif mode == "Random":

			fold = Fold()
			indexes = list(range(self.size))
			random.shuffle(indexes)
			split_idx =  int(self.size*p_train)
			fold.trainIndexes = indexes[:split_idx]
			fold.testIndexes = indexes[split_idx:]
			self.folds = [fold]

		elif mode == "Cross":

			indexes = list(range(self.size))
			random.shuffle(indexes)
			k_folds = [indexes[i::k] for i in range(k)]
			for k_test in range(k):
				fold = Fold()
				fold.trainIndexes = [y for x in k_folds for y in x]
				fold.testIndexes = [k_folds[k_test]]
				self.folds += [fold]

		elif mode == "Boostrap":

			fold = Fold()
			cont = self.size
			while cont > 0:
				fold.trainIndexes = fold.trainIndexes + [random.randint(0, self.size - 1)]
				cont -= 1
			fold.trainIndexes = list(set(fold.trainIndexes))
			random.shuffle(fold.trainIndexes)

			for i in range(self.size):
				if i not in fold.trainIndexes:
					fold.testIndexes=fold.testIndexes + [i]
			
			random.shuffle(fold.testIndexes)

			self.folds += [fold]
		else:

			print("ERROR: Create_fold FAILED")
			print("\tMode:", mode, "doesnt exist")

		return self.folds