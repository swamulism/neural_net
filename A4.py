import numpy as np
import pandas as pd
import utils
from sklearn.neural_network import MLPClassifier

#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	print("TODO")

def distort_input(instance, percent_distortion):
    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
	tmp = []
	for i in instance:
		if np.random.random() < percent_distortion:
			if i == 1:
				tmp.append(0)
			else:
				tmp.append(1)
		else:
			tmp.append(i)
	return tmp



class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		for row in range(len(self.h)):
			for col in range(len(self.h[row])):
				if row > col:
					self.h[row][col] += (2 * p[col] - 1) * (2 * p[row] - 1)
					self.h[col][row] = self.h[row][col]

	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		for p in patterns:
			self.addSinglePattern(p)

	def retrieve(self, input):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.
		vin = [0] * len(self.h)
		changed = True
		while changed:
			ls = list(range(len(self.h)))
			np.random.shuffle(ls)
			changed = False
			for x in ls:
				tmp = 0
				for i in range(len(self.h)):
					tmp += input[i] * self.h[i][x]
				if tmp >= 0:
					if vin[x] == 0:
						changed = True
					vin[x] = 1
				else:
					if vin[x] == 1:
						changed = True
					vin[x] = 0

		return vin

	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'
		tmp = self.retrieve(inputPattern)
		tmp_rev = [1 if x == 0 else 0 for x in tmp]
		if tmp == five or tmp_rev == five:
			return "five"
		elif tmp == two or tmp_rev == two:
			return "two"
		else:
			return "unknown"



if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	# utils.visualize(five)
	# utils.visualize(two)


	hopfieldNet.fit(patterns)

	df = pd.read_csv("saeu4280-TrainingData.csv")
	for index, row in df.iterrows():
		print(hopfieldNet.classify(row[:-1]))
