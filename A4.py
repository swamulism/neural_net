import numpy as np
import pandas as pd
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	return pd.read_csv("saeu4280-TrainingData.csv")

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

# part 2
def run_hopfield():
	hopfieldNet = HopfieldNetwork(25)
	hopfieldNet.fit(patterns)
	df = loadGeneratedData()
	for index, row in df.iterrows():
		print(hopfieldNet.classify(row[:-1]))

# part 3
def run_MLP():
	df = loadGeneratedData()
	nn = MLPClassifier()
	nn.fit(patterns, ["five", "two"])
	print(nn.predict(df.loc[:, df.columns != 'label']))

# part 4
def run_distorted(num_sample):
	nn = MLPClassifier()
	nn.fit(patterns, ["five", "two"])

	hopfieldNet = HopfieldNetwork(25)
	hopfieldNet.fit(patterns)

	y_true = ["five"] * num_sample + ["two"] * num_sample
	x = np.arange(0, 0.51, 0.01)
	y_mlp = []
	y_hopfield = []
	for i in x:
		distorted_values = [distort_input(five, i) for _ in range(num_sample)] + [distort_input(two, i) for _ in range(num_sample)]
		y_pred = nn.predict(distorted_values)

		y_hopfield.append(accuracy_score(y_true, [hopfieldNet.classify(val) for val in distorted_values]))
		y_mlp.append(accuracy_score(y_true, y_pred))


	fig, ax = plt.subplots(figsize=(12, 8))
	ax.plot(x, y_mlp, label="MLP")
	ax.plot(x, y_hopfield, label="hopfield")
	ax.set_title('Distortion vs Accuracy (1000 examples each)')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Percent Distortion')

	plt.legend()

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot1.png')

# part 5
def run_hidden_layers(num_sample):

	df1 = pd.read_csv("NewInput.csv")
	df2 = loadGeneratedData()

	df_comb = df1.append(df2)
	df_comb.loc[len(df_comb)] = five+["five"]
	df_comb.loc[len(df_comb)] = two+["two"]

	# for _, row in df.iterrows():
	# 	utils.visualize(row[:-1])


	

	y_true = ["five"] * num_sample + ["two"] * num_sample
	x = np.arange(0, 0.51, 0.01)
	
	fig, ax = plt.subplots(figsize=(12, 8))

	for num_layers in range(1, 22, 2):
		tmp = (100,) * num_layers
		nn = MLPClassifier(hidden_layer_sizes=tmp)
		nn.fit(df_comb.iloc[:,:-1], df_comb.iloc[:,-1])
		y_mlp = []
		for i in x:
			distorted_values = [distort_input(five, i) for _ in range(num_sample)] + [distort_input(two, i) for _ in range(num_sample)]
			y_pred = nn.predict(distorted_values)
			y_mlp.append(accuracy_score(y_true, y_pred))
		ax.plot(x, y_mlp, label=f"{num_layers} Layers")


	plt.legend()

	ax.set_title('Distortion vs MLP Accuracy (1000 examples each)')
	ax.set_ylabel('Accuracy')
	ax.set_xlabel('Percent Distortion')

	ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
	fig.savefig('plot2.png')

if __name__ == "__main__":

	print("Assignment 4")
	# part 1
	# print(loadGeneratedData())
	
	# part 2
	# run_hopfield()

	# part 3
	# run_MLP()
	
	# part 4
	# run_distorted(10)

	# part 5
	# run_hidden_layers(1000)

	