import numpy as np
import utils


#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	print("TODO")

def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
	utils.raiseNotDefined()
    print("TODO")


#  size of our weight matrix should be 25x25 and not 5x5??????
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



		print("TODO")
		utils.raiseNotDefined()

	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'

		print("TODO")
		utils.raiseNotDefined()




if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	utils.visualize(five)
	utils.visualize(one)


	#hopfieldNet.fit(patterns)
