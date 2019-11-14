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



class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(p):
		#Update the hopfield matrix using the passed pattern
		print("TODO")
		utils.raiseNotDefined()

	def fit(patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		print("TODO")
		utils.raiseNotDefined()

	def retrieve(input):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.



		print("TODO")
		utils.raiseNotDefined()

	def classify(inputPattern):
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
