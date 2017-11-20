import numpy as np
import sys
import matplotlib.pyplot as pyplot
from utils import loadMat
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy import spatial
from functools import partial
import operator
import matplotlib.pyplot as plt

class KMeans:

	def __init__(self,file_name):

		self.data = loadMat(file_name)
		self.it = 0

	def initCentroid(self,k):
		self.centroidHash = defaultdict(list)
		self.masterCentroidMean = defaultdict(list)
		self.localCentroidMean= defaultdict(list)
	
	def initKPlusPlus(self,k):

		i = 1
		while i <k:
			self.centroidMat = np.asarray(self.masterCentroidMean.values())
			distance,index = spatial.KDTree(self.centroidMat).query(self.data['data'])
			maxIndex, maxValue = max(enumerate(distance), key=operator.itemgetter(1))
			self.masterCentroidMean[i] = self.data['data'][maxIndex]
			i+=1


		# print self.masterCentroidMean

	
	def assignCluster(self,k):

		self.centroidMat = np.asarray(self.masterCentroidMean.values())
		distance,index = spatial.KDTree(self.centroidMat).query(self.data['data'])
		for i in xrange(k):
			self.centroidHash = defaultdict(list)
		for i in xrange(len(index)):
			self.centroidHash[index[i]].append(self.data['data'][i])

	def calNewCentroid(self,k):
		for i in xrange(k):
			self.centroidHash[i] = np.array((self.centroidHash[i]))
			self.localCentroidMean[i] = self.centroidHash[i].mean(axis=0)


	def checkConvergence(self):
		distance,index = spatial.KDTree(np.asarray(self.localCentroidMean.values())).query(np.asarray(self.masterCentroidMean.values()))

		if  not np.any(distance):
			self.masterCentroidMean = defaultdict(list)
			self.masterCentroidMean = self.localCentroidMean

			return 1
		else:	
			self.masterCentroidMean = defaultdict(list)
			self.masterCentroidMean = self.localCentroidMean
			self.localCentroidMean = defaultdict(list)
			return 0

	def calObjective(self,k):

		sumMat = 0
		for i in xrange(k):
			sumMat+= np.sum(cdist([self.masterCentroidMean[i]],self.centroidHash[i])**2)
		return sumMat


	def kmeans(self,k):

		self.centroid = self.initCentroid(k)
		for i in xrange(k):
			self.masterCentroidMean[i] = self.data['data'][np.random.choice(len(self.data['data']))]
		self.assignCluster(k)
		self.calNewCentroid(k)
		while not self.checkConvergence():
			self.it += 1
			self.assignCluster(k)
			self.calNewCentroid(k)

		return self.calObjective(k)

	def kmeansPlusPlus(self,k):

		self.centroid = self.initCentroid(k)
		# initial cluster far!!
		self.masterCentroidMean[0] = self.data['data'][np.random.choice(len(self.data['data']))]

		self.initKPlusPlus(k)
		# for i in xrange(k):
		self.assignCluster(k)
		self.calNewCentroid(k)
		while not self.checkConvergence():
			self.it += 1
			self.assignCluster(k)
			self.calNewCentroid(k)

		return self.calObjective(k)

if __name__ == "__main__":
	if len(sys.argv)!=1:
		file_name = sys.argv[1]
	else:
		print "path incorrect"
		exit()
	error= []
	k = [2,3,4,5,6,7,8,9,10]
	objective_funct = []
	km = KMeans(file_name)
	for i in range(2,11):
		# objective_funct.append(km.kmeans(i))
		objective_funct.append(km.kmeansPlusPlus(i))


	plt.title("K Means ++ ")
	plt.xlabel('K')
	plt.ylabel('Objective function')
	plt.plot(k,objective_funct)
	plt.show()	
	# for i in range(2,11):
	# 	print km.kmeansPlusPlus(i)
		# km.load_data(txtFilePath)
		# obj_function=[]
		# clusters=[]
		# k=[2,3,4,5,6,7,8,9,10]
		# for i in k:
		# 	values = km.k_means(i)
		# 	obj_function.append(values)
		# plt.plot(k,obj_function,'b')
		# plt.xlabel('Number of Clusters (k)')
		# plt.ylabel('Objective function')
		# plt.title('K-means plot')
		# plt.legend(loc=4)
		# plt.show()
