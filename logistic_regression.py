from utils import loadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class logReg():

	def __init__(self,featVec,label,k):
		
		self.acc = []
		self.pred = []
		self.featVec = featVec

		self.label = label['label_in_data']
		self.featVec.insert(0,"w0",np.ones((self.featVec.shape[0],1)))
		self.w = np.zeros((self.featVec.shape[1],1))
		self.df  = self.featVec


		self.df = pd.concat([self.featVec,  self.label],axis =1)
# 
		for i in range(len(k)):
			self.splitTrainTest(k[i])

			self.trainLogReg()
			self.testLogReg()
			self.acc.append(self.getAcc())

		plt.title("log Reg")
		plt.xlabel('training Set')
		plt.ylabel('Accuracy')
		plt.plot(k,self.acc)
		plt.show()		



	def splitTrainTest(self,k):

		self.df['is_train']  = np.random.uniform(0,1,len(self.df)) <= k

		self.train,self.test = self.df[self.df['is_train']==True], self.df[self.df['is_train']==False]
		df = self.df.drop('is_train', axis=1)

		train = self.train.iloc[:,0:self.train.shape[1]-2]
		label_train = self.train.iloc[:,self.train.shape[1]-2:self.train.shape[1]-1].as_matrix()

		self.train = train
		self.label_train=label_train
	
		test = self.test.iloc[:,0:self.test.shape[1]-2]
		label_test = self.test.iloc[:,self.test.shape[1]-2:self.test.shape[1]-1].as_matrix()

		self.test = test
		self.label_test=label_test



	def trainLogReg(self):
		iterations = 100
		lr = 1e-1

		for i in xrange(iterations):

			weightMul = np.dot(self.train,self.w)
			
			out = self.sigmoid(weightMul)
			l = self.label_train.reshape(self.label_train.shape[0],1)
			gradient =   l - out
			final_grad = np.dot(self.train.T,gradient)*lr
			self.w = self.w + final_grad



	def testLogReg(self):

		pred = np.dot(self.test,self.w)
		out = self.sigmoid(pred)

		for pred in out:
			if pred >= 0.5:
				self.pred.append(1)
			else:
				self.pred.append(0)

	def sigmoid(self, x):
		
		return 1. / (1. + np.exp(-x))

	def getAcc(self):

		acc = [a_i - b_i for a_i, b_i in zip(self.pred, self.label_test)]
		
		count = 0;

		for lab in acc:
			if lab == 0:
				count = count +1

		acc = count/float(len(self.label_test))
		print acc
		self.pred =[]
		return acc






if __name__ == "__main__":
	k = [0.1, 0.2, 0.3, 0.5, 0.7,0.8,0.9 ]

	file_name = 'farm-ads.txt'
	csv_mat_w_word_small = 'word_mat_with_word_small.csv'
	csv_mat = 'word_mat_with_word.csv'
	columns = []
	file_name_label = 'farm-ads-label.txt'
	featVec = loadData(csv_mat)
	columns = ["ad_in_data", "label_in_data"]
	label = loadData(file_name_label,columns,sep=" ")


	lr = logReg(featVec,label,k)
