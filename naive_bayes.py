from utils import loadData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
class NaiveBayes():

	def __init__(self,featVec,label,k):
		self.acc =[]
		for i in range(len(k)):
			self.featVec = featVec
			self.label = label
			print len(self.featVec)
			self.pred = []
			self.df = pd.concat([self.featVec,  self.label],axis =1)
			self.splitTrainTest(k[i])
			self.getProb()
			self.acc.append(self.getAcc())

		plt.title("Naive bayes")
		plt.xlabel('training Set')
		plt.ylabel('Accuracy')
		plt.plot(k,self.acc)
		plt.show()		



	def splitTrainTest(self,k):

		self.df['is_train']  = np.random.uniform(0,1,len(self.df)) <= k

		self.train,self.test = self.df[self.df['is_train']==True], self.df[self.df['is_train']==False]

		self.cols = self.test.columns.get_values()
		bt = self.test.apply(lambda x: x > 0)
		self.test['cols']= bt.apply(lambda x: list(self.cols[x.values]), axis=1)

	def getProb(self):

		pos_test = self.train.query('label_in_data == 1')

		neg_test = self.train.query('label_in_data == 0')
	
		count_total = len(self.train) 

		for index, data in self.test.iterrows():

			prob_one =0;
			prob_zero =0;
			words = data['cols']
			for word in words:
				if word!='label_in_data' and  word!='ad_in_data':
					count_one = pos_test[word].sum()
					count_zero = neg_test[word].sum()

					if count_one == 0:
						num_one = float((count_one + 1))
						deno_one = float(len(self.cols) + len(pos_test))
					else:
						num_one = float((count_one))
						deno_one = float(len(pos_test))
				
					if count_zero == 0:
						num_zero =float((count_zero +1))
						deno_zero= float(len(self.cols) + len(neg_test))	
					else:
						num_zero = float((count_zero))
						deno_zero = float(len(neg_test))



					prob_one +=  math.log(num_one/ deno_one)
					prob_zero += math.log(num_zero/ deno_zero)
				else:
					continue
			if (prob_one+math.log(len(pos_test)/float(count_total)))>(prob_zero+math.log(len(neg_test)/float(count_total))):
				self.pred.append(1)
			else:
				self.pred.append(0)


	def getAcc(self):

		acc = [a_i - b_i for a_i, b_i in zip(self.pred, self.test['label_in_data'].as_matrix())]
		
		count = 0;
		for lab in acc:
			if lab == 0:
				count = count +1

		acc = count/float(len(self.test))
		print acc
		return acc
		self.pred = []
	

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

	lr = NaiveBayes(featVec,label,k)
