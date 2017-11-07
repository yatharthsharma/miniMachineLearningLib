import pandas as pd
import matplotlib.pyplot as plt
def loadData(file_name,col=[],sep=","):
	if col!=[]:
		df = pd.read_csv(file_name,names = col,sep=sep)
	else:
		df = pd.read_csv(file_name,sep=sep)

	return df


csv_mat = 'word_mat.csv'

featVec = loadData(csv_mat)

def pl():
	#log acc after running the code
	log = [0.848476750401,0.869136540185,0.872683596431,0.901599612215,0.908293460925,0.941031941032,0.969773299748]
	#Naive bayes acc after running the code

	nb = [0.842091152815,0.846246973366,0.875990354805,0.874027237354,0.900787401575,0.873274780427,0.8875]

	k = [0.1, 0.2, 0.3, 0.5, 0.7,0.8,0.9 ]


	plt.title("log Reg vs Naive Bayes")
	plt.xlabel('training Set')
	plt.ylabel('Accuracy')
	plt.plot(k,log)
	plt.plot(k,nb)
	plt.legend(['log Reg','naive bayes'] ,loc='upper left')
	plt.show()		
