from collections import Counter
import re
import os
import commands
from glob import glob
import numpy as np
import pandas as pd



#Finding vocabulary
vocab = 1
n = []
n_legit = []
n_spam = []

Matrix = []

count_2d_mat = []
Label = []
#0 -- spam
#1 -- legit
for i in xrange(1,11):

	folderpaths = '../../Dataset/2_NaiveBayes/part'+str(i)+'/'
	filepaths = glob(os.path.join(folderpaths,'*.txt'))
	filepaths_legit = glob(os.path.join(folderpaths,'*legit*.txt'))
	filepaths_spam = glob(os.path.join(folderpaths,'*spmsg*.txt'))
	#print filepaths
	count_mat = []
	Label_folder = []
	for file in filepaths:

		words = re.findall(r'\w+', open(file).read().lower())
		count = Counter(words)
		del count['subject']
		count = {int(k):int(v) for k,v in count.items()}
		count_mat.append(count)
		if file in filepaths_legit:
			Label_folder.append(1)
		elif file in filepaths_spam:
			Label_folder.append(0)	

		key_max = max(count.keys())
		vocab = max(vocab,key_max)
	count_2d_mat.append(count_mat)
	my_df = pd.DataFrame(np.array(Label_folder).reshape(-1,1))
	my_df.to_csv('Label'+str(i)+'.csv', index=False, header=False)
	Label.append(Label_folder)
	print np.shape(Label_folder)
	n.append(len(filepaths))	
	n_legit.append(len(filepaths_legit))
	n_spam.append(len(filepaths_spam))	

print np.shape(Label)
N = sum(n)
N_legit = sum(n_legit)
N_spam = sum(n_spam)
print N_legit
#complete_mat = []
#Make Matrix1, 2, 3 ..10
for i in xrange(10):
	mat_folder = []
	for c in count_2d_mat[i]:
		#print "New row"
		row = []
		for v in xrange(1,vocab+1):
			#print v
			row.append(c.get(v,0))
		mat_folder.append(row)
		#complete_mat.append(row)
	print np.shape(mat_folder)
	mat_folder = np.array(mat_folder)
	my_df = pd.DataFrame(mat_folder)
	my_df.to_csv('Matrix'+str(i+1)+'.csv', index=False, header=False)
	Matrix.append(mat_folder)	

"""
Matrix = np.array(Matrix)
complete_mat = np.array(complete_mat)	
for i in xrange(10):
	Matrix[i] = Matrix[i][:,(complete_mat != 0).sum(axis=0) > 0]
	my_df = pd.DataFrame(Matrix[i])
	my_df.to_csv('Matrix'+str(i+1)+'.csv', index=False, header=False)
print np.shape(Matrix)	
print np.shape(Matrix[0])	
print np.shape(Label[0])	
"""