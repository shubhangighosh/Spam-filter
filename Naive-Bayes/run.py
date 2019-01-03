#Header Files
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

Matrix = []
Label = []
#Please run preprocess.py before running run.py to get frequency matrices
#Please ignore comments used for testing

#print cross validation score, PR curve

#Multinomial Naive Bayes
def MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel):
	#Training
	#Finding vocabulary and total no. of documents in training set
	vocab = np.shape(trainMatrix)[1]
	N = np.shape(trainMatrix)[0]
	#print N, vocab
	#No of spam and ham docs
	N_legit = trainLabel.sum(axis=0)[0]
	N_spam = N - N_legit
	#print N_legit, N_spam
	#Class priors for Spam and Ham
	prior = []
	prior.append(float(N_spam)/float(N)) #prior[0] -- prior_spam
	prior.append(float(N_legit)/float(N)) #prior[1] -- prior_legit
	#print prior
	#Splitting training set separately for Spam and Ham
	trainSpam = trainMatrix[trainLabel[:, 0] == 0, :]
	trainLegit = trainMatrix[trainLabel[:, 0] == 1, :]
	#print np.shape(trainSpam), np.shape(trainLegit)
	#Finding frequencies for every word in spam and ham
	tokenSpam = np.sum(trainSpam,axis=0)
	tokenLegit = np.sum(trainLegit,axis=0)
	#print np.shape(tokenLegit), np.shape(tokenSpam)
	token = []
	token.append(tokenSpam)
	token.append(tokenLegit)
	token = np.array(token)
	
	#print np.shape(token)
	#Finding conditional probability according to MLE for Multinomial
	condProb = token.astype(float)+1.0
	
	#print np.shape(condProb)
	condProbSum = np.sum(condProb,axis=1)
	condProb[0] = condProb[0]/condProbSum[0]
	condProb[1] = condProb[1]/condProbSum[1]
	condProb = condProb.T
	#print np.shape(condProb)
	#print condProb


	#apply -- test cases
	score = []
	score.append(np.log(prior[0]))
	score.append(np.log(prior[1]))
	score_all = np.dot(testMatrix,np.log(condProb))
	score_all[:,0] += score[0]
	score_all[:,1] += score[1]
	#print np.shape(score_all)
	#print score_all
	score_all = np.array(score_all)
	Y_pred = np.argmax(score_all, axis =1)
	Y_score = score_all[:,1]
	#print np.shape(Y_pred)
	target_names = ['Spam', 'Legit']
	#print(classification_report(testLabel[:,0], Y_pred, target_names=target_names))		
	return Y_pred, Y_score

def BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel):
	#Training
	#Finding vocabulary and total no. of documents in training set
	vocab = np.shape(trainMatrix)[1]
	N = np.shape(trainMatrix)[0]
	#print N, vocab
	#No of spam and ham docs
	N_legit = trainLabel.sum(axis=0)[0]
	N_spam = N - N_legit
	#print N_legit, N_spam
	#Class priors for Spam and Ham
	prior = []
	prior.append(float(N_spam)/float(N)) #prior[0] -- prior_spam
	prior.append(float(N_legit)/float(N)) #prior[1] -- prior_legit
	#print prior
	trainSpam = trainMatrix[trainLabel[:, 0] == 0, :]
	trainLegit = trainMatrix[trainLabel[:, 0] == 1, :]
	#Total no of spam and ham docs
	Nspam = (trainSpam != 0).sum(axis=0)
	Nlegit = (trainLegit != 0).sum(axis=0)

	Ndoc = []
	Ndoc.append(Nspam)
	Ndoc.append(Nlegit)
	#print np.shape(Ndoc)
	#print "NDOC!!!!"
	#print Ndoc
	#Finding Conditional Probability
	Ndoc = np.array(Ndoc)
	condProb = Ndoc.astype(float)+1.0
	condProb[0] = condProb[0]/(float(N_spam)+2.0)
	condProb[1] = condProb[1]/(float(N_legit)+2.0)
	condProb = condProb.T
	#print np.shape(condProb)
	#print condProb

	#apply
	score = []
	score.append(np.log(prior[0]))
	score.append(np.log(prior[1]))

	score_all = []

	for i in xrange(np.shape(testMatrix)[0]):
		row = []
		row0 = score[0]
		row1 = score[1]

		for j in xrange(np.shape(testMatrix)[1]):
			if(testMatrix[i][j]>0):
				row0 += np.log(condProb[j][0])
				row1 += np.log(condProb[j][1])
			elif(testMatrix[i][j]==0):	
				row0 += np.log(1-condProb[j][0])
				row1 += np.log(1-condProb[j][1])
		row.append(row0)
		row.append(row1)
		score_all.append(row)
	#print np.shape(score_all)
	#print score_all
	score_all = np.array(score_all)
	Y_pred = np.argmax(score_all, axis =1)
	Y_score = score_all[:,1]
	#print np.shape(Y_pred)
	target_names = ['Spam', 'Legit']
	#print(classification_report(testLabel[:,0], Y_pred, target_names=target_names))		
	return Y_pred, Y_score	


def DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha):
	#Training
	#Finding vocabulary and no of docs
	vocab = np.shape(trainMatrix)[1]
	N = np.shape(trainMatrix)[0]
	#print N, vocab
	#No.of Spam and Ham docs
	N_legit = trainLabel.sum(axis=0)[0]
	N_spam = N - N_legit
	#print N_legit, N_spam
	#Finding Class priors
	prior = []
	prior.append(float(N_spam)/float(N)) #prior[0] -- prior_spam
	prior.append(float(N_legit)/float(N)) #prior[1] -- prior_legit
	#print prior
	#Separating TRaining Set into Spam 
	trainSpam = trainMatrix[trainLabel[:, 0] == 0, :]
	trainLegit = trainMatrix[trainLabel[:, 0] == 1, :]
	#print np.shape(trainSpam), np.shape(trainLegit)
	tokenSpam = np.sum(trainSpam,axis=0)
	tokenLegit = np.sum(trainLegit,axis=0)
	#print np.shape(tokenLegit), np.shape(tokenSpam)
	token = []
	token.append(tokenSpam)
	token.append(tokenLegit)
	token = np.array(token)
	condProb = token.astype(float)+alpha
	condProb = token.astype(float)+1.0
	condProbSum = np.sum(condProb,axis=1)
	condProb[0] = condProb[0]/condProbSum[0]
	condProb[1] = condProb[1]/condProbSum[1]
	condProb = condProb.T
	#print np.shape(condProb)
	#print condProb


	#apply
	score = []
	score.append(np.log(prior[0]))
	score.append(np.log(prior[1]))
	score_all = np.dot(testMatrix,np.log(condProb))
	score_all[:,0] += score[0]
	score_all[:,1] += score[1]
	#print np.shape(score_all)
	#print score_all
	score_all = np.array(score_all)
	Y_pred = np.argmax(score_all, axis =1)
	Y_score = score_all[:,1]
	#print np.shape(Y_pred)
	target_names = ['Spam', 'Legit']
	#print(classification_report(testLabel[:,0], Y_pred, target_names=target_names))		
	return Y_pred, Y_score

def BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha,beta):
	#Training
	vocab = np.shape(trainMatrix)[1]
	N = np.shape(trainMatrix)[0]
	#print N, vocab
	N_legit = trainLabel.sum(axis=0)[0]
	N_spam = N - N_legit
	#print N_legit, N_spam
	prior = []
	prior.append(float(N_spam)/float(N)) #prior[0] -- prior_spam
	prior.append(float(N_legit)/float(N)) #prior[1] -- prior_legit
	#print prior
	trainSpam = trainMatrix[trainLabel[:, 0] == 0, :]
	trainLegit = trainMatrix[trainLabel[:, 0] == 1, :]
	
	Nspam = (trainSpam != 0).sum(axis=0)
	Nlegit = (trainLegit != 0).sum(axis=0)

	Ndoc = []
	Ndoc.append(Nspam)
	Ndoc.append(Nlegit)
	#print np.shape(Ndoc)
	#print Ndoc
	Ndoc = np.array(Ndoc)
	condProbSpam = Ndoc[0].astype(float)+alpha
	condProbLegit = Ndoc[0].astype(float)+beta
	condProb = []
	condProb.append(condProbSpam)
	condProb.append(condProbLegit)
	condProb = np.array(condProb) + 1.0
	den = alpha + beta
	den1 = den + float(N_spam) + 2.0
	den2 = den + float(N_legit) + 2.0
	condProb[0] = condProb[0]/den1
	condProb[1] = condProb[1]/den2
	condProb = condProb.T
	#print np.shape(condProb)
	#print condProb

	#apply
	score = []
	score.append(np.log(prior[0]))
	score.append(np.log(prior[1]))

	score_all = []

	for i in xrange(np.shape(testMatrix)[0]):
		row = []
		row0 = score[0]
		row1 = score[1]

		for j in xrange(np.shape(testMatrix)[1]):
			if(testMatrix[i][j]>0):
				row0 += np.log(condProb[j][0])
				row1 += np.log(condProb[j][1])
			elif(testMatrix[i][j]==0):	
				row0 += np.log(1-condProb[j][0])
				row1 += np.log(1-condProb[j][1])
		row.append(row0)
		row.append(row1)
		score_all.append(row)
	#print np.shape(score_all)
	#print score_all
	score_all = np.array(score_all)
	Y_pred = np.argmax(score_all, axis =1)
	Y_score = score_all[:,1]
	#print np.shape(Y_pred)
	target_names = ['Spam', 'Legit']
	#print(classification_report(testLabel[:,0], Y_pred, target_names=target_names))		
	return Y_pred, Y_score


#For Bayesian Parameter Estimation have to find condProb only!!!!!

for i in xrange(10):
	df=pd.read_csv('Matrix'+str(i+1)+'.csv', sep=',',header=None)
	X = df.values
	X = np.array(X)
	Matrix.append(X)
	df=pd.read_csv('Label'+str(i+1)+'.csv', sep=',',header=None)
	Y = df.values
	Y = np.array(Y)
	Label.append(Y)
Matrix = np.array(Matrix)
Label = np.array(Label)	

print np.shape(Matrix), np.shape(Label)

newMatrix = np.concatenate(Matrix[0:],axis=0)
newLabel = np.concatenate(Label[0:],axis=0)
Spam = newMatrix[newLabel[:, 0] == 0, :]
Legit = newMatrix[newLabel[:, 0] == 1, :]
print np.shape(Spam), np.shape(Legit)

#Assuming all words are spammy
alpha1 = []
alphaRow1 =  np.ones(np.shape(newMatrix)[1])
alphaRow2 =  np.zeros(np.shape(newMatrix)[1])
alpha1.append(alphaRow1)
alpha1.append(alphaRow2)
a1 = 10.0*alphaRow1
b1 = 10.0*(alphaRow2)
alpha1 = np.array(alpha1)
alpha1 = 10.0*alpha1

#Assuming all words are hammy
alpha3 = []
alphaRow1 =  np.ones(np.shape(newMatrix)[1])
alphaRow2 =  np.zeros(np.shape(newMatrix)[1])
alpha3.append(alphaRow1)
alpha3.append(alphaRow2)
a3 = 10.0*alphaRow1
b3 = 10.0*(alphaRow2)
alpha3 = np.array(alpha3)
alpha3 = 10.0*alpha3
a2 = 5.0*alphaRow1
b2 = 5.0*alphaRow1


#estimation of alpha for dirichlet
alpha2 = []




freqSpam = np.sum(Spam,axis=0)
freqLegit = np.sum(Legit,axis=0)
freqSpam = np.array(freqSpam)
freqLegit = np.array(freqLegit)
alphaRow = freqSpam.astype(float)/(freqSpam.astype(float)+freqLegit.astype(float))

alpha2.append(alphaRow)
alpha2.append(1.0 - alphaRow)

alpha2 = np.array(alpha2)
alpha2 = 10.0*alpha2
#print "Alpha shape"
#print np.shape(alpha2)


trainMatrix = np.concatenate(Matrix[0:8],axis=0)
trainLabel = np.concatenate(Label[0:8],axis=0)
testMatrix = np.concatenate(Matrix[8:],axis=0)
testLabel = np.concatenate(Label[8:],axis=0)

#print np.shape(trainMatrix), np.shape(trainLabel), np.shape(testMatrix), np.shape(testLabel)
Y_pred_m5,Y_score_m5 = MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel) #make Y-pred1 etc for PR curve
Y_pred_b5,Y_score_b5 = BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel)
Y_pred_d15,Y_score_d15 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha1)
Y_pred_d25,Y_score_d25 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha2)
Y_pred_d35,Y_score_d35 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha3)
Y_pred_b15,Y_score_b15 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a1,b1)
Y_pred_b25,Y_score_b25 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a2,b2)
Y_pred_b35,Y_score_b35 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a3,b3)

print "DOne"

trainMatrix = np.concatenate(Matrix[2:],axis=0)
trainLabel = np.concatenate(Label[2:],axis=0)
testMatrix = np.concatenate(Matrix[0:2],axis=0)
testLabel = np.concatenate(Label[0:2],axis=0)

#print np.shape(trainMatrix), np.shape(trainLabel), np.shape(testMatrix), np.shape(testLabel)
Y_pred_m1,Y_score_m1 = MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel) #make Y-pred1 etc for PR curve
Y_pred_b1,Y_score_b1 = BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel)
Y_pred_d11,Y_score_d11 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha1)
Y_pred_d21,Y_score_d21 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha2)
Y_pred_d31,Y_score_d31 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha3)
Y_pred_b11,Y_score_b11 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a1,b1)
Y_pred_b21,Y_score_b21 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a2,b2)
Y_pred_b31,Y_score_b31 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a3,b3)

print "DOne"

trainMatrix1 = np.concatenate(Matrix[4:],axis=0)
trainMatrix2 = np.concatenate(Matrix[0:2],axis=0)
trainMatrix = np.concatenate((trainMatrix1,trainMatrix2),axis=0)
trainLabel1 = np.concatenate(Label[4:],axis=0)
trainLabel2 = np.concatenate(Label[0:2],axis=0)
trainLabel = np.concatenate((trainLabel1,trainLabel2),axis=0)
testMatrix = np.concatenate(Matrix[2:4],axis=0)
testLabel = np.concatenate(Label[2:4],axis=0)

#print np.shape(trainMatrix), np.shape(trainLabel), np.shape(testMatrix), np.shape(testLabel)
Y_pred_m2,Y_score_m2 = MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel) #make Y-pred1 etc for PR curve
Y_pred_b2, Y_score_b2 = BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel)
Y_pred_d12, Y_score_d12 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha1)
Y_pred_d22, Y_score_d22 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha2)
Y_pred_d32, Y_score_d32 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha3)
Y_pred_b12, Y_score_b12 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a1,b1)
Y_pred_b22, Y_score_b22 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a2,b2)
Y_pred_b32,Y_score_b32 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a3,b3)

print "DOne"

trainMatrix1 = np.concatenate(Matrix[6:],axis=0)
trainMatrix2 = np.concatenate(Matrix[0:4],axis=0)
trainMatrix = np.concatenate((trainMatrix1,trainMatrix2),axis=0)
trainLabel1 = np.concatenate(Label[6:],axis=0)
trainLabel2 = np.concatenate(Label[0:4],axis=0)
trainLabel = np.concatenate((trainLabel1,trainLabel2),axis=0)
testMatrix = np.concatenate(Matrix[4:6],axis=0)
testLabel = np.concatenate(Label[4:6],axis=0)

#print np.shape(trainMatrix), np.shape(trainLabel), np.shape(testMatrix), np.shape(testLabel)
Y_pred_m3, Y_score_m3 = MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel) #make Y-pred1 etc for PR curve
Y_pred_b3,Y_score_b3 = BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel)
Y_pred_d13,Y_score_d13 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha1)
Y_pred_d23,Y_score_d23 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha2)
Y_pred_d33,Y_score_d33 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha3)
Y_pred_b13,Y_score_b13 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a1,b1)
Y_pred_b23,Y_score_b23 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a2,b2)
Y_pred_b33,Y_score_b33 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a3,b3)


print "DOne"

trainMatrix1 = np.concatenate(Matrix[8:],axis=0)
trainMatrix2 = np.concatenate(Matrix[0:6],axis=0)
trainMatrix = np.concatenate((trainMatrix1,trainMatrix2),axis=0)
trainLabel1 = np.concatenate(Label[8:],axis=0)
trainLabel2 = np.concatenate(Label[0:6],axis=0)
trainLabel = np.concatenate((trainLabel1,trainLabel2),axis=0)
testMatrix = np.concatenate(Matrix[6:8],axis=0)
testLabel = np.concatenate(Label[6:8],axis=0)

#print np.shape(trainMatrix), np.shape(trainLabel), np.shape(testMatrix), np.shape(testLabel)

Y_pred_m4,Y_score_m4 = MultinomialNB(trainMatrix,trainLabel,testMatrix,testLabel) #make Y-pred1 etc for PR curve
Y_pred_b4,Y_score_b4 = BernoulliNB(trainMatrix,trainLabel,testMatrix,testLabel)
Y_pred_d14,Y_score_d14 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha1)
Y_pred_d24,Y_score_d24 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha2)
Y_pred_d34,Y_score_d34 = DirichletNB(trainMatrix,trainLabel,testMatrix,testLabel,alpha3)
Y_pred_b14,Y_score_b14 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a1,b1)
Y_pred_b24,Y_score_b24 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a2,b2)
Y_pred_b34,Y_score_b34 = BetaNB(trainMatrix,trainLabel,testMatrix,testLabel,a3,b3)



print "DOne"

Y_pred_m = np.concatenate((Y_pred_m1,Y_pred_m2,Y_pred_m3,Y_pred_m4,Y_pred_m5),axis=0)
Y_pred_b = np.concatenate((Y_pred_b1,Y_pred_b2,Y_pred_b3,Y_pred_b4,Y_pred_b5),axis=0)
Y_pred_d1 = np.concatenate((Y_pred_d11,Y_pred_d12,Y_pred_d13,Y_pred_d14,Y_pred_d15),axis=0)
Y_pred_d2 = np.concatenate((Y_pred_d21,Y_pred_d22,Y_pred_d23,Y_pred_d24,Y_pred_d25),axis=0)
Y_pred_d3 = np.concatenate((Y_pred_d31,Y_pred_d32,Y_pred_d33,Y_pred_d34,Y_pred_d35),axis=0)
Y_pred_b1 = np.concatenate((Y_pred_b11,Y_pred_b12,Y_pred_b13,Y_pred_b14,Y_pred_b15),axis=0)
Y_pred_b2 = np.concatenate((Y_pred_b21,Y_pred_b22,Y_pred_b23,Y_pred_b24,Y_pred_b25),axis=0)
Y_pred_b3 = np.concatenate((Y_pred_b31,Y_pred_b32,Y_pred_b33,Y_pred_b34,Y_pred_b35),axis=0)


Y_score_m = np.concatenate((Y_score_m1,Y_score_m2,Y_score_m3,Y_score_m4,Y_score_m5),axis=0)
Y_score_b = np.concatenate((Y_score_b1,Y_score_b2,Y_score_b3,Y_score_b4,Y_score_b5),axis=0)
Y_score_d1 = np.concatenate((Y_score_d11,Y_score_d12,Y_score_d13,Y_score_d14,Y_score_d15),axis=0)
Y_score_d2 = np.concatenate((Y_score_d21,Y_score_d22,Y_score_d23,Y_score_d24,Y_score_d25),axis=0)
Y_score_d3 = np.concatenate((Y_score_d31,Y_score_d32,Y_score_d33,Y_score_d34,Y_score_d35),axis=0)
Y_score_b1 = np.concatenate((Y_score_b11,Y_score_b12,Y_score_b13,Y_score_b14,Y_score_b15),axis=0)
Y_score_b2 = np.concatenate((Y_score_b21,Y_score_b22,Y_score_b23,Y_score_b24,Y_score_b25),axis=0)
Y_score_b3 = np.concatenate((Y_score_b31,Y_score_b32,Y_score_b33,Y_score_b34,Y_score_b35),axis=0)

print "DOne"
print "Classification Report for MultinomialNB"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_m, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_m)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve MultinomialNB')
plt.show()
plt.savefig('MultinomialNB.png',format='png')

print "Classification Report for BernoulliNB"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_b, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_b)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve BernoulliNB')
plt.show()
plt.savefig('BernoulliNB.png',format='png')

print "Classification Report for DirichletNB with Parameter 1"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_d1, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_d1)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve DirichletNB with Parameter 1')
plt.show()
plt.savefig('DirichletNB1.png',format='png')

print "Classification Report for DirichletNB with Parameter 2"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_d2, target_names=target_names))



precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_d2)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve DirichletNB with Parameter 2')
plt.show()
plt.savefig('DirichletNB2.png',format='png')


print "Classification Report for DirichletNB with Parameter 3"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_d3, target_names=target_names))



precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_d3)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve DirichletNB with Parameter 3')
plt.show()
plt.savefig('DirichletNB3.png',format='png')

print "Classification Report for BernoulliNB with Parameter 1"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_b1, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_b1)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision,alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve BetaNB with Parameter 1')
plt.show()
plt.savefig('BetaNB1.png',format='png')

print "Classification Report for BernoulliNB with Parameter 2"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_b2, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_b2)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve BetaNB with Parameter 2')
plt.show()
plt.savefig('BetaNB2.png',format='png')

print "Classification Report for BernoulliNB with Parameter 3"
target_names = ['Spam', 'Legit']
print(classification_report(newLabel[:,0], Y_pred_b3, target_names=target_names))

precision, recall, _ = precision_recall_curve(newLabel[:,0], Y_score_b3)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision,alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve BetaNB with Parameter 3')
plt.show()
plt.savefig('BetaNB3.png',format='png')




"""df=pd.read_csv('Matrix'+str(i+1)+'.csv', sep=',',header=None)
X = df.values
X = np.array(X)
"""