import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.exceptions import NotFittedError
from pathlib import Path
import math as m
import numpy.linalg as la

def loadSubmission(name):
	path = 'submissions/' + name + '.csv'
	return pd.read_csv(path, sep=',',header=0).as_matrix()[1,:]

def getData():
	Xtrn = loadmat('Xtrn.mat')['Xtrn']
	Ytrn = np.array(loadmat('Ytrn.mat')['Ytrn'].ravel(), dtype=int)
	Strn = loadmat('Strn.mat')['Strn']
	Xtst = loadmat('Xtst.mat')['Xtst']
	# Add unlabelled data:
	Xall = np.concatenate((Xtrn,Xtst),axis=0)
	Yall = np.array(np.concatenate((Ytrn,-np.ones(Xtst.shape[0]))), dtype=int)
	Xfull = np.concatenate((Xtrn,Strn),axis=1)
	return Xtrn, Ytrn, Strn, Xtst, Xall, Yall,Xfull

def getSplitData(split, X,y=None):
	result = [X[ix,:] for ix in split]
	if y is not None:
		result = result + [y[ix] for ix in split]
	return result


def getActLabels(y,labs):
	return np.array([1 if yi in labs else 0 for yi in y])

# Split data based on labs. 
def getSplit(X,y,labs):
	C1 = [ i for i,yi in enumerate(y) if yi in labs ]
	C2 = [ i for i,yi in enumerate(y) if yi not in labs ]
	return C1, C2#, X[C1,:], X[C2,:], y[C1],y[C2]

def getMean(X,y,S,label, person):
	ixP = set(np.where(S==person)[0])
	ixL = set(np.where(y==label)[0])
	ix = list(ixP.intersection(ixL))
	if len(ix)==0:
		return None
	return X[ix,:].mean(axis=0).reshape((1,X.shape[1]))

def getWeights(X, y, S, sigma=1):
	#VI = np.divide(1,X.std(axis=0)) # std^-1 for each feature
	def getWeight(i):
		mean = getMean(X,y,S,y[i],S[i])
		if mean is None:
			return 0
		gamma = X[i,:]-mean
		return m.exp(-la.norm(gamma)/(sigma**2))
		#return m.exp(-gamma.dot(np.multiply(VI,gamma).T))
	weights = np.array([ getWeight(i) for i in range(0,X.shape[0]) ])
	rescaled = weights/weights.mean()
	return rescaled


def getMeanData(X, y, S):
	averages = []
	ynew = []
	for person in np.unique(S):
		for label in np.unique(y):
			mean = getMean(X,y,S,label, person)
			if mean is not None:
				ynew.append(label)
				averages.append(mean)
	return np.vstack(averages), np.array(ynew)

def makeSubmissionFile(Xtrn, Ytrn, Xtest, Classifier, name="testSubmission", override=False):

	# Fit classifier.
	Classifier.fit(Xtrn,Ytrn)

	# Score classifier on labelled part of training data, for referrence
	labIx, unlabIx = getSplit(Xtrn,Ytrn, list(set(np.unique(Ytrn))-{-1}))	
	scores = Classifier.score(Xtrn[labIx,:],Ytrn[labIx])
	print('\nScore on Training set: {} +/- {}'.format(scores.mean(), scores.std()))

	# CHeck if the path is occupied.
	path = 'submissions/' + name + '.csv'
	my_file = Path(path)
	if Path(path).exists() and override==False:
		raise Exception('The following path is already occupied:\n{}'.format(path))

	# Get prediction
	try:
		y = Classifier.predict(Xtest)
	except NotFittedError as e:
		print(repr(e))

	# Make sure the type is ok.
	if y.dtype!='int32' and y.dtype!='int64':
		raise Exception('Classifier should predict labels in type int.')

	# Write file
	with open(path, "w") as outfile:
		outfile.write("Id,Class\n")
		for e, lab in enumerate(list(y)):
			outfile.write("%s,%s\n" % (e+1,lab)) 
