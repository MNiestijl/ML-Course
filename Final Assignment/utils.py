import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.exceptions import NotFittedError
from pathlib import Path
import math as m
import numpy.linalg as la
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
from functools import cmp_to_key

def getNumberOfDifferences(y1, y2):
	if len(y1)!=len(y2):
		print(len(y1), len(y2))
		raise('Vectors should be of the same length.')
	result = sum([ 1 if y1[i]!=y2[i] else 0 for i in range(0,len(y1))])
	print(result)
	return result

def getDifferentPredictions(predictions, minDiff=15):
	result = []
	for i in range(0, predictions.shape[1]):
		if any([getNumberOfDifferences(prediction, predictions[:,i])<minDiff for prediction in result]):
			continue
		else:
			result.append(predictions[:,i])
	return np.array(result).T


def loadSubmission(name):
	path = 'submissions/' + name + '.csv'
	return pd.read_csv(path, sep=',',header=0).as_matrix()[:,1]

def reorder(predictions, ordering):
	return predictions[:,np.argsort(ordering)]

def getPredictions(names):
	predictions = []
	for name in names:
		predictions.append(loadSubmission(name))
	return np.array(predictions).T

# Combine opinions by majority voting. 
def combineOpinions(opinions):
	occurances = list(zip(*np.unique(opinions, return_counts=True)))
	return max(occurances, key=lambda tup: tup[1])[0]

# Combine predictions of all instances using majority voting.
# prediction of shape (nInstances × nPredictions)
def combinePredictions(predictions):
	return np.array([ combineOpinions(predictions[i,:]) for i in range(0,predictions.shape[0]) ])

def combineSubmissions(names, ordering=None, minDiff=15):
	predictions = getPredictions(names)
	predictions = reorder(predictions, ordering)
	predictions = getDifferentPredictions(predictions,minDiff=minDiff)
	return combinePredictions(predictions)

def getData():
	Xtrn = loadmat('Xtrn.mat')['Xtrn']
	Ytrn = np.array(loadmat('Ytrn.mat')['Ytrn'].ravel(), dtype=int)
	Strn = loadmat('Strn.mat')['Strn']
	Xtst = loadmat('Xtst.mat')['Xtst']
	# Add unlabelled data:
	Xall = np.concatenate((Xtrn,Xtst),axis=0)
	Yall = np.array(np.concatenate((Ytrn,-np.ones(Xtst.shape[0]))), dtype=int)
	return Xtrn, Ytrn, Strn, Xtst, Xall, Yall

def mapIsIn(y,labs, yes=1, no=0):
	return np.array([yes if yi in labs else no for yi in y])

def getInds(y, labs):
	return np.array([ i for i, yi in enumerate(y) if yi in labs ])

# Split data based on labs. 
def getSplit(y,labs):
	complement = list(set(y)-set(labs))
	return getInds(y,labs), getInds(y,complement)


def splitData(labs, X, y, S=None, w=None):
	split = getSplit(y,labs)
	return splitByInds(split, X, y, S=S, w=w)

def splitByInds(inds, X=None,y=None, S=None, w=None):
	result = []
	if X is not None:
		result = result + [X[ix,:] for ix in inds]
	if y is not None:
		result = result + [y[ix] for ix in inds]
	if S is not None:
		result = result + [S[ix,:] for ix in inds]
	if w is not None:
		result = result + [w[ix] for ix in inds]
	return result

def getPersonInds(S, person):
	return np.where(S==person)[0]

def getLabelInds(y,label):
	return np.where(y==label)[0]

def getPersonLabelInds(S,y, person, label):
	ixP = set(getPersonInds(S,person))
	ixL = set(getLabelInds(y,label))
	return list(ixP.intersection(ixL))

def getCovariance(X,y,S,person,label):
	inds = getPersonLabelInds(S,y,person,label)
	Xpl, ypl = X[inds,:], y[inds]
	lda = LDA(store_covariance=True).fit(Xpl,ypl)
	return lda.covariance_

def getWeights(X, y, S, k=1):
	weights = np.ones(X.shape[0])
	persons, labels = np.unique(S), np.unique(y)
	V = np.zeros((len(persons), len(labels)))
	for i,p in enumerate(persons):
		for j,l in enumerate(labels):
			 cov = getCovariance(X,y,S,p,l)
			 V[i,j] = np.trace(cov)**k
	Vpsum = V.sum(axis=0)
	for i,p in enumerate(persons):
		for j,l in enumerate(labels):
			inds = getPersonLabelInds(S,y,p,l)
			weights[inds] = (Vpsum[j]/V[i,j])**(1/k)
	return weights/weights.mean()

def getWeights2(X, y, S, getNewClassifierInst, measure='mean'):
	identifiers = np.unique(S)
	weights = np.ones(X.shape[0])
	for i, identifier in enumerate(identifiers):
		ixTrn = np.where(S==identifier)[0]
		ixTst = np.where(S!=identifier)[0]
		C = getNewClassifierInst()
		C.fit(X[ixTrn,:], y[ixTrn])
		if measure=='mean':
			weights[ixTrn] = C.score(X[ixTst,:], y[ixTst])
		elif measure=='min':
			inds = lambda s: np.where(S==s)[0]
			weights[ixTrn] = min([C.score(X[inds(s),:], y[inds(s)]) for s in identifiers])
	return weights#/weights.mean()

def savePrediction(y, name, override=False):
	path = getPath(name)
	checkPath(path, override)
	if y.dtype!='int32' and y.dtype!='int64':
		raise Exception('Classifier should predict labels in type int.')

	# Write file
	with open(path, "w") as outfile:
		outfile.write("Id,Class\n")
		for e, lab in enumerate(list(y)):
			outfile.write("%s,%s\n" % (e+1,lab)) 

def checkPath(path, override=False):
	if Path(path).exists() and override==False:
		raise Exception('The following path is already occupied:\n{}'.format(path))

def getPath(name):
	return 'submissions/' + name + '.csv'

def makeSubmissionFile(Xtrn, Ytrn, Xtst, getClassifierInst, name="testSubmission", override=False, sample_weight=None, repeats=1):
	if not isinstance(getClassifierInst(), BaseEstimator):
		raise("Classifier is not an instance of scikit's BaseEstimator Class.")

	checkPath(getPath(name), override)

	labIx, unlabIx = getSplit(Ytrn, list(set(np.unique(Ytrn))-{-1}))
	predictions = np.zeros((Xtst.shape[0], repeats), dtype=int)
	for i in range(0,repeats):
		classifier = getClassifierInst()
		# Fit classifier
		try:
			classifier.fit(Xtrn,Ytrn, sample_weight=sample_weight)
		except TypeError:
			warnings.warn('\nMaking Submission: Sample weights are not supported, continuing without sample weights.\n')
			classifier.fit(Xtrn,Ytrn)

		# Score classifier on labelled part of training data, for referrence	
		scores = classifier.score(Xtrn[labIx,:],Ytrn[labIx])
		print('\nScore on Training set: {} +/- {}'.format(scores.mean(), scores.std()))

		# Get prediction
		try:
			predictions[:,i] = classifier.predict(Xtst)
		except NotFittedError as e:
			print(repr(e))
	y = combinePredictions(predictions)
	savePrediction(y, name, override)
