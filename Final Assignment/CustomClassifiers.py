import numpy as np
import utils as u
import math as m
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import warnings

""" 
TODO: 
- Add outlier detection and fit on the inliers only! (Does this improve performance?)
- Can be done using trees or SVM with probability estimates (set treshhold)
""" 


# Split problem into two problems based on activity type using classifier1 and predict these two problems using classifier2 and classifier3
# The classifiers must support weighted input.
class CC1(BaseEstimator, ClassifierMixin): # HERNOEMEN NAAR SplittingClassifier???
	def __str__(self):
		return "CC1"

	def __init__(self,C1=SVC(),C2=SVC(),C3=SVC(), classes1=[1,2,3]):
		self.C1 = C1 	# binary classifier
		self.C2 = C2	
		self.C3 = C3
		self.classes1 = classes1


	def _tryFitWithWeights(self, classifier, X, Y, sample_weight):
		try:
			classifier.fit(X,Y,sample_weight=sample_weight)
		except TypeError:
			warnings.warn('Sample weights are not supported, continuing without sample weights.')
			classifier.fit(X,Y)

	def fit(self,X,y, sample_weight=None):
		self.classes_ = np.unique(y)
		if sample_weight is None:
			sample_weight = np.ones(X.shape[0])
		Yact = u.mapIsIn(y,self.classes1)
		self._tryFitWithWeights(self.C1, X,Yact, sample_weight)
		X1, X2, y1, y2, w1, w2 = u.splitData(self.classes1, X, y, w=sample_weight)
		self._tryFitWithWeights(self.C2, X1, y1, w1)
		self._tryFitWithWeights(self.C3, X2, y2, w2)
		return self

	def predict(self,X):
		pred1 = self.C1.predict(X)
		ix1,ix2 = u.getSplit(pred1,[1])

		# Check if both are non-empty
		if len(ix2)==0:
			y = self.C2.predict(X)
		elif len(ix1)==0:
			y = self.C3.predict(X)
		else:
			X1, X2 = u.splitByInds((ix1,ix2),X)
			y = np.ones(X.shape[0], dtype=int)
			y[ix1] = self.C2.predict(X1)
			y[ix2] = self.C3.predict(X2)
		return y

	def predict_proba(self, X):
		P = np.zeros((X.shape[0], len(self.classes_)))
		pred1 = self.C1.predict(X)
		ix1,ix2 = u.getSplit(pred1,[1])
		classesIx1, classesIx2 = u.getSplit(self.classes_, self.classes1)

		# Check if both are non-empty
		if len(ix2)==0:
			P[np.ix_(ix1,classesIx1)] = self.C2.predict(X)
		elif len(ix1)==0:
			P[np.ix_(ix2,classesIx2)] = self.C3.predict_proba(X)
		else:
			X1, X2 = u.splitByInds((ix1,ix2),X)
			P[np.ix_(ix1,classesIx1)] = self.C2.predict_proba(X1)
			P[np.ix_(ix2,classesIx2)] = self.C3.predict_proba(X2)
		return P

	def score(self,X,y):
		predicted = self.predict(X)
		return np.mean([1 if predicted[i]==y[i] else 0 for i in range(0,len(y))])


# predict activity type by preClassifier, add as feature and then predict using mainClassifier.
class CC2(BaseEstimator, ClassifierMixin):
	def __str__(self):
		return "CC2"

	def __init__(self,preCL=SVC(),mainCL=SVC(),classes1=[1,2,3]):
		self.preCL = preCL
		self.mainCL = mainCL
		self.classes1 = classes1

	def _tryFitWithWeights(self, classifier, X, Y, sample_weight):
		try:
			classifier.fit(X,Y,sample_weight=sample_weight)
		except TypeError:
			warnings.warn('Sample weights are not supported, continuing without sample weights.')
			classifier.fit(X,Y)

	def fit(self, X, y, sample_weight=None):
		self.classes_ = np.unique(y)
		Yact = u.mapIsIn(y,self.classes1)
		self._tryFitWithWeights(self.preCL, X, Yact, sample_weight)
		X2 = np.concatenate((X,Yact.reshape(len(Yact),1)),axis=1) 
		self._tryFitWithWeights(self.mainCL, X2, y, sample_weight)	
		return self

	def predict(self, X):
		pred1 = self.preCL.predict(X)
		X2 = np.concatenate((X,pred1.reshape(len(pred1),1)),axis=1)
		return self.mainCL.predict(X2)

	def score(self, X, y):
		predicted = self.predict(X)
		return np.mean([1 if predicted[i]==y[i] else 0 for i in range(0,len(y))])


# Predict separate classifiers using subsets of the training data. PERFORMS TERRIBLE (as expected)
class CC3(BaseEstimator, ClassifierMixin):

	def __init__(self, newClassifierInst):
		self.newClassifierInst = newClassifierInst

	def fit(self, X, y, S, sample_weight=None):
		identifiers = np.unique(S)
		self.classifiers_ = []
		for identifier in identifiers:
			ix = np.where(S==identifier)[0]
			C = self.newClassifierInst()
			C.fit(X[ix,:], y[ix], sample_weight=sample_weight)
			self.classifiers_.append(C)
		return self

	def predict(self, X, combiner='majority'):
		nInst, nClassifiers = X.shape[0], len(self.classifiers_)
		predictions = np.zeros((nInst, nClassifiers), dtype=int)
		for i,classifier in enumerate(self.classifiers_):
			predictions[:,i] = classifier.predict(X)
		if combiner=='majority':
			return u.combinePredictions(predictions)
		else:
			raise('Combiner "{}" not recognized.'.format(combiner))


