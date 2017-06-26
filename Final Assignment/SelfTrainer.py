import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class SelfTrainer(BaseEstimator, ClassifierMixin):

	# Base estimator must support estimation of probabilities.
	def __init__(self, classifier=SVC(probability=True), treshold=0.7, max_iter=100):
		self.classifier = classifier
		self.treshold = treshold
		self.max_iter = max_iter

	# Get indices of the labelled and unlabelled sample points respectively.
	def _getInds(self, X,y):
		labIx = np.where(y!=-1)[0]
		unlabIx = np.where(y==-1)[0]
		return labIx, unlabIx

	def _getLabel(self, probabilities):
		if np.all(probabilities<self.treshold):
			return -1
		return self.classifier.classes_[np.where(probabilities>self.treshold)]

	def fit(self,X,y):
		y_current = copy(y)
		for i in range(0,self.max_iter):
			print(i)
			labIx, unlabIx = self._getInds(X,y_current)
			self.classifier.fit(X[labIx,:],y_current[labIx])
			if len(unlabIx)==0:
				break
			P = self.classifier.predict_proba(X[unlabIx,:])
			y_current[unlabIx] = np.fromiter([self._getLabel(P[i,:]) for i in range(0,P.shape[1])], P.dtype)
		return self

	def predict(self, X):
		return self.classifier.predict(X)

	def score(self, X, y):
		return self.classifier.score(X,y)

	def predict_proba(self, X):
		return self.classifier.predict_proba(X)

	def predict_log_proba(self, X):
		return self.classifier.predict_log_proba(X)