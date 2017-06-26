import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from ClassifierWrapper import ClassifierWrapper

class SelfTrainer(ClassifierWrapper):

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

	# Given the predicted probability of each class, return the predicted label.
	def _getLabel(self, probabilities):
		if np.all(probabilities<self.treshold):
			return -1
		return self.classifier.classes_[np.where(probabilities=max(probabilities))]

	def fit(self,X,y):
		y_current = copy(y)
		unlabIx = np.array([])
		for i in range(0,self.max_iter):
			labIx, unlabIxNew = self._getInds(X,y_current)
			if np.all(unlabIxNew==unlabIx): # Break if nothing has changed in last iteration.
				break
			unlabIx = copy(unlabIxNew)
			print('number of unlabelled points: {}'.format(len(unlabIx)))
			self.classifier.fit(X[labIx,:],y_current[labIx])
			print(self.classifier.classes_)
			if len(unlabIx)==0:
				break
			print('Predicting unlabelled data')
			P = self.classifier.predict_proba(X[unlabIx,:])
			y_current[unlabIx] = np.fromiter([self._getLabel(P[i,:]) for i in range(0,P.shape[1])], int)
		return self