import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from ClassifierWrapper import ClassifierWrapper
import utils as u

class SelfTrainer(ClassifierWrapper):

	# Base estimator must support estimation of probabilities.
	def __init__(self, classifier=SVC(probability=True), treshold=0.7, max_iter=100, discount=1):
		self.classifier = classifier
		self.treshold = treshold
		self.max_iter = max_iter
		self.discount = discount

	# Given the predicted probability of each class, return the predicted label.
	def _getLabel(self, probabilities):
		if np.all(probabilities<self.treshold):
			return -1
		return self.classifier.classes_[np.where(probabilities==max(probabilities))]

	def fit(self,X,y):
		classes = list(set(np.unique(y)) - {-1})
		y_current = copy(y)
		weights = np.ones(len(y))
		unlabIx = np.array([])
		getInds = lambda: u.getSplit(X,y_current, classes)
		for i in range(0,self.max_iter):
			labIx, unlabIxNew = getInds()
			if len(list(set(unlabIxNew).symmetric_difference(set(unlabIx))))==0: # Break if nothing has changed in last iteration.
				break
			unlabIx = copy(unlabIxNew)
			weights[unlabIxNew] *= self.discount**(i+1)
			print('number of unlabelled points: {}'.format(len(unlabIx)))
			self.classifier.fit(X[labIx,:],y_current[labIx], sample_weight=weights[labIx])
			if len(unlabIx)==0:
				break
			print('Predicting unlabelled data')
			P = self.classifier.predict_proba(X[unlabIx,:])
			y_current[unlabIx] = np.fromiter([self._getLabel(P[i,:]) for i in range(0,P.shape[0])], int)
		return self


"""
1 - Fit on training data
2 - Predict test data
3 - Use predictions with high confidence to fit on test data
4 - Use resulting classifier to predict remaining point
5 - (Loop 2-4 or 3-4, ???)
""" 
class CustomSelfTrainer(ClassifierWrapper):

	# Base estimator must support estimation of probabilities.
	def __init__(self, classifier=SVC(probability=True), treshold=0.9):
		self.classifier = classifier
		self.treshold = treshold

	# Given the predicted probability of each class, return the predicted label.
	def _getLabel(self, probabilities):
		if np.all(probabilities<self.treshold):
			return -1
		return self.classifier.classes_[np.where(probabilities==max(probabilities))]

	def fit(self, X, y):
		classes = list(set(np.unique(y)) - {-1})
		# Train on labelled data
		labIx, unlabIx = u.getSplit(X,y,classes)
		Xtrn, Xtst, Ytrn, Ytst = u.getSplitData((labIx, unlabIx), X, y)
		self.classifier.fit(Xtrn, Ytrn)

		# Predict unlabelled data
		P = self.classifier.predict_proba(Xtst)
		Ytst = np.fromiter([self._getLabel(P[i,:]) for i in range(0,P.shape[0])], int)

		# Use predictions made with high enough confidence as training data.
		labIx2, unlabIx2 = u.getSplit(Xtst,Ytst, classes)
		self.classifier.fit(Xtst[labIx2,:], Ytst[labIx2])




