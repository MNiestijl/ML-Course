import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from ClassifierWrapper import ClassifierWrapper
import utils as u

class SelfTrainer(ClassifierWrapper):

	# Base estimator must support estimation of probabilities.
	# (Provide either treshold and max_iter or tresholds. In case of both, tresholds is used.)
	def __init__(self, classifier=SVC(probability=True), treshold=0.7, max_iter=100, discount=1, tresholds=None):
		self.classifier = classifier
		self.treshold = treshold
		self.max_iter = max_iter
		self.discount = discount
		self.tresholds = tresholds # list of tresholds. 
		self.checkConvergence_ = False

	# Given the predicted probability of each class, return the predicted label.
	def _getLabel(self, probabilities, treshold):
		if np.all(probabilities<treshold):
			return -1
		return self.classifier.classes_[np.where(probabilities==max(probabilities))]

	def fit(self,X,y, sample_weight=None):
		self.classes_ = list(set(np.unique(y)) - {-1})
		y_current = copy(y)

		# Initialze values
		sample_weight = np.ones(len(y)) if sample_weight is None else sample_weight
		unlabIxOld = np.array([])
		if self.tresholds is None:
			self.checkConvergence_ = True
			self.tresholds = self.treshold * np.ones(max_iter)

		for i,treshold in enumerate(self.tresholds):
			print('\nTreshold: {}'.format(treshold))
			labIx, unlabIxNew = u.getSplit(y_current, self.classes_)
			sample_weight[unlabIxNew] = self.discount**(i+1)
			
			# Fit data
			print('number of unlabelled points: {}\n'.format(len(unlabIxNew)))
			self.classifier.fit(X[labIx,:],y_current[labIx], sample_weight=sample_weight[labIx])

			# Break if nothing has changed in last iteration and 
			if self.checkConvergence_ and len(list(set(unlabIxNew).symmetric_difference(set(unlabIxOld))))==0: 
				break

			# Predict unlabelled data and update labels
			P = self.classifier.predict_proba(X[unlabIxNew,:])
			y_current[unlabIxNew] = np.fromiter([self._getLabel(P[i,:], treshold) for i in range(0,P.shape[0])], int)
			unlabIxOld = copy(unlabIxNew)
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
		Xtrn, Xtst, Ytrn, Ytst = u.splitData(classes, X, y)
		self.classifier.fit(Xtrn, Ytrn)

		# Predict unlabelled data
		P = self.classifier.predict_proba(Xtst)
		Ytst = np.fromiter([self._getLabel(P[i,:]) for i in range(0,P.shape[0])], int)

		# Use predictions made with high enough confidence as training data.
		labIx2, unlabIx2 = u.getSplit(Ytst, classes)
		self.classifier.fit(Xtst[labIx2,:], Ytst[labIx2])


