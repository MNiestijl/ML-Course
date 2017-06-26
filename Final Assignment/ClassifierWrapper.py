import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

# Wrapper for an abstract classifier (i.e., takes a classifier as input).
class ClassifierWrapper(BaseEstimator, ClassifierMixin):
	def __init__(self, classifier=SVC()):
		self.classifier=classifier

	def fit(self, X, y):
		return self.classifier.fit(X,y)

	def predict(self, X):
		return np.array(self.classifier.predict(X), dtype=int)

	def score(self, X, y):
		return self.classifier.score(X,y)

	def predict_proba(self, X):
		return self.classifier.predict_proba(X)

	def predict_log_proba(self, X):
		return self.classifier.predict_log_proba(X)