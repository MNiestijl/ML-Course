import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

"""
Always returns the label that was observed most often in the training set.
"""
class TrivialClassifier(BaseEstimator, ClassifierMixin):

	def fit(self,X, y):
		occurances = dict(zip(*np.unique(y, return_counts=True)))
		self.y_ = max(occurances, key=occurances.get)
		return self

	def predict(self, X):
		return self.y_*np.ones(X.shape[0], dtype=int)