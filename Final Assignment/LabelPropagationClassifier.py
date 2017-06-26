import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from ClassifierWrapper import ClassifierWrapper

class LabelPropagationClassifier(ClassifierWrapper):

	def __init__(self, classifier=SVC(), Propagator = LabelSpreading()):
		self.classifier = classifier
		self.Propagator = Propagator

	def fit(self, X, y):
		self.Propagator.fit(X,y)
		yPredicted = self.Propagator.predict(X)
		self.classifier.fit(X,yPredicted)
		return self

