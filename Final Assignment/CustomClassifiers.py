import numpy as np
import utils as u
import math as m
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

""" 
TODO: 
- Add outlier detection and fit on the inliers only! (Does this improve performance?)
- Can be done using trees or SVM with probability estimates (set treshhold)
""" 


# Split problem into two problems based on activity type using classifier1 and predict these two problems using classifier2 and classifier3
# The classifiers must support weighted input.
class CC1(BaseEstimator, ClassifierMixin):
	def __str__(self):
		return "CC1"

	def __init__(self,C1=SVC(),C2=SVC(),C3=SVC(),labs=[1,2,3]):
		self.C1 = C1 	# binary classifier
		self.C2 = C2	
		self.C3 = C3
		self.labs = labs

	def fit(self,X,y, sample_weight=None):
		Yact = u.getActLabels(y,self.labs)
		self.C1.fit(X,Yact,sample_weight=sample_weight)
		ix1,ix2, X1, X2, y1, y2 = u.splitData(X,y,self.labs)
		self.C2.fit(X1,y1,sample_weight=sample_weight[ix1])
		self.C3.fit(X2,y2,sample_weight=sample_weight[ix2])
		return self

	def predict(self,X):
		pred1 = self.C1.predict(X)
		ix1,ix2,X1,X2,_,_ = u.splitData(X,pred1,[1])
		y1 = self.C2.predict(X1)
		y2 = self.C3.predict(X2)
		y = np.ones(X.shape[0], dtype=int)
		y[ix1] = y1
		y[ix2] = y2
		return y

	def score(self,X,y):
		predicted = self.predict(X)
		return np.mean([1 if predicted[i]==y[i] else 0 for i in range(0,len(y))])


# predict activity type by preClassifier, add as feature and then predict using mainClassifier.
class CC2(BaseEstimator, ClassifierMixin):
	def __str__(self):
		return "CC2"

	def __init__(self,preCL=SVC(),mainCL=SVC(),labs=[1,2,3]):
		self.preCL = preCL
		self.mainCL = mainCL
		self.labs = labs

	def fit(self, X, y, sample_weight=None):
		Yact = u.getActLabels(y,self.labs)
		self.preCL.fit(X,Yact,sample_weight=sample_weight)
		X2 = np.concatenate((X,Yact.reshape(len(Yact),1)),axis=1) 	
		self.mainCL.fit(X2,y,sample_weight=sample_weight)
		self._classes = np.unique(y)
		return self

	def predict(self, X):
		pred1 = self.preCL.predict(X)
		X2 = np.concatenate((X,pred1.reshape(len(pred1),1)),axis=1)
		return self.mainCL.predict(X2)

	def score(self, X, y):
		predicted = self.predict(X)
		return np.mean([1 if predicted[i]==y[i] else 0 for i in range(0,len(y))])
