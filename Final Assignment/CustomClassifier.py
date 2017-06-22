import numpy as np
import utils as u
import math as m

# WERKT NOG NIET ECHT =(
class CC1():
	def __init__(self, classifier1, classifier2, classifier3, labs=[1,2,3]):
		self.C1 = classifier1 	# binary classifier
		self.C2 = classifier2	
		self.C3 = classifier3
		self.labs = labs

	def fit(self,X,y):
		Yact = u.getActLabels(y,self.labs)
		self.C1.fit(X,Yact)
		_,_, X1, X2, y1, y2 = u.splitData(X,y,self.labs)
		self.C2.fit(X1,y1)
		self.C3.fit(X2,y2)

	def predict(self,X):
		pred1 = self.C1.predict(X)
		ix1,ix2,X1,X2,_,_ = u.splitData(X,pred1,[0])
		y1 = np.array([m.floor(x) for x in self.C2.predict(X1)])
		y2 = np.array([m.floor(x) for x in self.C3.predict(X1)])
		y = np.ones(X.shape[0])
		y[ix1] = y1
		y[ix2] = y2
		return y

	def score(self,X,y):
		predicted = self.predict(X)
		return np.mean([predicted[i]==y[i] for i in range(0,len(y))])
