import numpy as np
from scipy.io import loadmat, savemat

def getData():
	Xtrn = loadmat('Xtrn.mat')['Xtrn']
	Ytrn = loadmat('Ytrn.mat')['Ytrn']
	Strn = loadmat('Strn.mat')['Strn']
	Xtst = loadmat('Xtst.mat')['Xtst']
	# Add unlabelled data:
	Xall = np.concatenate((Xtrn,Xtst),axis=0)
	Yall = np.concatenate((Ytrn,-np.ones((Xtst.shape[0],1))))
	Xfull = np.concatenate((Xtrn,Strn),axis=1)
	return Xtrn, Ytrn, Strn, Xtst, Xall, Yall,Xfull

def getActLabels(y,labs):
	return np.array([1 if yi in labs else 0 for yi in y])

# Split data based in labs. Includes unlabelled points in both splits.
def splitData(X,y,labs):
	C1 = [ i for i,yi in enumerate(y) if yi in labs or yi==-1 ]
	C2 = [ i for i,yi in enumerate(y) if yi not in labs or yi==-1 ]
	return C1, C2, X[C1,:], X[C2,:], y[C1],y[C2]