import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.exceptions import NotFittedError
from pathlib import Path

def getData():
	Xtrn = loadmat('Xtrn.mat')['Xtrn']
	Ytrn = loadmat('Ytrn.mat')['Ytrn'].ravel()
	Strn = loadmat('Strn.mat')['Strn']
	Xtst = loadmat('Xtst.mat')['Xtst']
	# Add unlabelled data:
	Xall = np.concatenate((Xtrn,Xtst),axis=0)
	Yall = np.concatenate((Ytrn,-np.ones(Xtst.shape[0])))
	Xfull = np.concatenate((Xtrn,Strn),axis=1)
	return Xtrn, Ytrn, Strn, Xtst, Xall, Yall,Xfull

def getActLabels(y,labs):
	return np.array([1 if yi in labs else 0 for yi in y])

# Split data based in labs. Includes unlabelled points in both splits.
def splitData(X,y,labs):
	C1 = [ i for i,yi in enumerate(y) if yi in labs ]
	C2 = [ i for i,yi in enumerate(y) if yi not in labs ]
	return C1, C2, X[C1,:], X[C2,:], y[C1],y[C2]

def makeSubmissionFile(Xtrn, Ytrn, Xtest, Classifier, name="testSubmission", override=False):

	Classifier.fit(Xtrn,Ytrn)
	scores = Classifier.score(Xtrn,Ytrn)
	print('\nScore on Training set: {} +/- {}'.format(scores.mean(), scores.std()))

	path = 'submissions/' + name + '.csv'
	my_file = Path(path)
	if Path(path).exists() and override==False:
		raise('The following path is already occupied:\n{}'.format(path))

	try:
		y = Classifier.predict(Xtest)
	except NotFittedError as e:
		print(repr(e))

	with open(path, "w") as outfile:
		outfile.write("Id,Class\n")
		for e, lab in enumerate(list(y)):
			outfile.write("%s,%s\n" % (e+1,lab)) 
