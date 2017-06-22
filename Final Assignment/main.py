import numpy as np 
import math as m 
from plots import *
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import utils as u
from CustomClassifier import CC1, CC2

# Settings
#mpl.rcParams['text.usetex'] = True

def main():
	Xtrn, Ytrn, Strn, Xtst, Xall, Yall,Xfull = u.getData()

	# Label active and unactive samples
	labs = [1,2,3]
	Yact = u.getActLabels(Ytrn,labs)
	_,_, Xtrn1, Xtrn2, Ytrn1, Ytrn2 = u.splitData(Xtrn,Ytrn,labs)

	# Make plots
	#plotSingularValues(plt.figure(1),Xfull,N=10,normalize=True)
	#plotPrincipalComponents(plt.figure(2),Xfull,Ytrn)
	#plotPrincipalComponents(plt.figure(3),Xfull,Yact)
	#plotPrincipalComponents(plt.figure(4),Xtrn1,Ytrn1)
	#plotPrincipalComponents(plt.figure(5),Xtrn2,Ytrn2)
	#plt.show()

	CL1 = SVC(kernel='rbf')
	CL2 = SVC(kernel='rbf')
	CL3 = SVC(kernel='rbf')
	classifiers = [
		SGDClassifier(loss='hinge', n_jobs=3),
		SGDClassifier(loss='log', n_jobs=3),
		SGDClassifier(loss='squared_hinge', n_jobs=3),
		GaussianProcessClassifier(n_jobs=3),
		QuadraticDiscriminantAnalysis(),
		KNeighborsClassifier(n_neighbors=5, n_jobs=3),
		KNeighborsClassifier(n_neighbors=20, n_jobs=3),
		CC1(CL1, CL2, CL3),
		CC2(CL1, CL2),
	]

	for classifier in classifiers:
		scores = cross_val_score(classifier,Xfull,Ytrn, cv=10)
		print('\nScore = {} +/- {}'.format(scores.mean(), scores.std()))



if __name__ == "__main__":
	main()