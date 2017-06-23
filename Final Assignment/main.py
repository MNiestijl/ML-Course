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

"""
TODO:
- Find optimal classifier for two separate problems.
- Hyperparameter selection! i.e., grid search. One for each problem! AUTOMATE, run ONCE, do not do manualy!!
- Use test data to improve performance (somehow..)!
- Self-training? 
	Fit on TRAIN 	 
	predict TEST 	 
	Fit on all 		
	predict all (score TRAIN) 
	Repeat untill predicted labels on TEST converge or max_iter is reached.
"""

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

	CL1 = lambda : SVC(C=1, kernel='rbf')
	CL2 = lambda : SVC(C=10, kernel='poly', degree=2)
	CL3 = lambda : SVC(C=10, kernel='poly', degree=3)
	CL4 = lambda : SVC(C=10, kernel='rbf')
	classifiers = [
		#SGDClassifier(loss='hinge', n_jobs=3),
		#SGDClassifier(loss='log', n_jobs=3),
		#SGDClassifier(loss='squared_hinge', n_jobs=3),
		#KNeighborsClassifier(n_neighbors=5, n_jobs=3),
		#KNeighborsClassifier(n_neighbors=20, n_jobs=3),
		#KNeighborsClassifier(weights='distance', n_neighbors=20, n_jobs=3),
		#CC1(CL1(), CL2(), CL2()),
		#CC1(CL1(), CL3(), CL3()),
		#CC1(CL1(), CL4(), CL4()),
		#CC2(CL2(), CL2()),
		#CC2(CL3(), CL3()),
		#CC2(CL4(), CL4()),
	]
	for classifier in classifiers:
		scores = cross_val_score(classifier,Xfull,Ytrn, cv=10)
		print('\nScore = {} +/- {}'.format(scores.mean(), scores.std()))

	# Make submission File
	u.makeSubmissionFile(Xtrn, Ytrn, Xtst, CL3(), name="SVC_C_10_poly_3", override=False)

if __name__ == "__main__":
	main()