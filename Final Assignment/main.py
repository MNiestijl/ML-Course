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
from CustomClassifiers import CC1, CC2
from sklearn.ensemble import IsolationForest
from SelfTrainer import SelfTrainer, CustomSelfTrainer
from LabelPropagationClassifier import LabelPropagationClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

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
- Label Propagation
- Other transductive methods??
- Covariate shift: Calculate the weights in the source distribution (don't initially assume target distribution is the same.) (use regularization)
"""

def OptimizeSVCParameters(X, Y, Cs, kernels, degrees=[3], cv=5):
	combinations = [(C,kernel) for C in Cs for kernel in kernels]
	allScores = {}
	best = {}
	bestScore = 0
	for C, kernel in combinations:
		print(C, kernel)
		if kernel=='poly':
			for degree in degrees:
				classifier = SVC(C=C, kernel=kernel, degree=degree)
				scores = cross_val_score(classifier, X, Y, cv=cv)
				allScores[(C,kernel, degree)] = { 'mean' : scores.mean(), 'std' : scores.std() }
				if scores.mean()>bestScore:
					best['C'] = C
					best['kernel'] = kernel
					best['degree'] = degree
					bestScore = scores.mean()
		else:
			classifier = SVC(C=C, kernel=kernel)
			scores = cross_val_score(classifier, X, Y, cv=cv)
			allScores[(C,kernel)] = { 'mean' : scores.mean(), 'std' : scores.std() }
			if scores.mean()>bestScore:
				best['C'] = C
				best['kernel'] = kernel
				best['score'] = { 'mean' : scores.mean(), 'std' : scores.std() }
				best['degree'] = -1
				bestScore = scores.mean()
	return best, allScores

def test(Xtrn, Ytrn, sample_weight=None):
	CL1 = lambda : SVC(C=1, kernel='rbf')
	CL2 = lambda : SVC(C=5, kernel='poly', degree=2)
	CL3 = lambda : SVC(C=10, kernel='poly', degree=2)
	CL4 = lambda : SVC(C=15, kernel='poly', degree=2)
	classifiers = [
		#SGDClassifier(loss='hinge', n_jobs=3),
		#SGDClassifier(loss='log', n_jobs=3),
		#SGDClassifier(loss='squared_hinge', n_jobs=3),
		#KNeighborsClassifier(n_neighbors=5, n_jobs=3),
		#KNeighborsClassifier(n_neighbors=20, n_jobs=3),
		#KNeighborsClassifier(weights='distance', n_neighbors=20, n_jobs=3),
		CL1(),
		CC1(CL1(),CL2(),CL3()),
		CC2(CL2(),CL3()),
		#CC1(CL1(), CL2(), CL2()),
		#CC1(CL1(), CL3(), CL3()),
		#CC1(CL1(), CL4(), CL4()),
		CC2(CL2(), CL2()),
		CC2(CL3(), CL3()),
		CC2(CL4(), CL4()),
	]
	for classifier in classifiers:
		scores = cross_val_score(classifier,Xtrn,Ytrn,fit_params={'sample_weight':sample_weight}, cv=10)
		print('\nScore = {} +/- {}'.format(scores.mean(), scores.std()))


def main():
	Xtrn, Ytrn, Strn, Xtst, Xall, Yall,Xfull = u.getData()

	# Label active and unactive samples
	labs = [1,2,3]
	Yact = u.getActLabels(Ytrn,labs)
	ix1,ix2 = u.getSplit(Xtrn,Ytrn,labs)
	Xtrn1, Xtrn2, Ytrn1, Ytrn2 = u.getSplitData((ix1,ix2),Xtrn, Ytrn)
	Yislab = np.concatenate((np.ones(Xtrn.shape[0]),-np.ones(Xtst.shape[0])))

	""" PLOT OUTLIERS
	isoForest = IsolationForest(n_estimators=100, contamination=0.005, n_jobs=3)
	isoForest.fit(Xtrn2, Ytrn2)
	isInlier = isoForest.predict(Xtrn2)
	plotPrincipalComponents(plt.figure(8), Xtrn2, isInlier)
	"""


	"""
	Xmn, Ymn = u.getMeanData(Xtrn, Ytrn, Strn)
	W = u.getWeights(Xtrn, Ytrn, Strn, sigma=1)
	print('\nScores without weights:')
	test(Xtrn, Ytrn)
	print('\nScore with weights')
	test(Xtrn, Ytrn, sample_weight=W)
	"""

	#classifier = SVC(C=10, kernel='poly', degree=3, probability=True, decision_function_shape='ovr')
	#selfTrainer = SelfTrainer(classifier=classifier, treshold=0.85, max_iter=10)
	#selfTrainer.fit(Xall,Yall)
	#labelProp = LabelPropagationClassifier(
	#	classifier = SVC(C=10, kernel='poly', degree=3, probability=True, decision_function_shape='ovr'),
	#	Propagator = LabelSpreading(kernel='rbf', alpha=0.8)
	#)
	#labelProp.fit(Xall, Yall)
	#predicted1 = labelProp.predict(Xtst)
	#predicted2 = labelProp.predict(Xall)
	
	# Make plots
	#plotSingularValues(plt.figure(1),Xfull,N=10,normalize=True)
	#plotPrincipalComponents(plt.figure(2),Xfull,Ytrn)
	#plotPrincipalComponents(plt.figure(3),Xfull,Yact)
	#plotPrincipalComponents(plt.figure(4),Xtrn1,Ytrn1)
	#plotPrincipalComponents(plt.figure(5),Xtrn2,Ytrn2)
	#plotPrincipalComponents(plt.figure(6), Xmn, Ymn)
	#plotPrincipalComponents(plt.figure(7), Xall, Yislab)

	#test(Xtrn, Ytrn)

	# Find optimal parameters:
	Cs = [1,3,5,7,10,13,15,17,20,24,27,30]
	kernels = ['rbf', 'poly']
	degrees = [1,2,3,4,5]

	#best1, allScores1 = OptimizeSVCParameters(Xtrn1, Ytrn1, Cs, kernels, degrees, cv=10)
	#best2, allScores2 = OptimizeSVCParameters(Xtrn2, Ytrn2, Cs, kernels, degrees, cv=10)	

	#Make submission File
	classifier = SVC(C=10, kernel='poly', degree=3, probability=True, decision_function_shape='ovr')
	customSelfTrainer = CustomSelfTrainer(classifier=classifier, treshold=0.8)
	name = 'customSelfTrainer_02'
	u.makeSubmissionFile(Xall, Yall, Xtst, customSelfTrainer, name=name, override=True)

	best = 'SVC_C_10_poly_3'
	y0 = u.loadSubmission(best)
	y1 = u.loadSubmission(name)
	print(sum([ y0[i]!=y1[i] for i in range(0,len(y0)) ]))


	plt.show()

if __name__ == "__main__":
	main()