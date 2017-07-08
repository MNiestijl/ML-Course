import numpy as np 
import math as m 
from plots import *
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import utils as u
from CustomClassifiers import CC1, CC2, CC3
from sklearn.ensemble import IsolationForest, AdaBoostClassifier, RandomForestClassifier
from SelfTrainer import SelfTrainer, CustomSelfTrainer
from LabelPropagationClassifier import LabelPropagationClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from TrivialClassifier import TrivialClassifier

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

- Find classifier that separates [1,2,3] and [4,5,6] best using cv
- find classifier that separates [6] and [4,5] best using cv
- find classifier that separates [4] and [5] best using cv

- Add weights to samples based on their variance.
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

def test(Xtrn, Ytrn, sample_weight=None, Strn=None):
	CL01 = lambda : SVC(C=10, kernel='poly', degree=3, class_weight='balanced')
	CL02 = lambda : SVC(C=10, kernel='poly', degree=3)
	CL1 = lambda C, deg: SVC(C=C, kernel='poly', degree=deg, class_weight='balanced')
	CL2 = lambda : LinearDiscriminantAnalysis(solver='svd', n_components=20)
	CL3 = lambda C,deg: SVC(C=C, kernel='poly', degree=deg, class_weight='balanced', decision_function_shape='ovr', probability=True)
	RFC = lambda n_estimators: RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', n_jobs=3, class_weight='balanced')
	best = lambda: CC1(CL1(10, 2), CL2(), CL3(10,5))
	classifiers = [
		CL01(),
	]
	for classifier in classifiers:
		if Strn is not None:
			scores = cross_val_score(classifier,Xtrn,Ytrn, fit_params={'S': Strn, 'sample_weight':sample_weight}, cv=10)
		else:
			scores = cross_val_score(classifier,Xtrn,Ytrn, fit_params={'sample_weight':sample_weight}, cv=10)
		print('\nScore = {} +/- {}'.format(scores.mean(), scores.std()))


def main():
	Xtrn, Ytrn, Strn, Xtst, Xall, Yall = u.getData()
	persons = np.unique(Strn)
	labels = np.unique(Ytrn)

	# Label active and unactive samples
	labs = [1,2,3]
	Yact = u.mapIsIn(Ytrn,labs)
	Xtrn1, Xtrn2, Ytrn1, Ytrn2, Strn1, Strn2 = u.splitData(labs, Xtrn, Ytrn, S=Strn)
	#ix1,ix2 = u.getSplit(Ytrn,labs)
	#Xtrn1, Xtrn2, Ytrn1, Ytrn2, Strn1, Strn2 = u.splitByInds((ix1,ix2),Xtrn, Ytrn, Strn)
	Yislab = np.concatenate((np.ones(Xtrn.shape[0]),-np.ones(Xtst.shape[0])))
	_, X45, _, Y45, _, S45 = u.splitData([6], Xtrn2, Ytrn2, S=Strn2)
	Y2is6 = u.mapIsIn(Ytrn2,[6])

	""" PLOT OUTLIERS
	isoForest = IsolationForest(n_estimators=100, contamination=0.005, n_jobs=3)
	isoForest.fit(Xtrn2, Ytrn2)
	isInlier = isoForest.predict(Xtrn2)
	plotPrincipalComponents(plt.figure(8), Xtrn2, isInlier)
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

	plotPersonData1 = lambda fig, person: plotPersonData(plt.figure(fig),Xtrn1,Ytrn1,Strn1,person)
	plotPersonData2 = lambda fig, person: plotPersonData(plt.figure(fig),Xtrn2,Ytrn2,Strn2,person)
	plotPersonData6_1 = lambda fig, person: plotPersonData(plt.figure(fig),X45,Y45,S45,person)
	plotPersonData6_2 = lambda fig, person: plotPersonData(plt.figure(fig),X45,Y45,S45,person, components=[1,2,3])

	#plotSingularValues(plt.figure(1),N=10,normalize=True)
	#plotPrincipalComponents(plt.figure(2),Xtrn,Ytrn)
	#plotPrincipalComponents(plt.figure(3),Xtrn,Yact)
	#plotPrincipalComponents(plt.figure(4),Xtrn1,Ytrn1)
	#plotPrincipalComponents(plt.figure(5),Xtrn2,Ytrn2)
	#plotPrincipalComponents(plt.figure(7), Xall, Yislab)
	#plotPersonData1(8,11)
	#plotPersonData2(8,9)
	#plotPrincipalComponents(plt.figure(10), X45, Y45)
	#plotPersonData6_1(11, 1)
	#plotPersonData6_2(12, 11)
	#plotRanks(plt.figure(13), Xtrn,Ytrn,Strn)


	"""
	# Find optimal parameters:
	Cs = [1,3,5,7,10,13,15,17,20,24,27,30]
	kernels = ['rbf', 'poly']
	degrees = [1,2,3,4,5]

	#best1, allScores1 = OptimizeSVCParameters(Xtrn1, Ytrn1, Cs, kernels, degrees, cv=10)
	#best2, allScores2 = OptimizeSVCParameters(Xtrn2, Ytrn2, Cs, kernels, degrees, cv=10)
	"""



	W = u.getWeights(Xtrn, Ytrn, Strn)
	Wall = np.ones(Xall.shape[0])
	Wall[np.where(Yall!=-1)] = W # NOTE: depends on stable ordering of Yall

	# Define classifiers	

	CL1 = SVC(C=10, kernel='poly', degree=2, class_weight='balanced')
	#CL2 = SVC(C=10, kernel='poly', degree=3, class_weight='balanced', decision_function_shape='ovr', probability=True)
	CL2 = LinearDiscriminantAnalysis(solver='svd')
	CL3 = SVC(C=10, kernel='poly', degree=3, class_weight='balanced', decision_function_shape='ovr', probability=True)
	CL5 = SVC(C=10, kernel='poly', degree=5, class_weight='balanced', decision_function_shape='ovr', probability=True)
	RFC = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=1, class_weight='balanced')
	classifier = CC1(CL1, CL2, CL3)
	#classifier = SVC(C=10, kernel='poly', degree=3, probability=True, decision_function_shape='ovr', class_weight='balanced')
	#classifier = KNeighborsClassifier()
	labelProp = LabelPropagationClassifier(
		classifier = classifier,
		Propagator = LabelSpreading(kernel='knn',n_neighbors=5, alpha=0.9)
	)
	customSelfTrainer = CustomSelfTrainer(classifier=classifier, treshold=0.95)
	getSelfTrainer = lambda: SelfTrainer(classifier=classifier, discount=0.99, tresholds=np.linspace(.99,.60, 200))
	
	# Make submission File

	name = 'selfTrainer_17'
	u.makeSubmissionFile(Xall, Yall, Xtst, getSelfTrainer, name=name, override=False, sample_weight=Wall, repeats=2)

	best = 'selfTrainer_14'
	y0 = u.loadSubmission(best)
	y1 = u.loadSubmission(name)
	print(sum([ 1 if y0[i]!=y1[i] else 0 for i in range(0,len(y0)) ]))

	"""
	print('\nScores without weights:')
	test(Xtrn, Ytrn)
	print('\nScore with weights')
	test(Xtrn, Ytrn, sample_weight=W)
	"""
	
	#plt.hist(W)


	plt.show()

if __name__ == "__main__":
	main()