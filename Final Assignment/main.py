import numpy as np 
import math as m 
from plots import *
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import utils as u
from CustomClassifier import CC1

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
	CL2 = LinearRegression(n_jobs=3)
	CL3 = LinearRegression(n_jobs=3)
	CL = CC1(CL1, CL2, CL3)
	CL.fit(Xtrn,Ytrn)
	print(CL.score(Xtrn, Ytrn))



if __name__ == "__main__":
	main()