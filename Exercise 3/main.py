from SSLDA_Classifier import SSLDA_Classifier
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy.stats import multivariate_normal

# Nlab and Nunlab are the number of labels per class
def generateData(N1, N2, Nunl, mean1=[0,0], cov1=np.eye(2),mean2=[0,0], cov2=np.eye(2)):
    unl = rnd.randint(0,2, Nunl)
    Nunl1 = len(np.where(unl==0)[0])
    Nunl2 = len(np.where(unl==1)[0])
    X1 = rnd.multivariate_normal(mean1, cov1, size=N1)
    X2 = rnd.multivariate_normal(mean2, cov2, size=N2)
    XU1 = rnd.multivariate_normal(mean1, cov1, size=Nunl1)
    XU2 = rnd.multivariate_normal(mean2, cov2, size=Nunl2)
    X = np.concatenate((X1,X2,XU1, XU2), axis=0)
    y = np.concatenate((np.zeros(N1),np.ones(N2), -np.ones(Nunl)))
    return X,y

def data_plot(X,y):
    C1, C2, Cunl = np.where(y==1)[0], np.where(y==0)[0], np.where(y==-1)[0]
    plt.scatter(X[C1,0],X[C1,1], marker='o', c='blue', s=40)
    plt.scatter(X[C2,0],X[C2,1], marker='x', c='red', s=40)
    plt.scatter(X[Cunl,0],X[Cunl,1], marker='.', c='grey')
    return plt

def contour_plot(X, classifier, n=100):
    normal1 = multivariate_normal(mean=classifier.means_[0,:], cov=classifier.covariance_)
    normal2 = multivariate_normal(mean=classifier.means_[1,:], cov=classifier.covariance_)
    (X1max, X2max) = np.amax(X,axis=0)
    (X1min, X2min) = np.amin(X,axis=0)
    x1, x2 = np.linspace(X1min, X1max, n), np.linspace(X2min, X2max, n)
    X1,X2 = np.meshgrid(x1,x2)
    Z1 = np.array([[ normal1.pdf([x,y]) for x in x1 ] for y in x2])
    Z2 = np.array([[ normal2.pdf([x,y]) for x in x1 ] for y in x2])
    plt.contour(X1,X2,Z1[:,:])
    plt.contour(X1,X2,Z2[:,:])
    return plt

def main():
    N1, N2, Nunl = 20,20,1000
    mean1, mean2 = [-5,-2], [2,-2]
    cov1 = np.array([[2,0],[0,1]])
    cov2 = np.array([[2,0],[0,1]])
    X,y = generateData(N1, N2, Nunl, mean1=mean1, cov1=cov1, mean2=mean2, cov2=cov2 )
    sslda = SSLDA_Classifier(max_iter=10)
    sslda.fit(X,y, method='label-propagation')
    plt.figure(1)
    data_plot(X,y)
    contour_plot(X, sslda)
    plt.show()

if __name__ == "__main__":
    main()
