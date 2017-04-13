import numpy as np
import numpy.random as rnd
import math as m
from scipy.stats import multivariate_normal, bernoulli
from copy import copy

def splitData(X,y, N1lab, N2lab, Nunl, p=0.5):
    unl = bernoulli.rvs(p=p, size=Nunl)
    N1unl = len(np.where(unl==0)[0])
    N2unl = len(np.where(unl==1)[0])
    inds1 = rnd.choice(np.where(y==0)[0],size=N1lab + N1unl, replace=False)
    inds2 = rnd.choice(np.where(y==1)[0],size=N2lab + N2unl, replace=False)
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask_unl = np.zeros(X.shape[0], dtype=bool)
    train_mask[np.concatenate((inds1, inds2))] = True
    train_mask_unl[np.concatenate((inds1[N1lab:], inds2[N2lab:]))] = True
    y_train_true = y[train_mask]
    y_train = copy(y)
    y_train[train_mask_unl] = -1
    y_train = y_train[train_mask]
    #y_train = np.concatenate((np.zeros(N1lab), -np.ones(N1unl),np.ones(N2lab), -np.ones(N2unl)))
    X_train = X[train_mask,:]
    X_test, y_test = X[~train_mask,:], y[~train_mask]
    return X_train, y_train,y_train_true, X_test, y_test

# gen generates N points from the respective distribution
def generateData(gen1, gen2,N, p=0.5):
    classes = bernoulli.rvs(p=p, size=N)
    N1 = len(np.where(classes==0)[0])
    N2 = len(np.where(classes==1)[0])
    X = np.concatenate((gen1(N1),gen2(N2)), axis=0)
    y = np.concatenate((np.zeros(N1), np.ones(N2)))
    return X,y

def gaussianData(N, mean1=[0,0], cov1=np.eye(2),mean2=[0,0], cov2=np.eye(2)):
    gaussian1 = lambda N: rnd.multivariate_normal(mean1, cov1, size=N)
    gaussian2 = lambda N: rnd.multivariate_normal(mean2, cov2, size=N)
    return generateData(gaussian1, gaussian2, N)

def customData1(N, p):
    gaussian1 = lambda N: rnd.multivariate_normal([3,5], np.array([[1,0],[0,1]]), size=N)
    gaussian2 = lambda N: rnd.multivariate_normal([-1,-2], np.array([[1,0],[0,1]]), size=N)
    gaussian3 = lambda N: rnd.multivariate_normal([12,4], np.array([[0.2,0],[0,0.2]]), size=N)
    gaussian4 = lambda N: rnd.multivariate_normal([-15,2], np.array([[1,0],[0,1]]), size=N)

    def path(N):
        x = np.atleast_2d(rnd.uniform(3, 7, N))
        y = np.atleast_2d(0*np.ones(N))
        return np.concatenate((x.T,y.T),axis=1)

    def pathc(N):
        gen = circularGenerator(7.5,0.1,angle_range=(0,-m.pi))
        gen = circularGenerator(7.5,0.1,angle_mean=-m.pi/2, angle_variance=m.pi/4)
        return np.array([7.5,0]) + gen(N)

    def getGen(N,probabilities):
        inds = rnd.choice(np.arange(0,len(probabilities)), size=N, p=probabilities)
        Ns = [len(np.where(inds==i)[0]) for i in range(0,len(probabilities))]
        data = np.concatenate((gaussian1(Ns[0]), gaussian2(Ns[1]),gaussian3(Ns[2]),gaussian4(Ns[3]),pathc(Ns[4])), axis=0)  
        rnd.shuffle(data)
        return data 

    gen1 = lambda N: getGen(N,[0.4, 0.2, 0, 0, 0.4])        
    gen2 = lambda N: getGen(N,[0  ,0 , 0.3, 0.7 , 0  ])
    return generateData(gen1, gen2, N, p=p)

def circularGenerator(radius_mean, radius_variance, angle_range=(0,2*m.pi), angle_mean=None, angle_variance=None):
    def generator(N):
        rs = rnd.normal(radius_mean, radius_variance, N)
        if angle_mean and angle_variance:
            thetas= rnd.normal(angle_mean, angle_variance,N)
        else:
            thetas = rnd.uniform(angle_range[0], angle_range[1], N)
        x1 = np.atleast_2d(rs * np.fromiter([m.cos(theta) for theta in thetas], thetas.dtype))
        x2 = np.atleast_2d(rs * np.fromiter([m.sin(theta) for theta in thetas], thetas.dtype))
        return np.concatenate((x1.T,x2.T),axis=1)
    return generator
