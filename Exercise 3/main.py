from SSLDA_Classifier import SSLDA_Classifier
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import math as m
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal, bernoulli
from copy import copy
from generateData import *

def data_plot(X,y):
    C1, C2, Cunl = np.where(y==1)[0], np.where(y==0)[0], np.where(y==-1)[0]
    plt.scatter(X[C1,0],X[C1,1], marker='o', c='blue', s=40)
    plt.scatter(X[C2,0],X[C2,1], marker='x', c='red', s=40)
    plt.scatter(X[Cunl,0],X[Cunl,1], marker='.', c='grey')

def contour_plot2(X, classifier, n=100):
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

def errors(X,y,y_true,classifier):
    mask = np.ones(len(y), dtype=bool)
    mask[np.where(y==-1)[0]]=False
    train_error = 1 - classifier.score(X[mask,:],y[mask])
    test_error = 1 - classifier.score(X[~mask,:],y_true[~mask])
    return train_error, test_error

def plot_methods(X,y,y_true,max_iter=100):
    methods = ['supervised','label-propagation','self-training']
    classifiers = {}
    labeled = np.where(y!=-1)[0]
    plt.figure(0)
    s = "           | Train error    |   Test error\n"
    for i,method in enumerate(methods): # note the order here matters for functionality
        sslda = SSLDA_Classifier(max_iter)
        sslda.fit(X,y, method=method)
        plt.subplot('23'+str(i+1))
        if method=='supervised':
            data_plot(X[labeled,:],y[labeled])
        else:
            data_plot(X,y)
        contour_plot(X,sslda)
        plt.title(str(method))
        train_error, test_error = errors(X,y,y_true,sslda)
        s+= "{}:    | {}    |   {}\n".format(method, train_error, test_error)
        classifiers[method]=sslda
    plt.subplot('234')
    data_plot(X, y_true)
    plt.title('True labels')
    plt.subplot('235')
    data_plot(X,classifiers['label-propagation'].propagated_labels)
    plt.title('Propagation of labels')
    #data_plot(X,classifiers['label-propagation'].predict(X))
    #plt.title('Label-prop: Predicted labels')
    plt.subplot('236')
    data_plot(X, classifiers['self-training'].predict(X))
    plt.title('Self-training: Predicted labels')
    #print(s)

def contour_plot(X, classifier, n=200):
    cov = classifier.covariance_
    m1,m2 = classifier.means_
    thetas = np.linspace(0,2*m.pi, n)
    x1 = np.atleast_2d(np.ones(n) * np.fromiter([m.cos(theta) for theta in thetas], thetas.dtype))
    x2 = np.atleast_2d(np.ones(n) * np.fromiter([m.sin(theta) for theta in thetas], thetas.dtype))
    S = np.concatenate((x1.T,x2.T),axis=1)
    L = np.dot(S, cov)
    plt.plot(L[:,0]+m1[0], L[:,1]+m1[1], c='red', linewidth=3)
    plt.plot(L[:,0]+m2[0], L[:,1]+m2[1], c='blue', linewidth=3)

def getError(X,y,method,Nunl, N1=75, N2=75,max_iter=100, p=0.5):
    X_train, y_train, y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    labelled = np.where(y_train!=-1)[0]
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    return 1-sslda.score(X_train[labelled,:], y_train_true[labelled]), 1-sslda.score(X_test, y_test)

def getErrors(X,y,method, Nunl, repeat, max_iter=100, p=0.5):
    errors = [getError(X,y,method, Nunl, max_iter=max_iter, p=0.5) for i in range(0,repeat)]
    train_errors = np.array([error[0] for error in errors])
    test_errors = np.array([error[1] for error in errors])
    return train_errors, test_errors

def getLikelihood(X,y,method, Nunl, N1=75, N2=75, max_iter=100, p=0.5):
    X_train, y_train, y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    C1 = np.where(y_train==0)[0]
    C2 = np.where(y_train==1)[0]
    log_proba = sslda.predict_log_proba(X_train)
    loglikelihood = sum(log_proba[C1,0]) + sum(log_proba[C2,0])
    return loglikelihood

def getLikelihoods(X,y,method, Nunl, repeat, max_iter=100, p=0.5):
    likelihoods = [getLikelihood(X,y,method, Nunl, max_iter=max_iter, p=p) for i in range(0,repeat)]
    return np.array(likelihoods)

def plotErrors(X,y, N_unlabelled, repeat, p=0.5, max_iter=100):
    methods = ['supervised', 'self-training', 'label-propagation']
    errors = {'supervised' : [], 'self-training' : [], 'label-propagation' : []}
    likelihoods = {'supervised' : [], 'self-training' : [], 'label-propagation' : []}
    for method in methods:
        print(method)
        for Nunl in N_unlabelled:
            train_likelihoods = getLikelihoods(X,y,method,Nunl,repeat,max_iter=max_iter, p=p)
            train_errors, test_errors = getErrors(X,y,method, Nunl, repeat, max_iter=max_iter,p=p)
            train_error = {'mean' : train_errors.mean(), 'std' : train_errors.std()}
            test_error = {'mean' : test_errors.mean(), 'std' : test_errors.std()}
            likelihood = {'mean' : train_likelihoods.mean(), 'std' : train_likelihoods.std()}
            errors[method].append({'train': train_error, 'test': test_error})
            likelihoods[method].append(likelihood)
        train_means = [obj['train']['mean'] for obj in errors[method]]
        train_stds = [obj['train']['std'] for obj in errors[method]]
        test_means = [obj['test']['mean'] for obj in errors[method]]
        test_stds = [obj['test']['std'] for obj in errors[method]]
        likelihood_means = [obj['mean'] for obj in likelihoods[method]]
        likelihood_stds = [obj['std'] for obj in likelihoods[method]]
        plt.figure(1)
        plt.errorbar(N_unlabelled, train_means, yerr = train_stds, label=method)
        plt.legend()
        plt.xlabel('$N_{unl}$', fontsize=18)
        plt.ylabel('Error', fontsize=15)
        plt.title('Error on training data')
        plt.legend()
        plt.figure(2)
        plt.errorbar(N_unlabelled, test_means, yerr = test_stds, label=method)
        plt.xlabel('$N_{unl}$', fontsize=18)
        plt.ylabel('Error', fontsize=15)
        plt.title('Error on test data')
        plt.legend()
        plt.figure(3)
        plt.errorbar(N_unlabelled, likelihood_means, yerr=likelihood_stds, label=method)
        plt.xlabel('$N_{unl}$', fontsize=18)
        plt.ylabel('Log-likelihood', fontsize=15)
        plt.title('Log-likelihood of training data')
        plt.legend()

def spambase(repeat, max_iter=100, N_unlabelled=[0, 10, 20, 40, 80, 160, 320, 640, 1280] ):
    filename = 'spambase.data'
    data = pd.read_csv(filename, sep=',',header=None).as_matrix()
    X = data[:,:-1]
    scale(X, axis=0, with_mean=False,copy=False)
    y = data[:,-1]
    plotErrors(X,y,N_unlabelled, repeat, max_iter=max_iter)

def main():
    N_unlabelled = [0,10, 20, 40,60, 80,140,250,320,400,500,640,750,900,1280]
    #N_unlabelled = [0,50,100,200,300]
    methods = ['supervised', 'self-training','label-propagation']
    repeat = 5
    #spambase(repeat, N_unlabelled=N_unlabelled)
 
    N1,N2, Nunl = 75, 75, 1200
    mean1, mean2 = [0,0], [0,0]
    cov1 = np.array([[10,0],[0,1]])
    cov2 = np.array([[1,0],[0,10]])
    #X,y, y_true = gaussianData(Nlab, Nunl, mean1=mean1, cov1=cov1, mean2=mean2, cov2=cov2 )
    gaussGen = lambda N: rnd.multivariate_normal([0,-0.5], np.array([[1,0],[0,1]]), N)
    circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_mean=m.pi/2, angle_variance=m.pi/3)
    #circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_range=(-1/4*m.pi, 5/4*m.pi))
    #X,y = generateData(gaussGen, circGen, N1+N2, p=0.8)
    p = 0.2
    X,y = customData1(N1+N2+10000, p=p)
    X_train, y_train,y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    #plt.figure(0)
 
    plot_methods(X_train,y_train,y_train_true, max_iter=100)
    plotErrors(X,y,N_unlabelled, repeat, p=p,max_iter=100)
    print('Train error:')
    for method in methods:
        train_errors, test_errors = getErrors(X,y,method, Nunl, repeat,p=p)
        print('{}: {:0.3f} +- {:0.4f}'.format(method,train_errors.mean(), train_errors.std()))

    plt.show()
    

if __name__ == "__main__":
    main()
