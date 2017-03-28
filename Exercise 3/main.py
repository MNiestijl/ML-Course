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
    #data_plot(X,classifiers['label-propagation'].propagated_labels)
    #plt.title('Propagation of labels')
    data_plot(X,classifiers['label-propagation'].predict(X))
    plt.title('Label-prop: Predicted labels')
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

def splitData(X,y, N1lab, N2lab, Nunl, p=0.5):
    unl = bernoulli.rvs(p=p, size=Nunl)
    N1unl = len(np.where(unl==0)[0])
    N2unl = len(np.where(unl==1)[0])
    inds1 = rnd.choice(np.where(y==0)[0],size=N1lab + N1unl, replace=False)
    inds2 = rnd.choice(np.where(y==1)[0],size=N2lab + N2unl, replace=False)
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask_unl = np.concatenate((inds1[N1lab:], inds2[N2lab:]))
    train_mask[np.concatenate((inds1, inds2))] = True
    y_train_true = y[train_mask]
    y_train = copy(y)
    y_train[train_mask_unl] = -1
    y_train = y_train[train_mask]
    #y_train = np.concatenate((np.zeros(N1lab), -np.ones(N1unl),np.ones(N2lab), -np.ones(N2unl)))
    X_train = X[train_mask,:]
    X_test, y_test = X[~train_mask,:], y[~train_mask]
    return X_train, y_train,y_train_true, X_test, y_test

def getError(X,y,method,Nunl, N1=75, N2=75,max_iter=100, p=0.5):
    X_train, y_train, y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    return 1-sslda.score(X_train, y_train_true), 1-sslda.score(X_test, y_test)

def getErrors(X,y,method, Nunl, repeat, max_iter=100, p=0.5):
    errors = [getError(X,y,method, Nunl, max_iter=max_iter, p=0.5) for i in range(0,repeat)]
    train_errors = np.array([error[0] for error in errors])
    test_errors = np.array([error[1] for error in errors])
    return train_errors, test_errors

def getLikelihood(X,y,method, Nunl, N1=75, N2=75, max_iter=100, p=0.5):
    X_train, y_train, y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    log_proba = sslda.predict_log_proba(X)
    loglikelihood = sum([log_proba[:,int(label)] for label in y_train_true])
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
        likelihood_std = [obj['std'] for obj in likelihoods[method]]
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
        plt.errorbar(N_unlabelled, likelihood_means, yerr=likelihood_std, label=method)
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

# gen generates N points from the respective distribution
def generateData(gen1, gen2,N, p=0.5):
    classes = bernoulli.rvs(p=p, size=N)
    N1 = len(np.where(classes==0)[0])
    N2 = len(np.where(classes==1)[0])
    X = np.concatenate((gen1(N1),gen2(N2)), axis=0)
    y = np.concatenate((np.zeros(N1), np.ones(N2)))
    return X,y

# Nlab and Nunlab are the number of labels per class
def gaussianData(N, mean1=[0,0], cov1=np.eye(2),mean2=[0,0], cov2=np.eye(2)):
    gaussian1 = lambda N: rnd.multivariate_normal(mean1, cov1, size=N)
    gaussian2 = lambda N: rnd.multivariate_normal(mean2, cov2, size=N)
    return generateData(gaussian1, gaussian2, N)

def customData1(N, p):
    gaussian1 = lambda N: rnd.multivariate_normal([3,9], np.array([[1,0],[0,1]]), size=N)
    gaussian2 = lambda N: rnd.multivariate_normal([0,-2], np.array([[1,0],[0,1]]), size=N)
    gaussian3 = lambda N: rnd.multivariate_normal([7,10], np.array([[1,0],[0,1]]), size=N)
    gaussian4 = lambda N: rnd.multivariate_normal([15,-5], np.array([[1,0],[0,1]]), size=N)

    def path(N):
        x = np.atleast_2d(rnd.uniform(-3, 20, N))
        y = np.atleast_2d(0*np.ones(N))
        return np.concatenate((x.T,y.T),axis=1)

    def pathc(N):
        gen = circularGenerator(7.5,0.1,angle_range=(0,-m.pi))
        return np.array([7.5,0]) + gen(N)

    def getGen(N,probabilities):
        inds = rnd.choice(np.arange(0,len(probabilities)), size=N, p=probabilities)
        Ns = [len(np.where(inds==i)[0]) for i in range(0,len(probabilities))]
        data = np.concatenate((gaussian1(Ns[0]), gaussian2(Ns[1]),gaussian3(Ns[2]),gaussian4(Ns[3]),path(Ns[4])), axis=0)  
        rnd.shuffle(data)
        return data 

    gen1 = lambda N: getGen(N,[0.75, 0.25, 0  , 0  , 0])        
    gen2 = lambda N: getGen(N,[0  , 0  , 0, 0.2, 0.8  ])
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
    

def main():
    N_unlabelled = [0,10, 20, 40,60, 80,140,250,320,400,640,900,1280]
    #N_unlabelled = [0,100,500]
    methods = ['supervised', 'self-training','label-propagation']
    repeat = 5
    spambase(repeat, N_unlabelled=N_unlabelled)

    """
    N1,N2, Nunl = 75, 75, 1000
    mean1, mean2 = [0,0], [0,0]
    cov1 = np.array([[10,0],[0,1]])
    cov2 = np.array([[1,0],[0,10]])
    #X,y, y_true = gaussianData(Nlab, Nunl, mean1=mean1, cov1=cov1, mean2=mean2, cov2=cov2 )
    gaussGen = lambda N: rnd.multivariate_normal([0,-0.5], np.array([[1,0],[0,1]]), N)
    circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_mean=m.pi/2, angle_variance=m.pi/3)
    #circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_range=(-1/4*m.pi, 5/4*m.pi))
    #X,y = generateData(gaussGen, circGen, N1+N2, p=0.8)
    p = 0.7
    X,y = customData1(N1+N2+5000, p=p)
    X_train, y_train,y_train_true, X_test, y_test = splitData(X,y,N1,N2,Nunl=Nunl, p=p)
    #plt.figure(0)
 
    plot_methods(X_train,y_train,y_train_true, max_iter=100)
    plotErrors(X,y,N_unlabelled, repeat, p=p,max_iter=100)
    print('Train error:')
    for method in methods:
        train_errors, test_errors = getErrors(X,y,method, Nunl, repeat,p=p)
        print('{}: {:0.3f} +- {:0.4f}'.format(method,train_errors.mean(), train_errors.std()))
     """

    plt.show()

if __name__ == "__main__":
    main()
