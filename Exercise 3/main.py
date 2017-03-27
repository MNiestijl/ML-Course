from SSLDA_Classifier import SSLDA_Classifier
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import math as m
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal, bernoulli


# gen generates N points from the respective distribution
def generateData(gen1, gen2,Nlab, Nunl, p=0.5):
    lab = bernoulli.rvs(p=p, size=Nlab)
    unl = bernoulli.rvs(p=p, size=Nunl)
    N1lab = len(np.where(lab==0)[0])
    N2lab = len(np.where(lab==1)[0])
    N1unl = len(np.where(unl==0)[0])
    N2unl = len(np.where(unl==1)[0])
    X1 = gen1(N1lab+N1unl)
    X2 = gen2(N2lab+N2unl)
    X = np.concatenate((X1,X2), axis=0)
    y = np.concatenate((np.zeros(N1lab), -np.ones(N1unl),np.ones(N2lab), -np.ones(N2unl)))
    ytrue = np.concatenate((np.zeros(N1lab+N1unl), np.ones(N2lab+N2unl)))
    return X,y,ytrue

# Nlab and Nunlab are the number of labels per class
def gaussianData(Nlab, Nunl, mean1=[0,0], cov1=np.eye(2),mean2=[0,0], cov2=np.eye(2)):
    gaussian1 = lambda N: rnd.multivariate_normal(mean1, cov1, size=N)
    gaussian2 = lambda N: rnd.multivariate_normal(mean2, cov2, size=N)
    return generateData(gaussian1, gaussian2, Nlab, Nunl)

def customData1(Nlab, Nunl, p):
    gaussian1 = lambda N: rnd.multivariate_normal([3,5], np.array([[1,0],[0,1]]), size=N)
    gaussian2 = lambda N: rnd.multivariate_normal([0,-5], np.array([[1,0],[0,1]]), size=N)
    gaussian3 = lambda N: rnd.multivariate_normal([7,10], np.array([[1,0],[0,1]]), size=N)
    gaussian4 = lambda N: rnd.multivariate_normal([10,-10], np.array([[1,0],[0,1]]), size=N)
    def path(N):
        x = np.atleast_2d(rnd.uniform(3, 7, N))
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

    gen1 = lambda N: getGen(N,[0.7, 0.3, 0  , 0  , 0])        
    gen2 = lambda N: getGen(N,[0  , 0  , 0.2, 0.2, 0.6  ])
    return generateData(gen1, gen2, Nlab, Nunl, p=0.3)

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
    print(s)

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

def getData(X,y, N1lab, N2lab, Nunl):
    unl = bernoulli.rvs(p=0.5, size=Nunl)
    N1unl = len(np.where(unl==0)[0])
    N2unl = len(np.where(unl==1)[0])
    inds1 = rnd.choice(np.where(y==0)[0],size=N1lab + N1unl, replace=False)
    inds2 = rnd.choice(np.where(y==1)[0],size=N2lab + N2unl, replace=False)
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[np.concatenate((inds1, inds2))] = True
    y_train = np.concatenate((np.zeros(N1lab), -np.ones(N1unl),np.ones(N2lab), -np.ones(N2unl)))
    X_train = X[train_mask,:]
    X_test, y_test = X[~train_mask,:], y[~train_mask]
    return X_train, y_train, X_test, y_test

def getScore(X,y,method,Nunl, max_iter=100):
    X_train, y_train, X_test, y_test = getData(X,y,75,75,Nunl=Nunl)
    sslda = SSLDA_Classifier(max_iter)
    sslda.fit(X_train,y_train, method=method)
    return 1-sslda.score(X_train, y_train), 1-sslda.score(X_test, y_test)

def spambase():
    max_iter=100
    repeat = 50
    filename = 'spambase.data'
    data = pd.read_csv(filename, sep=',',header=None).as_matrix()
    X = data[:,:-1]
    scale(X, axis=1, with_mean=False,copy=False)
    y = data[:,-1]
    test = SSLDA_Classifier(max_iter=100)
    #N_unlabelled = [0,10, 20,40,60, 80]
    N_unlabelled = [0,10, 20, 40,60, 80,140,250,320,400,640,900,1280]
    methods = ['supervised', 'self-training', 'label-propagation']
    scores = {'supervised' : [], 'self-training' : [], 'label-propagation' : []}
    
    for method in methods:
        print(method)
        for Nunl in N_unlabelled:
            errors = [getScore(X,y,method, Nunl, max_iter) for i in range(0,repeat)]
            train_errors = np.array([error[0] for error in errors])
            test_errors = np.array([error[1] for error in errors])
            train_error = {'mean' : train_errors.mean(), 'std' : train_errors.std()}
            test_error = {'mean' : test_errors.mean(), 'std' : test_errors.std()}
            scores[method].append({'train': train_error, 'test': test_error})
        train_means = [obj['train']['mean'] for obj in scores[method]]
        train_stds = [obj['train']['std'] for obj in scores[method]]
        test_means = [obj['test']['mean'] for obj in scores[method]]
        test_stds = [obj['test']['std'] for obj in scores[method]]
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
    plt.show()
    


def main():
    spambase()
    """
    Nlab, Nunl = 50,1000
    mean1, mean2 = [0,0], [0,0]
    cov1 = np.array([[10,0],[0,1]])
    cov2 = np.array([[1,0],[0,10]])
    #X,y, y_true = gaussianData(Nlab, Nunl, mean1=mean1, cov1=cov1, mean2=mean2, cov2=cov2 )
    gaussGen = lambda N: rnd.multivariate_normal([0,-0.5], np.array([[1,0],[0,1]]), N)
    circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_mean=m.pi/2, angle_variance=m.pi/3)
    #circGen = circularGenerator(radius_mean=5, radius_variance=0.5, angle_range=(-1/4*m.pi, 5/4*m.pi))
    #X,y, y_true = generateData(gaussGen, circGen, Nlab, Nunl, p=0.8)
    X,y,y_true = customData1(Nlab, Nunl, p=0.5)
    plt.figure(1)
    plot_methods(X,y,y_true, max_iter=100)
    plt.show()
    """

if __name__ == "__main__":
    main()
