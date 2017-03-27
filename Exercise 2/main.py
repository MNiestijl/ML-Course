from AdaBoost import DecisionStump, AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd
from sklearn.model_selection import ShuffleSplit
from copy import copy

def generateData(N):
    X1 = rnd.multivariate_normal(mean=[0,0], cov=np.eye(2), size=N)
    X2 = rnd.multivariate_normal(mean=[2,0], cov=np.eye(2), size=N)
    X = np.concatenate((X1,X2), axis=0)
    y1 = np.ones(N)
    y2 = -np.ones(N)
    y = np.concatenate((y1,y2))
    return X,y

def plotDecisionStump(stump,X,y,figure=1):
    pos1, pos2 = np.where(y==-1), np.where(y==1)
    plt.figure(figure)
    plt.scatter(X[pos1,0],X[pos2,1],marker='o', label='class 1')
    plt.scatter(X[pos2,0],X[pos2,1],marker='x', label='class 2')
    plt.xlabel('$x_{1}$', fontsize=18)
    plt.ylabel('$x_{2}$', fontsize=18)
    if stump.feature==0:
        plt.plot(stump.theta*np.ones(50),np.linspace(X[:,1].min(),X[:,1].max(),50), label='Decision boundary')
    else:
        plt.plot(np.linspace(X[:,0].min(),X[:,0].max(),50),stump.theta*np.ones(50), label='Decision boundary')
    plt.legend()

def plotDecisionStumpRescaled(X,y,w=None):
    stump1, stump2 = DecisionStump(), DecisionStump()
    stump1.fit(X,y,w=w)
    X2 = copy(X)
    X2[:,1] = 3*X2[:,1]
    stump2.fit(X,y,w=w)
    plotDecisionStump(stump1,X,y,figure=1)
    plotDecisionStump(stump2,X2,y,figure=2)
    plt.show()

def plotDecisionStumpWithWeights(X,y, weights):
    for i,w in enumerate(weights):
        stump = DecisionStump()
        stump.fit(X,y,w)
        plotDecisionStump(stump,X,y,figure=i)
        plt.show()

def plotDecisionBoundary(classifier,X,y):
    classifier.fit(X,y)
    pos1, pos2 = np.where(y==-1), np.where(y==1)
    rangex1, rangex2 = (X[:,0].min()-0.5,X[:,0].max()+0.5), (X[:,1].min()-0.5,X[:,1].max()+0.5)
    plt.scatter(X[pos1,0],X[pos1,1],marker='o', color='green',s=30, label='class 1')
    plt.scatter(X[pos2,0],X[pos2,1],marker='x',color='red',s=30, label='class 2')

    wpos = classifier.objectWeights.argsort()[-5:]
    plt.scatter(X[wpos,0],X[wpos,1],facecolors='none', edgecolors='yellow',linewidth=2,s=80)

    plt.xlim(rangex1)
    plt.ylim(rangex2)
    plt.xlabel('$x_{1}$', fontsize=18)
    plt.ylabel('$x_{2}$', fontsize=18)
    x1 = np.linspace(rangex1[0],rangex1[1], 100)
    x2 = np.linspace(rangex2[0],rangex2[1], 100)
    Z = np.array([[classifier.predict(np.array([[a,b]])) for b in x2] for a in x1])[:,:,0].T
    #Z = np.array([[classifier.expectedLabels(np.array([[a,b]]))/len(classifier.learners) for a in x1] for b in x2])[:,:,0]
    im = plt.imshow(np.flipud(Z), cmap=plt.cm.RdBu, extent=rangex1+rangex2, vmin=-1, vmax=1)  
    plt.colorbar(im) 

def cross_validate(classifier,X,y,n_splits,train_size):
    splits = ShuffleSplit(n_splits=n_splits, train_size=train_size).split(X)
    scores = []
    for train_indices, test_indices in splits:
        estimator = DecisionStump()
        estimator.fit(X[train_indices,:], y[train_indices])
        scores.append(estimator.score(X[test_indices,:],y[test_indices]))
    return np.array(scores)

def testDecisionStump(X,y):
    mask = np.zeros(len(y), dtype=bool)
    mask[np.concatenate((np.where(y==-1)[0][0:50],np.where(y==1)[0][0:50]))]=True
    stump = DecisionStump()
    stump.fit(X[mask,:],y[mask])
    first = stump.score(X[~mask,:],y[~mask])
    scores = cross_validate(DecisionStump,X,y,n_splits=50, train_size=50)
    print("Using first 50 of both classes as training data:\nscore: {}\n".format(first))
    print("Decision stump accuracy with random subsets:\nmean: {}, std: {}\n".format(scores.mean(),scores.std()))

def plotErrors(classifier):
    nIter = len(classifier.errors)
    plt.figure(1)
    iterations = np.arange(nIter)
    plt.subplot(1,2,1)
    plt.title('Error')
    plt.yscale('log')
    plt.plot(iterations,classifier.errors,'o')
    plt.subplot(1,2,2)
    plt.title('Weights')
    plt.plot(iterations,classifier.weights,'o')
    plt.show()

def adaBoostAccuracy(X,y, WeakLearner):
    mask = np.zeros(len(y), dtype=bool)
    mask[np.concatenate((np.where(y==-1)[0][0:50],np.where(y==1)[0][0:50]))]=True
    iterations = np.arange(1,200,5)
    scores = []
    for maxIter in iterations:
        classifier = AdaBoostClassifier(iterations=maxIter,WeakLearner=DecisionStump)
        classifier.fit(X[mask,:],y[mask])
        scores.append(classifier.score(X[~mask,:],y[~mask]))
    plt.plot(iterations, scores,'o')
    plt.title('Accuracy on test data')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    #plotErrors(classifier)
    plt.show()

def main():
    filename = 'optdigitsubset.txt'
    X = pd.read_csv(filename, delim_whitespace=True, header=None).as_matrix()
    y = np.concatenate((-np.ones(554),np.ones(571)),axis=0)
    #filename = 'banana.csv'
    #X = pd.read_csv(filename, header=None).as_matrix()
    #y = np.concatenate((-np.ones(50),np.ones(50)),axis=0)
    #N = 30
    #X,y = generateData(N)
    #w,w1,w2 = np.ones(2*N),np.ones(2*N),np.ones(2*N)
    #w1[0:N] = 5*w1[0:N]
    #w2[0:N] = 0.25*w2[0:N]
    #weights = [w,w1,w2]
    #plotDecisionStumpRescaled(X,y)
    #plotDecisionStumpWithWeights(X,y,weights)
    #testDecisionStump(X,y)
    """ 
    iterations = [5,20,50,200]
    plt.figure(2)
    for i,ix in enumerate(iterations):
        plt.subplot(2,2,i+1)
        classifier = AdaBoostClassifier(iterations=ix,WeakLearner=DecisionStump)
        plotDecisionBoundary(classifier,X,y)
    plt.show()
    """ 
    #adaBoostAccuracy(X,y,WeakLearner=DecisionStump)
    classifier = AdaBoostClassifier(iterations=5,WeakLearner=DecisionStump)
    mask = np.zeros(len(y), dtype=bool)
    mask[np.concatenate((np.where(y==-1)[0][0:50],np.where(y==1)[0][0:50]))]=True
    classifier.fit(X[mask,:],y[mask])
    wpos = classifier.objectWeights.argsort()[-3:]
    print(wpos)
    plt.figure(1)
    for i in range(1,4):
        plt.subplot('33' + str(i))
        plt.imshow(np.reshape(X[i,:],[8,8]), cmap='gray')
        plt.subplot('33' + str(i+3))
        plt.imshow(np.reshape(X[-i,:],[8,8]), cmap='gray')
        plt.subplot('33' + str(i+6))
        plt.imshow(np.reshape(X[wpos[i-1],:],[8,8]), cmap='gray')
    plt.show()

    

if __name__ == "__main__":
    main()