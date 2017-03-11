import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd
import math as m

class DecisionStump():

    def __init__(self, theta=None, feature=None):
        self.theta = theta
        self.feature = feature
        self.rule = None
        self.accuracy = None
    
    def fit(self, X, y, w=None):
        if w is None:
            w = np.ones(X.shape[0])
        computeAccuracy = lambda x,rule: sum([w[i] if rule(xi)==y[i] else 0 for (i,xi) in enumerate(x)])/sum(w)
        rule1 = lambda theta: lambda xi: 1 if xi<theta else -1
        rule2 = lambda theta: lambda xi: -1 if xi<theta else 1
        rules=[rule1,rule2]
        key = lambda tup:tup[3]
        bestByRule = lambda f,rule: sorted([(x,rule(x),f,computeAccuracy(X[:,f],rule(x))) for x in X[:,f]],key=key)[-1] 
        bestByFeature = lambda f: sorted([bestByRule(f,rule) for rule in rules],key=key)[-1]
        self.theta, self.rule, self.feature, self.accuracy = sorted([bestByFeature(f) for f in range(0,X.shape[1])],key=key)[-1]

    def predict(self, X):
        return np.array([self.rule(x) for x in X[:,self.feature]])

class AdaBoostClassifier():

    def __init__(self, iterations, WeakLearner):
        self.WeakLearner = WeakLearner
        self.learners = []
        self.learnerWeights = []
        self.learnerErrors = []
        self.N = iterations

    def expectedLabels(self,X):
        return np.array([self.learnerWeights[l]*learner.predict(X) for l,learner in enumerate(self.learners)]).sum(axis=0)

    def predict(self,X):
        return [1 if e>0 else -1 for e in self.expectedLabels(X)]

    def calculateError(self, X, y, w):
        return sum([w[i] if fxi!=y[i] else 0 for i,fxi in enumerate(self.predict(X))])/sum(w) 

    def objectWeights(self, X, y):
        if len(self.learners)==0:
            return np.ones(X.shape[0])
        return [m.exp(-y[i]*self.expectedLabels(X)[0]) for i in range(0,X.shape[0])]

    def fit(self, X, y):
        self.learners, self.learnerWeights, self.learnerErrors = ([],[],[])
        done = False
        counter = 0
        while True:
            w = self.objectWeights(X,y)
            newLearner = self.WeakLearner()
            newLearner.fit(X,y,w=w)
            self.learners.append(newLearner)
            self.learnerWeights.append(1)
            error = self.calculateError(X,y,w)
            self.learnerWeights[-1] = (0.5*m.log(1/(error+1e-10)-1))
            self.learnerErrors.append(error)
            counter+=1
            if counter == self.N or error<1e-5:
                break

def generateData(N):
    X1 = rnd.multivariate_normal(mean=[0,0], cov=np.eye(2), size=N)
    X2 = rnd.multivariate_normal(mean=[2,0], cov=np.eye(2), size=N)
    X = np.concatenate((X1,X2), axis=0)
    y1 = np.ones(N)
    y2 = -np.ones(N)
    y = np.concatenate((y1,y2))
    return (X,y)

def plotDecisionBoundary(X,y,classiier):
     x0 = np.linspace(np.min(X[:,0]), np.max(X[:,0]),50)
     x1 = np.linspace(np.min(X[:,1]), np.max(X[:,1]),50)
     X = np.meshgrid(x0,x1)
     print(X)

def plotLearnerErrors(classifier):
    nIter = len(classifier.learnerErrors)
    plt.figure(1)
    iterations = np.arange(nIter)
    plt.subplot(1,2,1)
    plt.title('Error')
    plt.yscale('log')
    plt.plot(iterations,classifier.learnerErrors,'o')
    plt.subplot(1,2,2)
    plt.title('Weights')
    plt.plot(iterations,classifier.learnerWeights,'o')
    plt.show()

def plotDecisionStump(X,y,w):
    plt.figure(2)
    stump = DecisionStump()
    stump.fit(X,y,w=w)

    plt.scatter(X[0:50,0],X[0:50,1],marker='o')
    plt.scatter(X[50:100,0],X[50:100,1],marker='x')
    if stump.feature==0:
        plt.plot(stump.theta*np.ones(50),np.linspace(X[:,1].min(),X[:,1].max(),50))
    else:
        plt.plot(np.linspace(X[:,0].min(),X[:,0].max(),50),stump.theta*np.ones(50))
    plt.show()

def main():
    N=50
    X,y = generateData(N)
    w = np.ones(2*N)
    maxIter = 10
    classifier = AdaBoostClassifier(iterations=maxIter,WeakLearner=DecisionStump)
    classifier.fit(X,y)
    plotDecisionBoundary(X,y,classifier)
    #plotDecisionStump(X,y,w)
    #plotLearnerErrors(classifier)

if __name__ == "__main__":
    main()
