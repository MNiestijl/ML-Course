import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd
import math as m
from sklearn.model_selection import ShuffleSplit

class DecisionStump():
    def __init__(self, theta=None, feature=None):
        self.theta, self.feature, self.rule, self.accuracy = theta, feature, None, None

    def predict(self, X):
        return np.array([self.rule(x) for x in X[:,self.feature]])

    def score(self,X,y):
        return sum([1 if fxi==y[i] else 0 for i,fxi in enumerate(self.predict(X))])/y.shape[0]
    
    def fit(self, X, y, w=None):
        m,n = X.shape
        if w is None:
            w = np.ones(m)
        accuracy = lambda x,rule: sum([w[i] if rule(xi)==y[i] else 0 for (i,xi) in enumerate(x)])/sum(w)
        rule1 = lambda theta: lambda xi: 1 if xi<theta else -1 # Can be partially applied
        rule2 = lambda theta: lambda xi: -1 if xi<theta else 1
        rules=[rule1,rule2]
        key = lambda tup:tup[3]
        bestRule = lambda f,rule: sorted([(x,rule(x),f,accuracy(X[:,f],rule(x))) for x in X[:,f]],key=key)[-1] 
        bestFeature = lambda f: sorted([bestRule(f,rule) for rule in rules],key=key)[-1]
        self.theta, self.rule, self.feature, self.accuracy = sorted([bestFeature(f) for f in range(0,n)],key=key)[-1]

class AdaBoostClassifier():
    def __init__(self, iterations, WeakLearner):
        self.WeakLearner, self.learners, self.learnerWeights, self.errors, self.N = WeakLearner, [], [], [], iterations

    def expectedLabels(self,X):
        return np.array([self.learnerWeights[l]*learner.predict(X) for l,learner in enumerate(self.learners)]).sum(axis=0)

    def predict(self,X):
        return [1 if e>0 else -1 for e in self.expectedLabels(X)]

    def score(self, X, y):
        return sum([1 if fxi==y[i] else 0 for i,fxi in enumerate(self.predict(X))])/y.shape[0] 

    def fit(self, X, y):
        objectWeights = lambda: np.exp(-y*self.expectedLabels(X))
        self.learners, self.learnerWeights, self.errors, w = ([],[],[], np.ones(X.shape[0]))
        for i in range(0,self.N):
            newLearner = self.WeakLearner()
            newLearner.fit(X,y,w=w)
            error = (1-newLearner.accuracy)
            self.learnerWeights.append(0.5*m.log(1/(error+1e-10)-1))
            self.learners.append(newLearner)
            totalError = 1 - self.score(X,y)
            self.errors.append(totalError)
            w = objectWeights()
            if totalError<1e-5:
                break

def generateData(N):
    X1 = rnd.multivariate_normal(mean=[0,0], cov=np.eye(2), size=N)
    X2 = rnd.multivariate_normal(mean=[2,0], cov=np.eye(2), size=N)
    X = np.concatenate((X1,X2), axis=0)
    y1 = np.ones(N)
    y2 = -np.ones(N)
    y = np.concatenate((y1,y2))
    return X,y

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
    plt.plot(iterations,classifier.learnerWeights,'o')
    plt.show()

def plotDecisionStump(X,y,w):
    plt.figure(2)
    stump = DecisionStump()
    stump.fit(X,y,w=w)
    pos1, pos2 = np.where(y==-1), np.where(y==1)
    plt.scatter(X[pos1,0],X[pos2,1],marker='o')
    plt.scatter(X[pos2,0],X[pos2,1],marker='x')
    if stump.feature==0:
        plt.plot(stump.theta*np.ones(50),np.linspace(X[:,1].min(),X[:,1].max(),50))
    else:
        plt.plot(np.linspace(X[:,0].min(),X[:,0].max(),50),stump.theta*np.ones(50))
    plt.show()

def cross_validate(classifier,X,y,n_splits,train_size):
    splits = ShuffleSplit(n_splits=n_splits, train_size=train_size).split(X)
    scores = []
    for train_indices, test_indices in splits:
        estimator = DecisionStump()
        estimator.fit(X[train_indices,:], y[train_indices])
        scores.append(estimator.score(X[test_indices,:],y[test_indices]))
    return np.array(scores)

def testDecisionStump(X,y):
    stump = DecisionStump()
    indices = np.arange(len(y)),
    mask = np.zeros(len(y), dtype=bool)
    mask[np.concatenate((np.where(y==-1)[0][0:50],np.where(y==1)[0][0:50]))]=True
    stump.fit(X[mask,:],y[mask])
    first = stump.score(X[~mask,:],y[~mask])
    scores = cross_validate(DecisionStump,X,y,n_splits=10, train_size=100)
    print(scores)
    print("Using first 50 of both classes as training data:\nscore: {}\n".format(first))
    print("Decision stump accuracy with random subsets:\nmean: {}, std: {}\n".format(scores.mean(),scores.std()))

def testAdaBoostClassifier(X,y, WeakLearner):
    maxIter = 100
    classifier = AdaBoostClassifier(iterations=maxIter,WeakLearner=DecisionStump)
    classifier.fit(X,y)
    print(classifier.errors[-1])
    plotErrors(classifier)

def main():
    filename = 'optdigitsubset.txt'
    #X = pd.read_csv(filename, delim_whitespace=True, header=None).as_matrix()
    #y = np.concatenate((-np.ones(554),np.ones(571)),axis=0)
    X,y = generateData(50)
    testAdaBoostClassifier(X,y,WeakLearner=DecisionStump)    
    #testDecisionStump(X,y)

if __name__ == "__main__":
    main()
