import numpy as np
import math as m

class DecisionStump():
    def __init__(self, theta=None, feature=None, rule=None):
        self.theta, self.feature, self.rule, self.accuracy = theta, feature, rule, None

    def predict(self, X):
        return np.array([self.rule(x) for x in X[:,self.feature]])

    def score(self,X,y):
        return sum([1 if fxi==y[i] else 0 for i,fxi in enumerate(self.predict(X))])/y.shape[0]
    
    def fit(self, X, y, w=None):
        m,n = X.shape
        w = np.ones(m) if w is None else w
        accuracy = lambda x,rule: sum([w[i] if rule(xi)==y[i] else 0 for (i,xi) in enumerate(x)])/sum(w)
        rule1 = lambda theta: lambda xi: 1 if xi<theta else -1 # Can be partially applied
        rule2 = lambda theta: lambda xi: -1 if xi<theta else 1
        rules=[rule1,rule2]
        results = [(x,rule(x),f,accuracy(X[:,f],rule(x))) for f in range(0,n) for x in X[:,f] for rule in rules]
        self.theta, self.rule, self.feature, self.accuracy = sorted(results,key=lambda tup:tup[3])[-1]

class AdaBoostClassifier():
    def __init__(self, iterations, WeakLearner):
        self.WeakLearner, self.learners, self.weights, self.errors, self.N = WeakLearner, [], [], [], iterations
        self.objectWeights  = []

    def expectedLabels(self,X):
        return np.array([self.weights[l]*learner.predict(X) for l,learner in enumerate(self.learners)]).sum(axis=0)

    def predict(self,X):
        return np.array([1 if e>0 else -1 for e in self.expectedLabels(X)])

    def score(self, X, y):
        return sum([1 if fxi==y[i] else 0 for i,fxi in enumerate(self.predict(X))])/y.shape[0] 

    def fit(self, X, y):
        objectWeights = lambda: np.exp(-y*self.expectedLabels(X))
        self.learners, self.weights, self.errors, w = ([],[],[], np.ones(X.shape[0]))
        for i in range(0,self.N):
            newLearner = self.WeakLearner()
            newLearner.fit(X,y,w=w)
            error = (1-newLearner.accuracy)
            self.weights.append(0.5*m.log(1/(error+1e-10)-1))
            self.learners.append(newLearner)
            totalError = 1 - self.score(X,y)
            self.errors.append(totalError)
            w = objectWeights()
            if totalError<1e-5:
                break
        self.objectWeights = w # Only store object weights of last iteration

