import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd

class DecisionStump():

    def __init__(self, theta=None, feature=None):
        self.feature = theta
        self.theta = feature 
        self.rule = None

    def get_accuracy(self,x,y,rule):
        return sum([1 if rule(xi)==y[i] else 0 for (i,xi) in enumerate(x) ])/len(x)

    def search_best(self,xs,y,rule):
        dtype = [('feature value', float), ('accuracy', float)]
        values = [(x, self.get_accuracy(xs,y,rule(x))) for x in xs]
        return np.sort(np.array(values, dtype=dtype), order='accuracy')[-1]
    
    def fit(self, X, y):
        rule1 = lambda theta: lambda xi: 1 if xi<theta else 0
        rule2 = lambda theta: lambda xi: 0 if xi<theta else 1
        rules=[rule1,rule2]
        feature, theta, accuracy,rule = (0,0,0,0)
        # WORK IN PROGRESS, becomes much shorter!
        #bests = [(self.search_best(X[:,f],y,rule1),self.search_best(X[:,f],y,rule2)) for f in range(0,X.shape[1]) ]
        #print(bests)
        for f in range(0,X.shape[1]):
            best1 = self.search_best(X[:,f],y,rule1)
            best2 = self.search_best(X[:,f],y,rule2)
            if (max((best1[1],best2[1]))>accuracy):
                feature, theta, accuracy,rule = \
                   (f, best1[0],best1[1],0) if best1[1]>best2[1] else (f, best2[0],best2[1],1)          
        self.theta = theta
        self.feature = feature
        self.rule = rules[rule]
 

def main():
    N = 50
    X1 = rnd.multivariate_normal(mean=[0,0], cov=np.eye(2), size=N)
    X2 = rnd.multivariate_normal(mean=[2,0], cov=np.eye(2), size=N)
    X = np.concatenate((X1,X2), axis=0)
    y1 = np.ones(N)
    y2 = np.zeros(N)
    y = np.concatenate((y1,y2))
    stump = DecisionStump()
    stump.fit(X,y)
    plt.scatter(X1[:,0],X1[:,1],marker='o')
    plt.scatter(X2[:,0],X2[:,1],marker='x')
    print(stump.theta)
    if stump.feature==0:
        plt.plot(stump.theta*np.ones(50),np.linspace(X[:,1].min(),X[:,1].max(),50))
    else:
        plt.plot(np.linspace(X[:,0].min(),X[:,0].max(),50),stump.theta*np.ones(50))
    plt.show()

if __name__ == "__main__":
    main()
