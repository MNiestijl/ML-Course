import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import math as m
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
from MarkovDecisionProcess import MarkovDecisionProcess
from collections import OrderedDict

# SETTINGS
failureProb = 0

def newState(s,a):
    if rnd.choice([True, False],size=1, p=[failureProb,1-failureProb]):
        return s
    return s if s==1 or s==6 else s+a

def reward(sold,snew):
    if sold!=6 and snew==6:
        return 5
    elif sold!= 1 and snew==1:
        return 1
    else:
        return 0

def transitionProb(s1,s2,a):
    if s1==1 or s1==6:
        return 1 if s2==s1 else 0
    elif s2==s1+a:
        return 1-failureProb
    elif s2==s1:
        return failureProb
    else:
        return 0

def getMDP():
    # Definition of MDP
    States = {1,2,3,4,5,6}
    AbsorbingStates = {1,6}
    Actions = OrderedDict.fromkeys([-1,1]) # Use OrderedDict instead of set so that the order stays the same
    discount = 0.5
    probabilities = { (s1,s2,a): transitionProb(s1,s2,a) for s1 in States for s2 in States for a in Actions }
    return MarkovDecisionProcess(States, Actions, newState, reward, discount, probabilities, AbsorbingStates=AbsorbingStates)

def plotQError():

    # Settings
    #eps_values = [0.2,0.4, 0.7]
    #alpha_values = [0.2, 0.5, 0.7]
    eps_values = [0.3]
    alpha_values = [0.4]
    repeats = 10 # NOG IMPLEMENTEREN!! zet TOL onredelijk laag (1e-40) en cap door max_iter, zodat ze allemaal dezelfde lengte hebben.
    tol = 1e-12
    n_plots = len(eps_values) * len(alpha_values)

    MDP = getMDP()
    MDP.q_iteration(max_iter=100, tol=tol)
    QTrue = MDP.getQMatrix()
    for i,eps in enumerate(eps_values):
        for j,alpha in enumerate(alpha_values):
            Qs = MDP.Q_Learning(eps=eps, alpha=alpha, max_iter=1000, tol=tol, return_all=True)
            error = [ la.norm(QTrue - Qs[i], ord='fro') for i in range(0,len(Qs))]
            fig = plt.figure(1)
            ax = fig.add_subplot(len(eps_values),len(alpha_values),i+1+3*j)
            ax.plot(np.arange(0,len(Qs)), error)
            ax.set_yscale('log')
            #ax.set_xlabel('Iteration')
            #ax.set_ylabel('Error')
            ax.set_title('eps={}, alpha={}'.format(eps, alpha))
    plt.show()

def main():
    plotQError()
    """
    MDP = getMDP()
    MDP.q_iteration(max_iter=100, tol=1e-6)
    Q1 = MDP.getQMatrix().T
    MDP.Q_Learning(eps=0.3, alpha=0.5, max_iter=1000, tol=1e-6)
    Q2 = MDP.getQMatrix().T

    # print results
    policy = MDP.getOptimalPolicyFromQ()
    actions = [policy(s) for s in States]
    print(Q1)
    print(Q2)
    print(actions)
    """

if __name__ == "__main__":
    main()
