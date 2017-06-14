import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import math as m
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli
from MarkovDecisionProcess import MarkovDecisionProcess
from ValueFunctionApprox import ValueFunctionApprox
from collections import OrderedDict
import scipy.integrate as integrate

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

def newState2(s,a):
    return s if s<1.5 or s>5.5 else s+a+rnd.normal(0, 0.01,1)

def reward2(sold, snew):
    if sold<=5.5 and snew>5.5:
        return 5
    elif sold>=1.5 and snew<1.5:
        return 1
    else:
        return 0

def transition2(sold,a):
    snew = newState2(sold,a)
    return snew, reward2(sold,snew)

def getMDP(discount=0.5):
    # Definition of MDP
    States = {1,2,3,4,5,6}
    AbsorbingStates = {1,6}
    Actions = OrderedDict.fromkeys([-1,1]) # Use OrderedDict instead of set so that the order stays the same
    probabilities = { (s1,s2,a): transitionProb(s1,s2,a) for s1 in States for s2 in States for a in Actions }
    return MarkovDecisionProcess(States, Actions, newState, reward, discount, probabilities=probabilities, AbsorbingStates=AbsorbingStates)
    
def getVFA(discount, nActFuncs):
    getRandomState = lambda : rnd.uniform(0, 6, size=1)[0]
    Actions = {1, -1}
    width = 6/nActFuncs
    rbf = lambda mean: lambda x: m.exp(-(x-mean)**2/(2*width**2))/m.sqrt(2*m.pi)
    actFuncs = [ rbf(mean) for mean in np.linspace(1,6,nActFuncs) ]
    return ValueFunctionApprox(getRandomState, Actions, newState2, reward2, discount, actFuncs)

def playGame(discount, policy):
    state = rnd.uniform(1.5, 5.5, size=1)[0]
    gameOver = lambda s: s<1.5 or s>5.5
    reward = 0
    count = 0
    while not gameOver(state):
        action = policy(state)
        state, r = transition2(state, action)
        reward += r*discount**count
        count += 1
    return reward


def plotReward(discount=0.5, nActFuncs=100):
    eps = 0.7
    alpha = 0.3
    VFA = getVFA(discount=discount, nActFuncs=nActFuncs)
    nSteps = 100
    step_size = 100
    xs = np.linspace(1,6,100)
    reward = np.zeros(nSteps)
    for i in range(0,nSteps):
        for j in range(0,step_size):
            VFA.Q_Learning_iterate(eps, alpha)
        # Approximate Expected Return using numerical integration.
        y = np.array([ 1/5*VFA.getBestQVal(x) for x in xs ])
        reward[i] = np.trapz(y,xs) 
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    iterations = np.cumsum(step_size*np.ones(nSteps))
    ax.plot(iterations, reward)
    ax.set_title('Reward for various number of iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Reward')
    plt.show()


def plotApproxQ(discount=0.5, nActFuncs=100):
    eps = 0.7
    alpha = 0.3
    VFA = getVFA(discount=discount, nActFuncs=nActFuncs)
    VFA.Q_Learning(eps, alpha, max_iter=10000)
    xs = np.linspace(1,6,1000)
    y1 = [ VFA.evaluate(x,-1) for x in xs ]
    y2 = [ VFA.evaluate(x,1) for x in xs ]
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.plot(xs, y1)
    ax.plot(xs, y2)
    ax.set_title('Approximated Q-function for different actions')
    ax.set_xlabel('x')
    ax.set_ylabel('Q')
    ax.legend(['left', 'right'])
    plt.show()


def plotQError():

    # Settings
    eps_values = [0.2,0.4, 0.7]
    alpha_values = [0.2, 0.5, 0.7]
    #eps_values = [0.3]
    #alpha_values = [0.4]
    repeats = 10 # NOG IMPLEMENTEREN!! zet TOL onredelijk laag (1e-40) en cap door max_iter, zodat ze allemaal dezelfde lengte hebben.
    tol = 1e-12
    n_plots = len(eps_values) * len(alpha_values)

    MDP = getMDP(discount=0.5)
    MDP.q_iteration(max_iter=1000, tol=1e-15)
    QTrue = MDP.getQMatrix()
    for i,eps in enumerate(eps_values):
        for j,alpha in enumerate(alpha_values):
            fig = plt.figure(1)
            ax = fig.add_subplot(len(eps_values),len(alpha_values),i+1+3*j)
            ax.set_yscale('log')
            #ax.set_xlabel('Iteration')
            #ax.set_ylabel('Error')
            ax.set_title('eps={}, alpha={}'.format(eps, alpha))
            for k in range(0,repeats):
                Qs = MDP.Q_Learning(eps=eps, alpha=alpha, max_iter=1000, tol=tol, return_all=True)
                error = [ la.norm(QTrue - Qs[i], ord='fro') for i in range(0,len(Qs))]
                ax.plot(np.arange(0,len(Qs)), error)
    plt.show()

def printQTables(gamma_values):
    for gamma in gamma_values:
        MDP = getMDP(discount=gamma)
        MDP.q_iteration(max_iter=100, tol=1e-6)
        print(MDP.getQMatrix().T)

def main():
    plotQError()
    #plotApproxQ(discount=0.9, nActFuncs=100)
    #plotReward(discount=0.5, nActFuncs=100)
    #printQTables(gamma_values=[0,0.1, 0.9, 1])
    

if __name__ == "__main__":
    main()
