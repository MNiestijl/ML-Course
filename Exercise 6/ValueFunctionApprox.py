import numpy as np
import numpy.random as rnd
import math as m

"""
Discrete Time Markov Decision Process with continuous states and finite actions. 
Solve using value function approximation.
"""
class ValueFunctionApprox(): # VERANDER (newState, reward) naar transition (die als output (newState, reward) heeft)
	def __init__(self, Actions, newState, reward, discount, actFuncs):
		self.Actions = Actions
		self.newState = newState
		self.discount = discount
		self.reward = reward	
		self.policy = None
		self.actFuncs = actFuncs
		self.nFuncs = len(actFuncs)
		self.W = { a: np.ones(nfuncs) for a in Actions } # List of weights for each action a.

	def evaluate(self,s,a):
		return np.multiply(self.W[a], [f(s) for f in self.actFuncs])

	def getMaxVal(s):
		values = [(a,evaluate(s,a)) for a in self.Actions]
		return sorted(Qvalues,key=lambda x: x[1])[-1]

	def getBestAction(self,s):
		return self.getMaxVal(s)[0]

	def getBestQVal(self,s):
		return self.getMaxVal(s)[1]

	def getTemporalDifference(self, sold, a,snew, reward):
		return reward + self.discount*self.getBestQVal(snew) - self.evaluate(s,a)