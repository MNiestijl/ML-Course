import numpy as np
import numpy.random as rnd
import math as m

"""
Discrete Time Markov Decision Process with continuous states and finite actions. 
Solve using value function approximation.
"""
class ValueFunctionApprox(): # VERANDER (newState, reward) naar transition (die als output (newState, reward) heeft)
	def __init__(self, getRandomState, Actions, newState, reward, discount, actFuncs):
		self.getRandomState = getRandomState
		self.Actions = Actions
		self.newState = newState
		self.discount = discount
		self.reward = reward	
		self.policy = None
		self.actFuncs = actFuncs # list of functions S×A -> Reals
		self.W = self.initialize_W() # List of weights for each action a.

	def initialize_W(self):
		self.W = { a: np.ones(len(self.actFuncs)) for a in self.Actions }
		return self.W

	def transition(self,sold,a):
		snew = self.newState(sold,a)
		return snew, self.reward(sold,snew)

	def evalActFuncs(self, s):
		evaluated = np.array([fun(s) for fun in self.actFuncs])
		normalized = evaluated/sum(evaluated)
		return normalized

	def evaluate(self,s,a):
		return np.dot(self.W[a], self.evalActFuncs(s))

	def getMaxVal(self,s):
		values = [(a,self.evaluate(s,a)) for a in self.Actions]
		return sorted(values, key=lambda x: x[1])[-1]

	def getBestAction(self,s):
		return self.getMaxVal(s)[0]

	def getBestQVal(self,s):
		return self.getMaxVal(s)[1]

	def getTemporalDifference(self, sold, a,snew, reward):
		return reward + self.discount*self.getBestQVal(snew) - self.evaluate(sold,a)

	def Q_Learning_iterate(self, eps, alpha):
		s = self.getRandomState()
		a = self.getBestAction(s)
		if rnd.choice([True, False],size=1, p=[eps,1-eps]):
			a = rnd.choice(list(self.Actions), 1)[0]
		snew, reward = self.transition(s,a)
		TD = self.getTemporalDifference(s,a,snew,reward)
		self.W[a] = self.W[a] + alpha*TD*self.evalActFuncs(s)

	def Q_Learning(self, eps, alpha, max_iter=100):
		self.initialize_W()
		for i in range(0,max_iter):
			self.Q_Learning_iterate(eps, alpha)
		# return self.W