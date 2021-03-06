import numpy as np
import numpy.random as rnd
import math as m

def isclose(val1, val2, abs_tol=1e-12):
	return abs(val1-val2)<abs_tol

class MarkovDecisionProcess(): # VERANDER (newState, reward) naar transition (die als output (newState, reward) heeft)
	def __init__(self, States, Actions, newState, reward,discount, probabilities=None, AbsorbingStates=set()):
		self.States = States
		self.Actions = Actions
		self.newState = newState
		self.discount = discount
		self.reward = reward
		self.probabilities = probabilities 			# Matrix(States × States × Actions).
		self.AbsorbingStates = AbsorbingStates
		self.Q = self.initialize_Q()
		self.Qfunc = lambda s,a: self.Q[(s,a)] 		
		self.policy = None;

	def initialize_Q(self):
		self.Q = { (s,a): 0 for s in self.States for a in self.Actions }
		return self.Q

	def transition(self,sold,a):
		snew = self.newState(sold,a)
		return snew, self.reward(sold,snew)

	def getBest(self, s):
		Qvalues = [(a,self.Q[(s,a)]) for a in self.Actions]
		return sorted(Qvalues,key=lambda x: x[1])[-1]

	def getBestAction(self,s):
		return self.getBest(s)[0]

	def getBestQVal(self,s):
		return self.getBest(s)[1]

	def getOptimalPolicyFromQ(self):
		policy = { s: self.getBestAction(s) for s in self.States }
		return lambda s: policy[s]

	def getQMatrix(self):
		states, actions = list(self.States), list(self.Actions)
		Qmat = np.zeros((len(states), len(actions)))
		for i,s in enumerate(states):
			for j,a in enumerate(actions):
				Qmat[i,j] = self.Qfunc(s,a)
		return Qmat

	def QisEqual(self,Q1,Q2, tol=1e-6):
		isequal = { k: isclose(Q1[k],Q2[k], abs_tol=tol) for k, v in Q1.items() }
		return all(isequal.values())

	# Calculate expected value of function: S -> R,  w.r.t. random variable s2 (defined by initial state s and action a).
	def expectedValue(self,function,s,a):
		return sum([self.probabilities[(s,s2,a)]*function(s,s2,a) for s2 in self.States])

	# Perform 1 iteration of the q_iteration algorithm
	def q_iterate(self):
		func = lambda s,s2,a: self.reward(s,s2) + self.discount*self.getBestQVal(s2)
		return { k: self.expectedValue(func, *k) for k, v in self.Q.items() }

	def q_iteration(self,max_iter=100, tol=1e-6):
		if self.probabilities is None:
			raise('Transition probabilities not known.')
		self.initialize_Q()
		counter = 0
		while True:
			counter += 1
			#print(self.getQMatrix().T)
			Qnew = self.q_iterate()
			converged = self.QisEqual(self.Q, Qnew, tol)
			self.Q = Qnew
			if counter==max_iter or converged:
				break

	def getTemporalDifference(self, sold, a,snew, reward):
		return reward + self.discount*self.getBestQVal(snew) - self.Q[(sold, a)]

	def resetConvergece(self):
		return { (s,a): False for s in self.States-self.AbsorbingStates for a in self.Actions }

	def getRandomState(self):
		return rnd.choice(list(self.States-self.AbsorbingStates), 1)[0]

	def Q_Learning_iterate(self, eps, alpha, converged, tol=1e-6):
		s = self.getRandomState()
		a = self.getBestAction(s)
		if rnd.choice([True, False],size=1, p=[eps,1-eps]):
			a = rnd.choice(list(self.Actions), 1)[0]
		qold = self.Q[(s,a)]
		snew, reward = self.transition(s,a)
		TD = self.getTemporalDifference(s, a, snew, reward)
		qnew = qold + alpha*TD
		if isclose(qold, qnew, abs_tol=tol): # abs(TD)<tol
			converged[(s, a)] = True
		else:
			self.Q[(s, a)] = qnew
			converged = self.resetConvergece()
		return snew

	def Q_Learning(self, eps, alpha, decay_rate=0,max_iter=100, tol=1e-6, return_all=False):
		alphaDecayed =lambda n: alpha*m.exp(-decay_rate*n)
		converged = self.resetConvergece()
		self.initialize_Q()
		result = []
		for i in range(0,max_iter):
			if return_all:
				result.append(self.getQMatrix())
			self.Q_Learning_iterate(eps, alphaDecayed(i), converged, tol=tol)
			if all(converged.values()):
				break
		return result if return_all else self.Q
