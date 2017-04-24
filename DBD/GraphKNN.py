import numpy as np
import networkx as nx

"""
GraphKNN implements the k-nearest neighbors classifier on a given connected weighted graph. The weights of the graph are used as distance.
"""
class GraphKNN():

	def __init__(self,graph,k_neighbors=3, weights='uniform'):
		if not nx.is_connected(graph):
			raise('Graph should be connected.')
		if nx.number_of_nodes(graph)<k_neighbors:
			raise("k_neighbors larger than the number of nodes in the graph")
		self.graph, self.k_neighbors, self.weights = graph, k_neighbors, weights

	def getKNN(self, node):
		return self.getKNNfun([node], visited={node:0})

	def getKNNfun(self, nodes, visited={}):
		edges = self.graph.edges(nodes, data=['weight','labeled'])
		edges = [edge for edge in edges if edge[1] not in visited and edge[3] is True]
		newNodes = list(set([edge[1] for edge in edges]))
		newWeightsLists = [[visited[edge[0]] + edge[2] for edge in edges if edge[1]==n] for n in newNodes]
		newWeights = [sorted(newWeightsLists[i])[0] for i in range(0,len(newNodes))]
		for i,n in enumerate(newNodes):
			visited[n] = newWeights[i]
		nNodesVisited = len(list(visited.keys()))
		if nNodesVisited>=self.k_neighbors:
			nodes = list(visited.keys())
			sortedResult = sorted([(n,visited[n]) for n in nodes], key=lambda tup:tup[1])
			return sortedResult[1:self.k_neighbors]
		return self.getKNNfun(newNodes, visited)

	def getPrediction(self, neighbors):
		pass

	def fit(self, X, y=None):
		pass

	def score(self, X,y):
		pass