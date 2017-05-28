import numpy as np
import networkx as nx

"""
GraphKNN implements the k-nearest neighbors classifier on a given connected weighted graph. The weights of the graph are used as distance.
"""
class GraphKNN():

	def __init__(self,dbd,k_neighbors=3, weights='uniform', algorithm='brute'):
		if not nx.is_connected(dbd.graph):
			raise('Graph should be connected.')
		if nx.number_of_nodes(dbd.graph)<k_neighbors:
			raise("k_neighbors larger than the number of nodes in the graph")
		self.dbd, self.k_neighbors, self.weights, self.algorithm = dbd, k_neighbors, weights, algorithm

	"""
	Calculate the k nearest neighbors and their distances. Returns k (node, distance) tuples.
	"""
	def getKNN(self,node):
		if self.algorithm=='brute':	
			labeledNodes = self.dbd.labeledGraph.nodes(data=True)
			distances = [(node2[0], self.dbd.metric(node,node2[0])) for node2 in labeledNodes]
			distances.sort(key=lambda tup: tup[1])
			return distances[:self.k_neighbors]

		elif self.algorithm=='expand':
			return self.getKNNfun([node],visited={node:0})

	def getKNNfun(self, nodes, visited={}):
		edges = self.dbd.graph.edges(nodes, data='weight')
		edges = [edge for edge in edges if edge[1] not in visited]
		newNodes = list(set([edge[1] for edge in edges]))
		newWeightsLists = [[visited[edge[0]] + edge[2] for edge in edges if edge[1]==n] for n in newNodes]
		newWeights = [sorted(newWeightsLists[i])[0] for i in range(0,len(newNodes))]
		for i,n in enumerate(newNodes):
			visited[n] = newWeights[i]
		nNodesVisited = len(list(visited.keys()))
		if nNodesVisited>=self.k_neighbors:
			nodes = list(visited.keys())
			sortedResult = sorted([(n,visited[n]) for n in nodes], key=lambda tup:tup[1])
			return sortedResult[:self.k_neighbors]
		return self.getKNNfun(newNodes, visited)

	def getPrediction(self, neighbors):
		if self.weights=='uniform':
			neighborLabels = [self.dbd.labeledGraph.node[n[0]]['label'] for n in neighbors]
			labels = set(neighborLabels)
			votes = [(label, neighborLabels.count(label)) for label in labels]
			votes.sort(key=lambda vote: vote[1])
			return votes[-1][0]
		else:
			pass #TODO

	def predictNode(self, node):
		# Node might not have attribute 'label'
		try:
			if self.dbd.graph[node]['label']!=-1:
				return self.dbd.graph[node]['label']
		except: 
			neighbors = self.getKNN(node)
			return self.getPrediction(neighbors)
		

	def predict(self, nodes):
		return np.array([self.predictNode(node) for node in nodes])

	def score(self, nodes,labels):
		n = len(nodes)
		predictions = self.predict(nodes)
		return sum([1 if nodes[i]==labels[i] else 0 for i in range(0,n)])/n