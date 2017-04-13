import numpy as np
import numpy.linalg as nl
import math as m
import networkx as nx
import numbers
from scipy.stats import norm, rv_continuous
from sklearn.neighbors import KernelDensity

def makeGraph(X,weightFunc):
    n,d = X.shape
    graph = nx.Graph()
    graph.add_nodes_from([(i,dict(values=X[i,:])) for i in range(0,n)])
    edges = [(i,j,weightFunc(i,j)) for i in range(0,n) for j in range(0,n)]
    edges = [tup for tup in edges if tup[2]!=-1]
    graph.add_weighted_edges_from(edges,weight='weight')
    return graph

class DBD():

    def __init__(self,X,h, g=None,alpha=None,eps=None,kernel='normal'):
        """
        X:      data ~ (n_samples × d_features)
        h:      bandwith parameter.
        g:      Decreasing and positive function, see paper.
        alpha:  Parameter>0
        eps:    parameter>0
        kernel: 1-d kernel function.

        available kernels: 
            - normal
            TODO
            - epanechnikov
            - sigmoid
        """
        n,d = X.shape
        self.h, self.kernel= h,kernel

        print("Estimating density")
        kde = KernelDensity(kernel=kernel, bandwidth=h, algorithm='kd_tree',rtol=1e-4).fit(X)
        self.pdf = lambda data: np.exp(kde.score_samples(data))
        self.pdfx = self.pdf(X)

        self.alpha = alpha if alpha is not None else self.pdfx.min()
        self.eps = eps if eps else n**(-1/(2*d))
        a,b = self.pdfx.min(), self.pdfx.max()
        self.g = g if g else lambda x: 1 if x<self.alpha else m.exp((a-x)/(b-a))
        print("eps: {},\nalpha: {}".format(self.eps,self.alpha))

        self.graph = self.makeDBDGraph(X)
        print("Initialized")

    def metric(self,node1,node2):
        return nx.shortest_path_length(self.graph,node1,node2,weight='weight')

    def shortest_path(self,node1,node2):
        return nx.shortest_path(self.graph,node1,node2,weight='weight')

    def updatePartition(self,tup,X,node):
        x = X[node,:]
        locations, components = tup
        for i,component in enumerate(components):
            if np.any([nl.norm(x-y)<self.eps for y in component]):
                component.append(x)
                locations[node]=i
                return
        if self.pdfx[node]>=self.alpha:
            components.append([x])
            locations[node]=len(components)+1

    def partitionData(self,X):
        # returns (component of node i, partition as a list of lists)
        n,d = X.shape
        tup = (np.zeros(n),[])
        mask = np.zeros(n, dtype=bool)
        mask[np.where([self.pdfx[i]>=self.alpha for i in range(0,n)])[0]] = True
        inds = np.arange(0,n)
        X1 = X[mask,:]  # pdf(x)>=alpha
        X2 = X[~mask,:] # pdf(x)<alpha
        for i in range(0,X1.shape[0]): 
            self.updatePartition(tup,X,inds[mask][i])
        for i in range(0,X2.shape[0]):
            self.updatePartition(tup,X,inds[~mask][i])
        return tup


    def computeWeight(self,X,node1,node2,partition):
        """
        Return -1 if the weight is infinite.
        """
        xi, xj = X[node1,:], X[node2,:]
        L2dist = nl.norm(xi-xj)
        loc1,loc2 = partition[0][node1],partition[0][node2]
        if (L2dist<self.eps):
            avg = (1/2*(xi+xj)).reshape(1,-1)
            return self.g(self.pdf(avg))*L2dist
        elif loc1==-1 or loc2==-1:
            raise('Both points must be in the partition')
        elif loc1!=loc2:
            return L2dist
        else:
            return -1

    def makeDBDGraph(self,X): 
        print("Making partition") 
        partition = self.partitionData(X)
        print("Number of components: {}".format(len(partition[1])))
        weightFunc = lambda node1,node2: self.computeWeight(X,node1,node2,partition)
        print("Making graph")
        return makeGraph(X,weightFunc)
    