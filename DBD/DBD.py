import numpy as np
import numpy.linalg as nl
import math as m
import networkx as nx
import numbers
from scipy.stats import norm, rv_continuous
from sklearn.neighbors import KernelDensity

def makeDBDGraph(X,getEdge, labels=None, node_indices=None):
    n,d = X.shape
    node_indices = node_indices if node_indices is not None else np.arange(0,n)
    if labels is None:
        nodes = [(node_indices[i],dict(values=X[i,:])) for i in range(0,n)]
    else:    
        nodes = [(node_indices[i],dict(values=X[i,:],label=int(labels[i]))) for i in range(0,n)]
    edges = [(node_indices[i],node_indices[j],getEdge(node_indices[i],node_indices[j])) for i in range(0,n) for j in range(0,n)]
    edges = [tup for tup in edges if tup[2]!=-1]
    return makeWeightedGraph(nodes, edges)

def makeWeightedGraph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph

class DBD():

    def __init__(self,X,h,y=None, g=None,alpha=None,eps=None,kernel='normal'):
        """
        X:      data ~ (n_samples × d_features)
        y:      Label vector. Use -1 for unlabeled samples, 0, 1, ... for labels.
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
        self.g = g if g else lambda x: 1 if x<self.alpha else m.exp(-(x-self.pdfx.min())/(2*self.pdfx.std()))

        print("eps: {},\nalpha: {}".format(self.eps,self.alpha))

        self.graph = self.getGraph(X)
        if y is not None:
            self.labeledGraph = self.getLabeledGraph(X,y)
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
        locations = -np.ones(n,dtype=int)
        components = []
        tup = (locations, components)
        mask = np.zeros(n, dtype=bool)
        # pdf(x)>=alpha
        mask[np.where([self.pdfx[i]>=self.alpha for i in range(0,n)])[0]] = True
        inds = np.arange(0,n)
        inds1, inds2 = inds[mask], inds[~mask]
        X1, X2 = X[mask,:], X[~mask,:]

        n_components=0
        for i in range(0,len(X1)):
            if locations[i]==-1:
                components.append([X[i,:]])
                locations[i]=n_components
                n_components+=1
            for j in range(0,len(X1)):
                if nl.norm(X[i,:]-X[j,:])>=self.eps or i==j or locations[i]==locations[j]:
                    continue
                # Add xj to component
                if locations[j]==-1:
                    components[locations[i]].append(X[j,:])
                    locations[j]=locations[i]
                # Merge components of xi and xj
                else:
                    a = min(locations[i],locations[j])
                    b = max(locations[i],locations[j])
                    components[a] += components[b]
                    del components[b]
                    locations[np.where(locations==b)[0]] = a
                    for k in range(b+1,n_components):
                        locations[np.where(locations==k)[0]] = k-1
                    n_components-=1

        for i in range(0,X2.shape[0]):
            self.updatePartition(tup,X,inds[~mask][i])
        return tup


    def computeWeight(self,X,node1,node2,partition):
        """
        Return -1 if the weight is infinite.
        """
        if node1==node2: 
            return -1
        xi, xj = X[node1,:], X[node2,:]
        L2dist = nl.norm(xi-xj)
        loc1,loc2 = partition[0][node1],partition[0][node2]
        if loc1==-1 or loc2==-1:
            raise('Both points must be in the partition')
        elif (L2dist<self.eps):
            avg = (1/2*(xi+xj)).reshape(1,-1)
            return self.g(self.pdf(avg))*L2dist
        elif loc1!=loc2:
            return L2dist
        else:
            return -1

    def getGraph(self,X,y=None): 
        print("Making partition") 
        partition = self.partitionData(X)
        print("Number of components: {}".format(len(partition[1])))
        weightFunc = lambda node1,node2: self.computeWeight(X,node1,node2,partition)
        print("Making graph")
        return makeDBDGraph(X,weightFunc)

    def getLabeledGraph(self,X,y):
        print("Making labeled graph")
        n,d = X.shape
        inds = np.where(y!=-1)[0]
        weightFunc = lambda node1,node2: self.metric(node1,node2)
        return makeDBDGraph(X[inds],weightFunc,labels=y[inds],node_indices=inds)
    