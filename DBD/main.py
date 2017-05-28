import numpy as np 
from DBD import *
from generateData import *
from GraphKNN import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy.random as rnd
import pandas as pd
from sklearn.neighbors import DistanceMetric, NearestNeighbors, KNeighborsClassifier

def plot_data(X,y):
    C1, C2, Cunl = np.where(y==1)[0], np.where(y==0)[0], np.where(y==-1)[0]
    plt.scatter(X[C1,0],X[C1,1], marker='o', c='blue', s=40)
    plt.scatter(X[C2,0],X[C2,1], marker='x', c='red', s=40)
    plt.scatter(X[Cunl,0],X[Cunl,1], marker='.', c='grey')

def plot_shortest_path(X,dbd,node1,node2):
    path = dbd.shortest_path(node1,node2)
    data = X[path,:]
    plt.plot(data[:,0],data[:,1],linewidth=2)

def custom():
    # Settings
    Nlab, Nunl = 10, 200
    p = 0.5
    d = 2 # dimension of feature space
    s = 2 # Smoothness assumption
    kernel = 'gaussian'
    g = None
    alpha = None
    N = Nlab+Nunl
    eps = N**(-(1/(2*d)))
    h = 1/(N**(1/(s+d)))

    # Data generators
    gen1=circularGenerator(5, 0.1, bias=[1,-3], angle_range=(0,2/3*m.pi))
    gen2=circularGenerator(5, 0.1, bias=[-1,3], angle_range=(m.pi,5/3*m.pi))

    # generate data
    X,y = generatePartialData(gen1,gen2,Nlab,Nunl,p=p)
    #y = -np.ones(N) # Only care about shape right now!
    scale(X, axis=0, with_mean=False,copy=False)

    # get distance object
    dbd = DBD(X,h, y=y,g=g,alpha=alpha,eps=eps,kernel=kernel)
    knn = GraphKNN(dbd, k_neighbors=1, algorithm='brute')
    print("Predicting data using DBD KNN")
    y_dbdKNN = knn.predict(np.arange(0,len(y)))

    # Plot data
    plt.figure(1)
    """
    for i in range(0,3):
        plot_shortest_path(X,dbd,rnd.randint(0,N),rnd.randint(0,N))
    """
    plt.subplot(131)
    plot_data(X,y)
    plt.subplot(132)
    plot_data(X,y_dbdKNN)
    plt.show()

def useMNIST(dir_path):

    # Settings
    d = 2 # dimension of feature space
    s = 2 # Smoothness assumption
    kernel = 'gaussian'
    g = None
    alpha = None
    eps = N**(-(1/(2*d)))
    h = 1/(N**(1/(s+d)))
    #h=1

    # Load Data
    print("Loading Data")
    train_file = dir_path + "train.csv"
    test_file = dir_path + "test.csv"
    train = pd.read_csv(train_file, sep=',').as_matrix()
    test = pd.read_csv(test_file, sep=',').as_matrix()
    X_train = train[:,1:]
    X_test = test[:,1:]
    y_train = train[:,0]
    y_test = test[:,0]


    # Pre-process data
    # NVT

    # Get distance object
    print("Computing metric")
    dbd = DBD(X_train,h, g=g,alpha=alpha,eps=eps,kernel=kernel)
    pdfmax = dbd.pdfx.max()
    print("Maximum value of pdf(x): {}".format(pdfmax))
    print("Minimum value of g(x): {}".format(dbd.g(pdfmax)))

    # KNN 

    # Score on test set:


def main():
    dir_path = 'C:/Users/Milan Niestijl/Documents/datasets/ML/MNIST/'
    custom()
    #useMNIST(dir_path)
    
if __name__=="__main__":
    main()