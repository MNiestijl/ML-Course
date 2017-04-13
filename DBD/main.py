import numpy as np 
from DBD import *
from generateData import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def plot_data(X,y):
    C1, C2, Cunl = np.where(y==1)[0], np.where(y==0)[0], np.where(y==-1)[0]
    plt.scatter(X[C1,0],X[C1,1], marker='o', c='blue', s=40)
    plt.scatter(X[C2,0],X[C2,1], marker='x', c='red', s=40)
    plt.scatter(X[Cunl,0],X[Cunl,1], marker='.', c='grey')

def plot_shortest_path(X,dbd,node1,node2):
    path = dbd.shortest_path(node1,node2)
    data = X[path,:]
    plt.plot(data[:,0],data[:,1],linewidth=2)

def main():

    # Settings
    N1,N2, Nunl = 10, 10, 1000
    N = N1+N2+Nunl
    d = 2 # dimension of feature space
    s = 2 # Smoothness assumption
    kernel = 'gaussian'
    g = None
    alpha = None
    eps = N**(-(1/(2*d)))
    h = 1/(N**(1/(s+d)))
    #h=1

    # Data generators
    gen1=circularGenerator(5, 0.5, angle_range=(0,2/3*m.pi))
    gen2=gen1

    # generate data
    X,y = generateData(gen1,gen2,N=N)
    y = -np.ones(N) # Only care about shape right now!
    scale(X, axis=0, with_mean=False,copy=False)

    # get distance object
    dbd = DBD(X,h, g=g,alpha=alpha,eps=eps,kernel=kernel)
    pdfmax = dbd.pdfx.max()
    print("Maximum value of pdf(x): {}".format(pdfmax))
    print("Minimum value of g(x): {}".format(dbd.g(pdfmax)))

    # Plot data
    plt.figure(1)
    plot_data(X,y)
    plot_shortest_path(X,dbd,1,2)
    plot_shortest_path(X,dbd,10,15)
    plot_shortest_path(X,dbd,200,201)
    plt.show()

if __name__=="__main__":
    main()