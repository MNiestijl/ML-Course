import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import pandas as pd

class NMC():
    def __init__(self, Lambda):
        self.Lambda = Lambda

    # X ~ d × N, where d is the dimension of the data and N is the number of training samples.
    # y ~ N, classifying the i-th training example as 1 or 0, respectively.
    def fit(self, X, y):
        X0 = X[:,np.where(y==0)[0]]
        X1 = X[:,np.where(y==1)[0]]
        X0Mean = X0.mean(1)
        X1Mean = X1.mean(1)
        d = X.shape[0]
        n0 = X0.shape[1]
        n1 = X1.shape[1]
        M = np.zeros([d,2])
        for j in range(0,d):
            lambdaTerm = self.Lambda*(1/(2*n0) + 1/(2*n1))
            if (X0Mean[j] - X1Mean[j] >= lambdaTerm):
                M[j,0] = X0Mean[j] - self.Lambda/(2*n0)
                M[j,1] = X1Mean[j] + self.Lambda/(2*n1)
            elif (X1Mean[j] - X0Mean[j] >= lambdaTerm):
                M[j,0] = X0Mean[j] + self.Lambda/(2*n0)
                M[j,1] = X1Mean[j] - self.Lambda/(2*n1)
            else:
                M[j,[0,1]] = X[j,:].mean()
        self.M = M        

    def loss(self,X,y,M):
        return nl.norm(X - M[:,y.T], 'fro')**2 + self.Lambda * nl.norm(M[:,0] - M[:,1],1)

    def calcM0(self, X, y):
        M0 = np.zeros([X.shape[0],2])
        for i in range(0,2):
            if (np.where(y==i)[0].size == 0):
                continue
            M0[:,i] = X[:,np.where(y==i)[0]].mean(1)
        return M0

def plotLoss1D(X, y, lambdas, m1range):
    plt.figure()
    cmap = plt.get_cmap('jet')
    colors = [ cmap(x) for x in np.linspace(0,1,len(lambdas))]
    legend = [ "$\lambda$ = " + str(l) for l in lambdas ]
    
    ms = np.linspace(m1range[0], m1range[1], 1000)
    for i, l in enumerate(lambdas):
        nmc = NMC(l)
        loss = [ nmc.loss(X,y,np.array([[1,m]])) for m in ms ]
        minimizer = loss.index(min(loss))
        plt.plot(ms,loss,color=colors[i], label=legend[i], linewidth=2)
        plt.plot(ms[minimizer],loss[minimizer],'o',color=colors[i])
        print(loss[minimizer])
    plt.legend()
    plt.ylabel('L($m_{+}$)')
    plt.xlabel('$m_{+}$')
    plt.title('Loss Function L($m_{+}$)')
    plt.show()
    
def plotContours(X,y,lambdas,m0range, m1range):
    m0s = np.linspace(m0range[0], m0range[1], 100)
    m1s = np.linspace(m1range[0], m1range[1], 100)
    M0, M1 = np.meshgrid(m0s, m1s)
    for i, l in enumerate(lambdas):
        nmc = NMC(l)
        nmc.fit(X,y)
        plt.figure(i)
        plt.plot([nmc.M[0,0]],[nmc.M[0,1]],'o')
        nmc = NMC(l)
        loss = [[ nmc.loss(X,y,np.array([[m0,m1]])) for m0 in m0s ] for m1 in m1s]
        CS = plt.contour(M0,M1,loss)
        plt.title('$\lambda$ = ' + str(l))
        plt.xlabel('$m_{-}$', fontsize=18)
        plt.ylabel('$m_{+}$', fontsize=18)
        plt.clabel(CS, inline=1, fontsize=10)
    plt.show()




def exercise3():
    filename = 'optdigitsubset.txt'
    data = pd.read_csv(filename, delim_whitespace=True, header=None).as_matrix().T
    y = np.concatenate((np.zeros(554),np.ones(571)),axis=0)

    Lambda = 10000000
    nmc1 = NMC(0)
    nmc1.fit(data,y)
    nmc2 = NMC(Lambda)
    nmc2.fit(data,y)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(np.reshape(nmc1.M[:,0],[8,8]), cmap='gray')
    ax2.imshow(np.reshape(nmc1.M[:,1],[8,8]), cmap='gray')
    ax3.imshow(np.reshape(nmc2.M[:,0],[8,8]), cmap='gray')
    ax4.imshow(np.reshape(nmc2.M[:,1],[8,8]), cmap='gray')
    ax1.set_title('$m_-$', fontsize=18)
    ax2.set_title('$m_+$', fontsize=18)
    ax3.set_title('$m_-$', fontsize=18)
    ax4.set_title('$m_+$', fontsize=18)   
    plt.show()

def main():
    X = np.array([[-1,1,3,-1]])
    y = np.array([1,1,0,0])
    #X = np.array([[-1,1]])
    #y = np.array([1,1])
    #M0 = nmc.calcM0(X,y)
    #plotLoss1D(X,y,lambdas=[0,2,4,6],m1range=[-1,2])  
    plotContours(X,y,lambdas=[0,2,4,6],m0range=[-1,3],m1range=[-1,3])
    #exercise3()

if __name__ == "__main__":
    main()
