import numpy as np 
import numpy.linalg as la
from scipy.io import loadmat, savemat
import math as m 
import matplotlib.pyplot as plt 
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
from sklearn.decomposition import PCA

# mpl.rcParams['text.usetex'] = True

def plotSingularValues(fig,X,N=10,normalize=True):
	ax = fig.add_subplot(111)
	s = la.svd(X,compute_uv=False)
	s = s/sum(s)
	N = min(N,len(s))
	ax.plot(range(0,N), s[0:N],'o',)

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''
	color_norm = colors.Normalize(vmin=0, vmax=N)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def plotPrincipalComponents(fig,X,y=None):
	ax = fig.add_subplot(1,1,1, projection='3d')
	pca = PCA(n_components=3, whiten=False)
	Xtrans = pca.fit_transform(X)
	ylab = y[np.where(y!=-1)[0]]
	unlab = np.where(y==-1)[0]

	alpha = 0.7 if len(unlab)==0 else 0.5
	
	if len(unlab)>0:
		ax.scatter(Xtrans[unlab,0],Xtrans[unlab,1],Xtrans[unlab,2],'o',c='gray', alpha=1,label='unlabelled')
	if y is None:
		ax.scatter(Xtrans[:,0],Xtrans[:,1],Xtrans[:,2],'o')
	else:
		labels = np.unique(ylab)
		cmap = get_cmap(len(labels))
		for i,lab in enumerate(labels):
			x1 = Xtrans[np.where(y==lab),0]
			x2 = Xtrans[np.where(y==lab),1]
			x3 = Xtrans[np.where(y==lab),2]
			ax.scatter(x1,x2,x3,'o',c=cmap(i), alpha=alpha,label=str(lab))
	
	ax.legend()
	ax.set_xlabel('1^{st} Principal component')
	ax.set_ylabel('2^{d} Principal component')
	ax.set_zlabel('3^{d} Principal component')