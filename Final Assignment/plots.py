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
import utils as u

mpl.rcParams['text.usetex'] = True

def plotSingularValues(fig,X,N=10,normalize=True):
	ax = fig.add_subplot(111)
	s = la.svd(X,compute_uv=False)
	s = s/sum(s)
	N = min(N,len(s))
	ax.plot(range(1,N+1), s[0:N],'o',)
	ax.set_xlabel('N-largest',fontsize=15)
	if normalize:
		ax.set_ylabel('Percentage of total', fontsize=15)
	else:
		ax.set_ylabel('Singular value', fontsize=15)
	ax.set_title('Principal component')

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''
	color_norm = colors.Normalize(vmin=0, vmax=N)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def plotPrincipalComponents(fig,X,y=None, components=[0,1,2], title=''):
	ax = fig.add_subplot(1,1,1, projection='3d')
	pca = PCA(n_components=max(components)+1, whiten=False)
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
			xs = [ Xtrans[np.where(y==lab),c] for c in components ]
			ax.scatter(xs,'o',c=cmap(i), alpha=alpha,label=str(lab))
	ax.legend()
	ax.set_xlabel('Principal component {}'.format(components[0]+1), fontsize=15)
	ax.set_ylabel('Principal component {}'.format(components[1]+1), fontsize=15)
	ax.set_zlabel('Principal component {}'.format(components[2]+1), fontsize=15)
	ax.set_title(title, fontsize=17)

def plotPersonData(fig, X,y,S, person, components=[0,1,2], title=''):
	plotPrincipalComponents(fig, *u.splitByInds([u.getPersonInds(S,person)],X, y), components=components, title=title)

def plotRanks(fig, X,y,S):
	persons, labels = np.unique(S), np.unique(y)
	ranks = [la.matrix_rank(u.getCovariance(X,y,S,p,l)) for p in persons for l in labels]
	print(min(ranks))
	ax = fig.add_subplot(111)
	ax.hist(ranks)