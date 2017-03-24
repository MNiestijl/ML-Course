import numpy as np
import math as m
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation
from copy import copy


class SSLDA_Classifier():
    def __init__(self, max_iter=10, n_components=2):
        self.n_components = n_components
        self.covariance_, self.means_, self.classifier = None, None, None

    def predict(self, X):
        return self.classifier.predict(X)

    def score(self, X, y):
        return self.classifier.score(X,y)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    # B ~ [N_samples * d_features]
    # Unlabeled data should have label -1
    def fit(self, X, y, method='self-training', threshhold=0.7):
        getLabel = lambda p: np.where(p>threshhold)[0][0] if np.any(p>threshhold) else -1
        yp = copy(y) # copy of original labels
        lda = LinearDiscriminantAnalysis(solver='lsqr',store_covariance=True, n_components=2)

        if method=='self-training':
            counter = 0
            while True: # Temporary
                mask = np.ones(len(y), dtype=bool)
                mask[np.where(yp==-1)[0]]=False
                if len(yp[~mask])==0 or counter==max_iter:
                    break
                lda.fit(X[mask,:],yp[mask])
                probs = lda.predict_proba(X[~mask])
                yp[~mask] = np.fromiter([getLabel(p) for p in probs], probs.dtype)
                counter+=1

        if method=='label-propagation':
            label_prop_model = LabelPropagation(kernel='knn', n_neighbors=10, alpha=0.9)
            label_prop_model.fit(X,yp)
            mask = np.ones(len(y), dtype=bool)
            mask[np.where(yp==-1)[0]]=False
            probs = label_prop_model.predict_proba(X[~mask])
            yp[~mask] = np.fromiter([getLabel(p) for p in probs], probs.dtype)
            print(len(np.where(yp==-1)[0]))
            lda.fit(X[mask,:],yp[mask])

        self.classifier, self.means_, self.covariance_ = lda, lda.means_, lda.covariance_