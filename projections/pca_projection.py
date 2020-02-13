from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import numpy as np

class PCAProjection:


    def __init__(self, X, n_components = 2):
        self.pcas = [PCA(n_components=n_components) for i in range(0, X.shape[2])]
        self.proj_X = np.array([self.pcas[i].fit_transform(X[:, :, i])
                                for i in range(0, X.shape[2])])
        self.proj_X.shape = (X.shape[0], n_components, X.shape[2])

    def invert(self, proj_X):
        mesh = np.array([self.pcas[i].inverse_transform(proj_X[:, i])
                         for i in range(0, proj_X.shape[1])])
        mesh.shape = (self.pcas[0].n_features_, proj_X.shape[1])
        return mesh
