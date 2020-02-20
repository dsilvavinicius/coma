from sklearn.decomposition import PCA
from projections.tensor_pca import TensorPCA
import numpy as np

class PCAProjection:

    def __init__(self, X, n_components = 2):
        self.pca = PCA(n_components=n_components)
        self.mesh_size = X.shape[1]

        print('PCA projection. X shape: ' + str(X.shape))

        X_2d = np.concatenate((X[:, :, 0], X[:, :, 1], X[:, :, 2]), axis=1)

        print('Matricized X shape: ' + str(X_2d.shape))

        self.proj_X = self.pca.fit_transform(X_2d)

        print('Projected X shape: ' + str(self.proj_X.shape))

    def invert(self, proj_X):
        print('Inverse projecting ' + str(proj_X))

        mesh = self.pca.inverse_transform(proj_X)

        print('Inverse projection temp shape: ' + str(mesh.shape))

        limits = (self.mesh_size, 2 * self.mesh_size, 3 * self.mesh_size)

        print('limits: ' + str(limits))

        slice0 = mesh[0:limits[0]]
        slice1 = mesh[limits[0]:limits[1]]
        slice2 = mesh[limits[1]:limits[2]]
        mesh = np.array([slice0, slice1, slice2]).transpose()

        print('Inverse projection shape: ' + str(mesh.shape))

        return mesh

    #print('PCA Components shape: ' + str(self.pcas[0].components_.shape))
    #mesh = np.array([self.pcas[i].mean_ + np.matmul(self.pcas[i].components_.transpose(), proj_X[:, i])
    #                for i in range(0, proj_X.shape[1])])
    #mesh.shape = (self.pcas[0].n_features_, proj_X.shape[1])
    #return mesh

    # Implementation with tensors
    #def __init__(self, X, n_components=2):
    #    self.pca = TensorPCA(ranks=[X.shape[0], n_components, 1], modes=[0, 1, 2])
    #    self.pca.fit(X)
    #    self.proj_X = self.pca.transform(X)
    #    self.proj_X.shape = (X.shape[0], n_components)

    #def invert(self, proj_X):
    #    core = np.zeros(self.pca.core.shape)
    #    proj = proj_X
    #    proj.shape = (proj.shape[0], 1)
    #    for i in range(0, self.pca.core.shape[0]):
    #        core[i] = proj
    #    mesh = self.pca.invert(core)[0]
    #    print('Inverted mesh shape: ' + str(mesh.shape))
    #    return mesh
