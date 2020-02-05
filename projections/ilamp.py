import numpy as np
from math import sqrt
from sklearn.neighbors import NearestNeighbors

class Ilamp:
    """Inverse projection using the ILAMP algorithm described in "iLAMP: exploring high-dimensional spacing through
    backward multidimensional projection" (https://ieeexplore.ieee.org/document/6400489)."""

    def __init__(self, X, Y, k):
        """ Constructor.
        Parameters
        ----------
        X : array
            Mesh array with shape (N, m, 3), where N is the number of meshes, m is the number of vertices per mesh.
            Last dimension is the dimension of each vertex (3 since we are working with 3-dim meshes).
        Y : array
            Array with the projections of the meshes in `x`. Its shape must be (N, 2), where N is the number of meshes.
            Last dimension is the projection dimension (2 since we are in the plane).
        k : float
            Is the number of neighbors considered in the first step of the algorithm.
        """
        self.X = X
        self.Y = Y
        self.k = k
        self.NN = NearestNeighbors(n_neighbors=k)
        self.NN.fit(Y)

        print('ILAMP created. X shape: ' + str(X.shape) + ' Y shape: ' + str(Y.shape))

    def invert(self, p):
        """ Inverse project `p`.
        Parameters
        ----------
        p : 2d point to be inverse projected.
        Returns
        -------
        array
            Array with the mesh that is the inverse projection of `p`. Its shape is (m, 3), where m is number of
            vertices of the mesh.
        """
        print('Inverting ' + str(p))

        # First, find the k nearest neighbors
        k = self.k
        p.shape = (1, p.shape[0])
        idx = self.NN.kneighbors(p, return_distance=False)
        x = np.array([self.X[idx[i]] for i in range(0, len(idx))])
        y = np.array([self.Y[idx[i]] for i in range(0, len(idx))])
        x.shape = (k, self.X.shape[1], self.X.shape[2])
        y.shape = (k, p.shape[1])
        print('Neighbors found. X shape: ' + str(x.shape) + ' Y shape: ' + str(y.shape))

        # Then, compute the inverse projection
        alpha = np.array([1 / np.dot((y[i] - p)[0], (y[i] - p)[0]) for i in range(0, k)])
        sum_alpha = np.sum(alpha)
        x_til = np.sum(np.array([alpha[i] * x[i] for i in range(0, k)]), axis=0) / sum_alpha
        y_til = np.sum(np.array([alpha[i] * y[i] for i in range(0, k)]), axis=0) / sum_alpha
        x_hat = x - x_til
        y_hat = y - y_til

        print('Tildes and hats computed. x_til shape: ' + str(x_til.shape) + ' y_til shape: ' + str(y_til.shape)
              + ' x_hat shape: ' + str(x_hat.shape) + ' y_hat shape: ' + str(y_hat.shape))

        sqrt_alpha = [sqrt(alpha[i]) for i in range(0, k)]
        A = np.array([sqrt_alpha[i] * y_hat[i] for i in range(0, k)])
        B = np.array([sqrt_alpha[i] * x_hat[i] for i in range(0, k)])

        A_t = np.transpose(A)
        print('A transposed shape ' + str(A_t.shape) + ' B shape ' + str(B.shape))

        q = np.zeros((5023, 3))
        for i in range(0, 3):
            AB = np.matmul(A_t, B[:, :, i])

            print('AB shape' + str(AB.shape))

            U, D, V = np.linalg.svd(AB, full_matrices=False)
            print('SVD complete. U shape: ' + str(U.shape) + ' D shape: ' + str(D.shape) + ' V shape: ' + str(V.shape))
            M = np.matmul(U, V)
            q[:, i] = np.matmul((p - y_til), M) + x_til[:, i]

        print('Reverse projection q computed. Shape: ' + str(q.shape))
        return q
