import numpy as np
from math import sqrt
from sklearn.neighbors import NearestNeighbors

class Ilamp:

    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k
        self.NN = NearestNeighbors(n_neighbors=k)
        self.NN.fit(Y)

        print('ILAMP created. X shape: ' + str(X.shape) + ' Y shape: ' + str(Y.shape))

    def invert(self, p):
        print('Inverting ' + str(p))

        # First, find the k nearest neighbors
        k = self.k
        p.shape = (1, 2)
        idx = self.NN.kneighbors(p, return_distance=False)
        x = np.array([self.X[idx[i]] for i in range(0, len(idx))])
        y = np.array([self.Y[idx[i]] for i in range(0, len(idx))])
        x.shape = (k, 5023, 3)
        y.shape = (k, 2)
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

        q = np.zeros(3)
        for i in range(0, 3):
            AB = np.matmul(A_t, B[:, :, i])

            print('AB shape' + str(AB.shape))

            U, D, V = np.linalg.svd(AB)
            print('SVD complete. U shape: ' + str(U.shape) + 'D shape: ' + str(D.shape) + ' V shape: ' + str(V.shape))
            M = np.matmul(U, V)
            q[i] = (p - y_til) * M + x_til[i]

        print('Reverse projection q computed. Shape: ' + str(q.shape))
        return q
