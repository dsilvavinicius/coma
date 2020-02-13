import numpy as np
from scipy.linalg import lu_factor, lu_solve
from math import exp

class Rbf:
    """Inverse projection using Radial Basis Functions (rbfs), as described in "Facing the high-dimensions: inverse
    projection with radial basis functions" (https://repositorio.usp.br/bitstream/handle/BDPI/51320/2683523.pdf)"""

    def __rbf(self, p0, p1):
        r = np.linalg.norm(p0 - p1)
        if self.type == 'gaussian':
            return exp(-self.e * r * r)
        else:
            # multiquadrics
            return np.sqrt(self.c * self.c + self.e * r * r)

    def __init__(self, x, y, pca, c, e, _type='multiquadrics'):
        """ Constructor.
        Parameters
        ----------
        x : array
            Mesh array with shape (N, m, 3), where N is the number of meshes, m is the number of vertices per mesh.
            Last dimension is the dimension of each vertex (3 since we are working with 3-dim meshes).
        y : array
            Array with the projections of the meshes in `x`. Its shape must be (N, 2), where N is the number of meshes.
            Last dimension is the projection dimension (2 since we are in the plane).
        pca: PCAProjection
            PCAProjection used to invert further to mesh space.
        c : float
            first constant for rbf function. Used for multiquadrics.
        e : float
            second constant for rbf function. Used for multiquadrics and gaussian.
        """
        print('Starting RBF inverse projection.')

        self.pca = pca
        self.c = c
        self.e = e
        self.type = _type

        N = x.shape[0]
        m = x.shape[1]

        # Create phi matrix
        phi = np.empty((N, N))
        for i in range(0, N):
            for j in range(0, N):
                if i <= j:
                    phi[i, j] = phi[j, i] = self.__rbf(y[i], y[j])

        print('Phi shape: ' + str(phi.shape))

        # Create b
        b = x
        print('b shape ' + str(b.shape))

        # Solve system phi * lambda = b for lambda
        phi_factor = lu_factor(phi, overwrite_a=True)
        self._lambda = np.empty((N, m, b.shape[2]))
        for i in range(0, b.shape[2]):
            for k in range(0, m):
                self._lambda[:, k, i] = lu_solve(phi_factor, b[:, k, i])

        print('Lambda shape ' + str(self._lambda.shape))

        self.y = y
        self.N = N
        self.m = m

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
        print('Inverting ' + str(p) + ' into q')
        q = np.empty((self.m, self._lambda.shape[2]))
        for j in range(0, q.shape[1]):
            for k in range(0, self.m):
                q[k, j] = np.sum(np.array(
                    [self._lambda[i, k, j] * self.__rbf(self.y[i], p)
                     for i in range(0, self.N)]
                ))

        print('q shape before pca: ' + str(q.shape))
        q = self.pca.invert(q)
        print('q shape after pca: ' + str(q.shape))
        return q
