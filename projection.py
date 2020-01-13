import numpy as np

class Projection:
    """Tools for projecting meshes (in the dimensionality reduction sense) to a plane and comparing the projections."""

    def compute_vertex_distance(self, i, j, k):
        """Computes the distance of the same vertex k in meshes i and j."""
        vertex_dim = self.vertices.shape[2]
        # l is used to loop the vertex coordinates.
        distance_vector = [self.vertices[i][k][l] - self.vertices[j][k][l]
                           for l in range(0, vertex_dim)]
        distance_vector = np.array(distance_vector)
        return np.linalg.norm(distance_vector)

    def compute_mesh_distances(self, i, j):
        """Computes the distance between meshes i and j."""
        if i < j:
            vertices_dist = [self.compute_vertex_distance(i, j, k)
                             for k in range(0, self.vertices.shape[2])]
            vertices_dist = np.array(vertices_dist)
            return vertices_dist.mean
        return 0.0

    def project(self, i):
        """Projects mesh i."""
        return self.model.encode(np.array([self.vertices[i]]))

    def compute_projection_distances(self, i, j):
        """Computes the distance between projection of meshes i and j."""
        if i < j:
            return self.compute_vertex_distance(i, j, 0)
        return 0.0

    def __init__(self, model, facedata):
        print("##### Starting projection. #####")

        self.model = model
        self.vertices = facedata.vertices_test
        shape = self.vertices.shape
        print("Facedata vertex matrix shape: " + str(shape))

        print("Computing mesh distances...")
        self.similarity_meshes = np.array([self.compute_mesh_distances(i, j)
                      for i in range(0, shape[0])
                      for j in range(0, shape[0])])
        self.similarity_meshes.shape = (shape[0], shape[0])

        print("Similarity meshes: " + str(self.similarity_meshes))

        # print("Projecting...")
        # self.projections = [self.project(i)
        #                     for i in range(0, shape[0])]
        # self.projections = np.array(self.projections)
        # self.projections.shape = (shape[0], 2)
        #
        # print("Projections: " + str(self.projections.shape))
        #
        # print("Computing projection distances...")
        # self.similarity_projections = [self.compute_projection_distances(i, j)
        #                                for i in range(0, shape[0])
        #                                for j in range(0, shape[0])]
        # self.similarity_projections = np.array(self.similarity_projections)
        # self.similarity_projections.shape = (shape[0], shape[0])
        #
        # print("Similarity projections: " + str(self.similarity_projections.shape))
        #
        # numerator = np.sum(np.square(self.similarity_meshes - self.similarity_projections))
        # DEBUG
        numerator = 1.0
        #
        denominator = np.sum(np.square(self.similarity_meshes))
        self.stress = np.sqrt(np.divide(numerator, denominator))

        print("Stress: " + str(self.stress))