import numpy as np
from os import path, makedirs
import matplotlib.pyplot as plt


def get_files(name):
    directory = path.join("projections", name)
    makedirs(directory, exist_ok=True)
    files = {"meshes_similarity" : path.join(directory, "meshes_similarity.npy"),
             "projections" : path.join(directory, "projection.npy") ,
             "projections_similarity" : path.join(directory, "projections_similarity.npy")}
    return files


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
            return vertices_dist.mean()
        return 0.0

    def project(self, i):
        """Projects mesh i."""
        return self.model.encode(np.array([self.vertices[i]]))

    def compute_projection_distances(self, i, j):
        """Computes the distance between projection of meshes i and j."""
        if i < j:
            return self.compute_vertex_distance(i, j, 0)
        return 0.0

    def plot_stress(self):
        # Copy upper triangle to bottom for convenience
        meshes_similarity = self.meshes_similarity + self.meshes_similarity.transpose()
        projections_similarity = self.projections_similarity + self.projections_similarity.transpose()

        # Calculate the means for meshes and projections similarities
        meshes_means = np.array([np.mean(meshes_similarity[i])
                                 for i in range(0, self.meshes_similarity.shape[0])])

        projections_means = np.array([np.mean(projections_similarity[i] )
                                 for i in range(0, self.projections_similarity.shape[0])])

        # Calculate the stress per projected point
        local_stress = np.divide(meshes_means, projections_means)

        plt.scatter(self.projections[:, 0], self.projections[:, 1], c=local_stress)
        plt.title("Mesh projections")
        plt.colorbar(label="local stress")
        plt.show()

    def __init__(self, name, model=None, facedata=None):
        files = get_files(name)

        if model is None:
            print("##### Loading projection. #####")

            self.meshes_similarity = np.load(files["meshes_similarity"])
            self.projections = np.load(files["projections"])
            self.projections_similarity = np.load(files["projections_similarity"])
        else:
            print("##### Starting projection. #####")

            self.model = model
            self.vertices = facedata.vertices_test
            shape = self.vertices.shape
            print("Facedata vertex matrix shape: " + str(shape))

            print("Computing mesh distances...")
            self.meshes_similarity = np.array([self.compute_mesh_distances(i, j)
                                               for i in range(0, shape[0])
                                               for j in range(0, shape[0])])
            self.meshes_similarity.shape = (shape[0], shape[0])

            # TEST
            # self.meshes_similarity = np.zeros((2, 2))
            #

            print("Similarity meshes: " + str(self.meshes_similarity))
            np.save(files["meshes_similarity"], self.meshes_similarity)

            print("Projecting...")
            self.projections = np.array([self.project(i)
                                for i in range(0, shape[0])])
            self.projections.shape = (shape[0], 2)

            # TEST
            # self.projections = np.zeros((2, 2))
            #

            print("Projections: " + str(self.projections))
            np.save(files["projections"], self.projections)

            print("Computing projection distances...")
            self.projections_similarity = np.array([self.compute_projection_distances(i, j)
                                                    for i in range(0, shape[0])
                                                    for j in range(0, shape[0])])
            self.projections_similarity.shape = (shape[0], shape[0])

            # TEST
            # self.projections_similarity = np.zeros((2, 2))
            #

            print("Similarity projections: " + str(self.projections_similarity))
            np.save(files["projections_similarity"], self.projections_similarity)

        numerator = np.sum(np.square(self.meshes_similarity - self.projections_similarity))
        denominator = np.sum(np.square(self.meshes_similarity))
        self.stress = np.sqrt(numerator / denominator)
        print("Global stress: " + str(self.stress))

        self.plot_stress()