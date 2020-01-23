import numpy as np
from sklearn.manifold import MDS, TSNE
from os import path, makedirs
import matplotlib.pyplot as plt
from math import ceil


def plot_projections(data_name, projections):
    for i in range(0, len(projections)):
        plt.subplot(2, ceil(len(projections) / 2), i + 1)
        projection = projections[i]
        plt.scatter(projection.projections[:, 0], projection.projections[:, 1], c=projection.local_stress)
        plt.title(data_name + ': ' + projection.proj_type + ' projection. N = ' + str(len(projection.projections))
                  + ' Global stress = ' + str(projection.global_stress))
        plt.colorbar(label='local stress')
    plt.show()


def get_files(proj_type, data_name):
    main_dir = path.join('projections', data_name)
    proj_dir = path.join(main_dir, proj_type)
    makedirs(proj_dir, exist_ok=True)
    files = {'meshes_similarity': path.join(main_dir, 'meshes_similarity.npy'),
             'projections': path.join(proj_dir, 'projection.npy'),
             'projections_similarity': path.join(proj_dir, 'projections_similarity.npy'),
             'local_stress': path.join(proj_dir, 'local_stress.npy'),
             'global_stress': path.join(proj_dir,'global_stress.npy')}
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

    def project_coma(self, i):
        """Projects mesh i."""
        return self.model.encode(np.array([self.vertices[i]]))

    def compute_projection_distances(self, i, j):
        """Computes the distance between projection of meshes i and j."""
        if i < j:
            return self.compute_vertex_distance(i, j, 0)
        return 0.0

    # Second implementation
    # def compute_stress(self):
    #     # Copy upper triangle to bottom for convenience
    #     meshes_similarity = self.meshes_similarity + self.meshes_similarity.transpose()
    #     projections_similarity = self.projections_similarity + self.projections_similarity.transpose()
    #
    #     # Calculate the means for meshes and projections similarities
    #     meshes_means = np.array([np.mean(meshes_similarity[i])
    #                              for i in range(0, self.meshes_similarity.shape[0])])
    #
    #     projections_means = np.array([np.mean(projections_similarity[i] )
    #                              for i in range(0, self.projections_similarity.shape[0])])
    #
    #     # Calculate the stress per projected point
    #     # local_stress = np.divide(meshes_means, projections_means)
    #     local_stress = meshes_means - projections_means
    #
    #     numerator = np.sum(np.square(self.meshes_similarity - self.projections_similarity))
    #     denominator = np.sum(np.square(self.meshes_similarity))
    #     global_stress = np.sqrt(numerator / denominator)
    #
    #     return local_stress, global_stress

    def compute_stress(self):
        local_stress = np.array(
            [np.sum(abs(self.meshes_similarity[i] - self.projections_similarity[i]))
             for i in range(0, self.meshes_similarity.shape[0])])

        global_stress = np.sum(abs(self.meshes_similarity - self.projections_similarity))

        return local_stress, global_stress

    def __init__(self, proj_type, data_name, model=None, facedata=None):
        self.proj_type = proj_type
        print('Projection type: ' + proj_type)

        files = get_files(proj_type, data_name)

        if model is None:
            print('##### Loading projection. #####')

            self.meshes_similarity = np.load(files['meshes_similarity'])
            self.projections = np.load(files['projections'])
            self.projections_similarity = np.load(files['projections_similarity'])

            self.local_stress = np.load(files['local_stress'])
            self.global_stress = np.load(files['global_stress'])

            print('Matrices loaded. Meshes similarity: ' + str(self.meshes_similarity.shape)
                  + '\nProjections similarity: ' + str(self.projections_similarity.shape)
                  + '\nProjections: ' + str(self.projections.shape)
                  + '\nLocal stress: ' + str(self.local_stress.shape)
                  + '\nGlobal stress: ' + str(self.global_stress))
        else:
            print('##### Starting projection. #####')

            self.model = model
            self.vertices = facedata.vertices_test
            shape = self.vertices.shape
            print('Facedata vertex matrix shape: ' + str(shape))

            print('Computing mesh distances...')
            # self.meshes_similarity = np.array([self.compute_mesh_distances(i, j)
            #                                    for i in range(0, shape[0])
            #                                    for j in range(0, shape[0])])
            # self.meshes_similarity.shape = (shape[0], shape[0])

            # DEBUG
            self.meshes_similarity = np.load(files['meshes_similarity'])
            #

            print('Similarity meshes: ' + str(self.meshes_similarity))
            np.save(files['meshes_similarity'], self.meshes_similarity)

            print('Projecting...')
            if proj_type == 'coma':
                self.projections = np.array([self.project_coma(i)
                                            for i in range(0, shape[0])])
            elif proj_type == 'mds':
                embedding = MDS(dissimilarity='precomputed')
                meshes_similarity = self.meshes_similarity + self.meshes_similarity.transpose()
                self.projections = np.array(embedding.fit_transform(meshes_similarity))
                self.global_stress = embedding.stress_
            elif proj_type == 'tsne':
                embedding = TSNE(metric='precomputed')
                meshes_similarity = self.meshes_similarity + self.meshes_similarity.transpose()
                self.projections = np.array(embedding.fit_transform(meshes_similarity))
                self.global_stress = embedding.kl_divergence_
            else:
                raise ValueError('Unexpected projection type.')

            self.projections.shape = (shape[0], 2)

            print('Projections: ' + str(self.projections))
            np.save(files['projections'], self.projections)

            print('Computing projection distances...')
            self.projections_similarity = np.array([self.compute_projection_distances(i, j)
                                                    for i in range(0, shape[0])
                                                    for j in range(0, shape[0])])
            self.projections_similarity.shape = (shape[0], shape[0])

            # TEST
            # self.projections_similarity = np.zeros((2, 2))
            #

            print('Similarity projections: ' + str(self.projections_similarity))
            np.save(files['projections_similarity'], self.projections_similarity)

            self.local_stress, global_stress = self.compute_stress()
            if proj_type == 'coma':
                self.global_stress = global_stress
            np.save(files['local_stress'], self.local_stress)
            np.save(files['global_stress'], self.global_stress)
