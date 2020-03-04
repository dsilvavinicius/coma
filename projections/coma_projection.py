class ComaProjection:

    def __init__(self, model):
        self.model = model

    def invert(self, proj_X):
        proj_X.shape = (1, proj_X.shape[0])
        print('Projection shape: ' + str(proj_X.shape))

        mesh = self.model.decode(proj_X)
        mesh.shape = (mesh.shape[1], mesh.shape[2])
        return mesh
