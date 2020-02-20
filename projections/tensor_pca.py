import tensorly as tl
from tensorly.decomposition import partial_tucker

class TensorPCA:
    def __init__(self, ranks, modes):
        self.ranks = ranks
        self.modes = modes

    def fit(self, tensor):
        self.core, self.factors = partial_tucker(tensor, modes=self.modes, ranks=self.ranks)
        print('Core shape: ' + str(self.core.shape))
        print('Factors shape: ' + str([f.shape for f in self.factors]))
        return self

    def transform(self, tensor):
        print('Tensor to be transformed shape: ' + str(tensor.shape))

        return tl.tenalg.multi_mode_dot(tensor, self.factors, modes=self.modes, transpose=True)

    def invert(self, core):
        print('Core to be reconstructed shape: ' + str(core.shape))

        return tl.tucker_to_tensor((core, self.factors))