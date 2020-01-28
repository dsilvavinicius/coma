import matplotlib.pyplot as plt
import numpy as np
from math import ceil


class ProjectionUI:

    def onclick(self, event):
        self.model_visualizer.latent_vector = np.array((event.xdata, event.ydata))
        self.model_visualizer.latent_vector.shape = (1, 2)
        self.model_visualizer.decode_and_show()

    def plot_projections(self):
        for i in range(0, len(self.projections)):
            plt.subplot(2, ceil(len(self.projections) / 2), i + 1)
            projection = self.projections[i]
            plt.scatter(projection.projections[:, 0], projection.projections[:, 1], c=projection.local_stress)
            plt.title(self.data_name + ': ' + projection.proj_type + ' projection. N = '
                      + str(len(projection.projections)) + ' Global stress = ' + str(projection.global_stress))
            plt.colorbar(label='local stress')
            plt.get_current_fig_manager()
            plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def __init__(self, data_name, projections, model_visualizer):
        self.data_name = data_name
        self.projections = projections
        self.model_visualizer = model_visualizer
        self.plot_projections()
