import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from projections.ilamp import Ilamp

class ProjectionUI:

    def invert_and_show(self, event):
        fig_axes = plt.gcf().get_axes()
        for i in range(0, len(fig_axes)):
            projection = self.projections[i//2]
            if fig_axes[i].in_axes(event):
                xy = np.array((event.xdata, event.ydata))
                if projection.proj_type != 'coma':
                    mesh = self.proj_inverse_map[projection.proj_type].invert(xy)
                    mesh.shape = (1, 5023, 3)
                    self.model_visualizer.show(mesh)
                else:
                    self.model_visualizer.latent_vector = xy
                    self.model_visualizer.latent_vector.shape = (1, 2)
                    self.model_visualizer.decode_and_show()
                break

    def on_click(self, event):
        self.invert_and_show(event)
        self.clicking = True

    def on_release(self, event):
        self.clicking = False

    def on_move(self, event):
        if self.clicking:
            self.invert_and_show(event)

    def plot_projections(self):
        for i in range(0, len(self.projections)):
            plt.subplot(2, ceil(len(self.projections) / 2), i + 1)
            projection = self.projections[i]
            plt.scatter(projection.projections[:, 0], projection.projections[:, 1], c=projection.local_stress)
            plt.title(self.data_name + ': ' + projection.proj_type + ' projection. N = '
                      + str(len(projection.projections)) + ' Global stress = ' + str(projection.global_stress))
            plt.colorbar(label='local stress')
            plt.get_current_fig_manager()
            plt.gcf().canvas.mpl_connect('button_press_event', self.on_click)
            plt.gcf().canvas.mpl_connect('button_release_event', self.on_release)
            plt.gcf().canvas.mpl_connect('motion_notify_event', self.on_move)
        plt.show()

    def __init__(self, data_name, projections, model_visualizer):
        self.data_name = data_name
        self.projections = projections
        self.model_visualizer = model_visualizer
        self.clicking = False
        self.proj_inverse_map = {}

        for projection in projections:
            if projection.proj_type != 'coma':
                self.proj_inverse_map[projection.proj_type] = Ilamp(projection.vertices, projection.projections, 5)

        self.plot_projections()