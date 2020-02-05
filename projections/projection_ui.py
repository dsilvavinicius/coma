import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from projections.ilamp import Ilamp
from projections.rbf import Rbf

class ProjectionUI:
    """Exploratory UI for mesh dimensionality reduction projections. Generates a plot for each projection and a mesh
    visualizer to present results of inverse projections. Each plot can be clicked to update the mesh visualizer with
    the inverse projection of the point clicked."""

    def __invert_and_show(self, event):
        fig_axes = plt.gcf().get_axes()
        for i in range(0, len(fig_axes)):
            # Two axes per plot: the scatter plot and the color bar.
            projection = self.projections[i // 2]
            if fig_axes[i].in_axes(event):
                xy = np.array((event.xdata, event.ydata))
                if projection.proj_type != 'coma':
                    mesh = self.proj_inverse_map[projection.proj_type].invert(xy)
                    mesh.shape = (1, mesh.shape[0], mesh.shape[1])
                    self.model_visualizer.show(mesh)
                else:
                    self.model_visualizer.latent_vector = xy
                    self.model_visualizer.latent_vector.shape = (1, xy.shape[0])
                    self.model_visualizer.decode_and_show()
                break

    def __on_click(self, event):
        self.__invert_and_show(event)
        self.clicking = True

    def __on_release(self, event):
        self.clicking = False

    def __on_move(self, event):
        if self.clicking:
            self.__invert_and_show(event)

    def __plot_projections(self):
        # Each projection has it plot and color bar.
        for i in range(0, len(self.projections)):
            plt.subplot(2, ceil(len(self.projections) / 2), i + 1)
            projection = self.projections[i]
            plt.scatter(projection.projections[:, 0], projection.projections[:, 1], c=projection.local_stress)
            plt.title(self.data_name + ': ' + projection.proj_type + ' projection. N = '
                      + str(len(projection.projections)) + ' Global stress = ' + str(projection.global_stress))
            plt.colorbar(label='local stress')
            plt.gcf().canvas.mpl_connect('button_press_event', self.__on_click)
            plt.gcf().canvas.mpl_connect('button_release_event', self.__on_release)
            plt.gcf().canvas.mpl_connect('motion_notify_event', self.__on_move)

        # Set up Figure size and position
        plt.gcf().set_size_inches(self.fig_size)
        manager = plt.get_current_fig_manager()
        try:
            manager.window.SetPosition(self.fig_pos)
        except AttributeError:
            try:
                manager.window.wm_geometry('+' + str(self.fig_pos[0]) + '+' + str(self.fig_pos[1]))
            except AttributeError:
                print("Could not set plots position. Using default.")
        plt.show()

    def __init__(self, data_name, projections, model_visualizer, fig_size=(8.0, 8.0), fig_pos=(800, 0), inverse='lamp'):
        self.data_name = data_name
        self.projections = projections
        self.model_visualizer = model_visualizer
        self.clicking = False
        self.proj_inverse_map = {}
        self.fig_size = fig_size
        self.fig_pos = fig_pos

        for projection in projections:
            if projection.proj_type != 'coma':
                if inverse == 'lamp':
                    self.proj_inverse_map[projection.proj_type] = Ilamp(projection.vertices, projection.projections, 5)
                else:
                    self.proj_inverse_map[projection.proj_type] = Rbf(projection.vertices, projection.projections, c=0,
                                                                      e=1)

        self.__plot_projections()