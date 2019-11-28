from psbody.mesh import Mesh, MeshViewers
import readchar
import numpy as np
import tkinter as tk
import os

class LatentSpaceVisualization:

    def decode_and_show(self):
        # Decode the latent vector and show resulting mesh
        self.recon_vec = self.model.decode(self.latent_vector)
        self.facedata.show_mesh(viewer=self.viewer, mesh_vecs=self.recon_vec, figsize=(1, 1))

    def update_value(self, event):
        for i in range(len(self.sliders)):
            self.latent_vector[0][i] = self.sliders[i].get()
        self.decode_and_show()

    # Save the current mesh in a text file, along with its parameters
    def save_unity(self):
        fileExists = False
        if os.path.exists('meshes.nd'):
            fileExists = True

        meshFile = open('meshes.nd', 'a')
        paramsFile = open('meshes.' + str(len(self.latent_vector[0])) + 'd', 'a')

        if fileExists:
            meshFile.write('\n')
            paramsFile.write('\n')

        for v in self.recon_vec[0]:
            meshFile.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + ' ')
        for pi in self.latent_vector:
            for pj in pi:
                paramsFile.write(str(pj) + ' ')

        meshFile.close()
        paramsFile.close()

    def save_ply(self):
        self.facedata.save_meshes('mesh.ply', self.recon_vec)

    def __init__(self, model, facedata, mesh_path=None):
        # Init members
        self.model = model
        self.facedata = facedata
        self.mesh_path = mesh_path
        self.viewer = MeshViewers(window_width=800, window_height=800, shape=[1, 1], titlebar='Meshes')

        # Encode
        if mesh_path is not None:
            normalized_mesh = self.facedata.get_normalized_meshes([mesh_path])
        else:
            normalized_mesh = np.array([self.facedata.vertices_test[0]])

        self.latent_vector = self.model.encode(normalized_mesh)

        self.decode_and_show()

        # Create the UI.
        self.sliders = []
        master = tk.Tk()

        for i in range(len(self.latent_vector[0])):
            slider = tk.Scale(master, from_=-5.0, to=5.0, resolution=0.01, length=500, orient=tk.HORIZONTAL)
            slider.set(self.latent_vector[0][i])
            slider.pack()
            slider.bind("<ButtonRelease-1>", self.update_value)
            self.sliders.append(slider)

        buttonUnity = tk.Button(text="Save unity", command=self.save_unity)
        buttonUnity.pack()

        buttonPly = tk.Button(text="Save ply", command=self.save_ply)
        buttonPly.pack()

        tk.mainloop()