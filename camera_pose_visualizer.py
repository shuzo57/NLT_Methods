import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CameraPoseVisualizer:
    def __init__(self,):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlabel('x-axis')
        self.ax.set_ylabel('y-axis')
        self.ax.set_zlabel('z-axis')
        self.xrange = [1e10, -1e10]
        self.yrange = [1e10, -1e10]
        self.zrange = [1e10, -1e10]
    
    def add_camera(self, extrinsic, focal_len_scaled=5, aspect_ratio=0.3, color="blue"):
        vertex_std = np.array([
            [0, 0, 0, 1],
            [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]
        ])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
        ]
        self.ax.add_collection3d(Poly3DCollection(meshes, facecolors=color, linewidths=1, edgecolors='r', alpha=.25))

        # update range
        xmin, xmax = np.min(vertex_transformed[:, 0]), np.max(vertex_transformed[:, 0])
        ymin, ymax = np.min(vertex_transformed[:, 1]), np.max(vertex_transformed[:, 1])
        zmin, zmax = np.min(vertex_transformed[:, 2]), np.max(vertex_transformed[:, 2])
        self.xrange = [min(self.xrange[0], xmin), max(self.xrange[1], xmax)]
        self.yrange = [min(self.yrange[0], ymin), max(self.yrange[1], ymax)]
        self.zrange = [min(self.zrange[0], zmin), max(self.zrange[1], zmax)]
    
    def update_range(self, xrange: list=None, yrange: list=None, zrange: list=None):
        if xrange is not None:
            self.xrange = xrange
        if yrange is not None:
            self.yrange = yrange
        if zrange is not None:
            self.zrange = zrange
        self.ax.set_xlim(self.xrange)
        self.ax.set_ylim(self.yrange)
        self.ax.set_zlim(self.zrange)
    
    def rotate_view(self, elev=30, azim=30):
        self.ax.view_init(elev=elev, azim=azim)

    def show(self,):
        self.update_range()
        plt.show()
    
    def save(self, path):
        self.update_range()
        self.fig.savefig(path)