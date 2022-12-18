import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import mpl_toolkits.mplot3d.art3d as art3d

class CameraPoseVisualizer:
    def __init__(self,xlim,ylim,zlim):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='y', focal_len_scaled=0.1, aspect_ratio=0.3):
        # vertex_std = np.array([[0, 0, 0, 1],
        #                        [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #                        [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #                        [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #                        [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,1]])
        #
        # vertex_transformed = vertex_std @ extrinsic.T
        # meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
        #           [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
        #           [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
        #           [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
        #           [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
        #            vertex_transformed[4, :-1]]]
        # self.ax.add_collection3d(
        #     Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

        ## 绘制平移向量
        extrinsic = extrinsic.detach().cpu().numpy()
        T = extrinsic[:3, 3]

        self.ax.scatter(T[0], T[1], T[2], marker='o', color='k',s=1)

        Rx, Ry, Rz = extrinsic[:3, 0], extrinsic[:3, 1], extrinsic[:3, 2]

        endX_point = T + Rx
        endY_point = T + Ry
        endZ_point = T + Rz

        ## draw三个坐标轴
        self.ax.plot((T[0], endX_point[0]), (T[1], endX_point[1]), (T[2], endX_point[2]), color='r')
        self.ax.plot((T[0], endY_point[0]), (T[1], endY_point[1]), (T[2], endY_point[2]), color='b')
        self.ax.plot((T[0], endZ_point[0]), (T[1], endZ_point[1]), (T[2], endZ_point[2]), color='g')



    def show(self):
        plt.title('Extrinsic Parameters')
        plt.legend(['x axis','y axis','z axis'])
        plt.savefig("Camera_visual.png")
