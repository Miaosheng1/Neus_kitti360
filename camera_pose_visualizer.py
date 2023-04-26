import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import mpl_toolkits.mplot3d.art3d as art3d
import torch
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
class CameraPoseVisualizer:
    def __init__(self,xlim,ylim,zlim):
        plt.close('all')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera  visualizer')

    def extrinsic2pyramid(self, extrinsic, color='y', focal_len_scaled=0.1, aspect_ratio=0.3):
        ## 绘制平移向量
        if isinstance(extrinsic,torch.Tensor):
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

    def vis_camera_rays(self,rays_o,rays_d,N_rays = 50):
        H,W,_ = rays_o.shape
        if isinstance(rays_d,torch.Tensor):
            rays_o = rays_o.detach().cpu().numpy()
            rays_d = rays_d.detach().cpu().numpy()
        coordinate_y = np.random.randint(low=0,high=H,size=[N_rays,])
        coordinate_x = np.random.randint(low=0,high=W,size=[N_rays,])
        coords = np.stack([coordinate_y,coordinate_x],axis=-1)
        ## 根据坐标，提取出对应coord 的 rays_d 和 rays_o
        orign = rays_o[coords[...,0],coords[...,1]]
        view_direction = rays_d[coords[...,0],coords[...,1]]
        t = np.linspace(0, 1, 64).reshape(1,-1)
        line_points = orign[...,None,:] + view_direction[...,None,:]*t[...,:,None]
        for i in range(N_rays): 
            self.ax.plot(line_points[i,:,0], line_points[i,:,1], line_points[i,:,2], color='red')
        self.show(file_name='vis_ray.png')




    def show(self,file_name="Camera_visual.png"):
        plt.title('Extrinsic Parameters')
        plt.legend(['orign_point','x axis','y axis','z axis'])
        plt.show()
        plt.savefig(file_name)
        plt.close('all')
        
