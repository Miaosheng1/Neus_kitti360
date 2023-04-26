import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
# from models.vis_3d import Vis
import yaml
import re
# from camera_pose_visualizer import CameraPoseVisualizer

class Kitti360_fisheye:
    def __init__(self,conf):
        super(Kitti360_fisheye, self).__init__()
        self.device = 'cuda'
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.intrinsic_dir = conf.get_string('intrinsic_fisheye')
        self.intrinsic_fisheye_02_dir = os.path.join(self.intrinsic_dir,"image_02.yaml")
        self.intrinsic_fisheye_03_dir = os.path.join(self.intrinsic_dir,"image_03.yaml")
        self.image_dir = os.path.join(self.data_dir,'2013_05_28_drive_0000_sync')
       

        self.came2world = {}
        self.start_index = 3353
        num_images = int(conf.get_string('num_images'))
        self.num_images = num_images
        

        '''Load extrinstic matrix'''
        CamPose_02 = {}
        CamPose_03 = {}
        extrinstic_file = os.path.join(self.data_dir, os.path.join('data_poses', '2013_05_28_drive_0000_sync'))
        pose_file = os.path.join(extrinstic_file, 'poses.txt')  ## IMU --> world
        poses = np.loadtxt(pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        
        self.CamToPose = self.loadCameraToPose(os.path.join(os.path.join(self.data_dir, 'calibration'), 'calib_cam_to_pose.txt'))  ## Cam--> IMU
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
            CamPose_02[frame] = np.matmul(pose, self.CamToPose['image_02:'])
            CamPose_03[frame] = np.matmul(pose, self.CamToPose['image_03:'])

        '''Load intrinstic Dictory'''
        self.intrinsic_fisheye_02 = self.readYAMLFile(os.path.join(self.data_dir,self.intrinsic_fisheye_02_dir))
        self.intrinsic_fisheye_03 = self.readYAMLFile(os.path.join(self.data_dir,self.intrinsic_fisheye_03_dir))
        
        ''' load Image'''
        image_02 = os.path.join(self.image_dir, 'image_02/data_rgb')
        image_03 = os.path.join(self.image_dir, 'image_03/data_rgb')   
        self.all_images = []
        self.all_poses = []

        for idx in range(self.start_index, self.start_index + num_images, 1):
            ## read image_02
            image = cv.imread(os.path.join(image_02, "{:010d}.png").format(idx)) / 256.0
            self.all_images.append(image)
            self.all_poses.append(CamPose_02[idx])

            ## read image_03
            image = cv.imread(os.path.join(image_03, "{:010d}.png").format(idx)) / 256.0
            self.all_images.append(image)
            self.all_poses.append(CamPose_03[idx])

        self.images = np.array(self.all_images).astype(np.float32)
        self.images = torch.from_numpy(self.images).cpu()
        self.n_images = len(self.all_images)
        self.poses = np.stack(self.all_poses).astype(np.float32)
        self.masks = torch.ones_like(self.images)
        
        ''' Normalize Pose'''
        self.poses = self.Normailize_T(self.poses)
        self.poses = torch.from_numpy(self.poses).to(self.device)

        '''     Visualize Pose
        visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])
        for i in np.arange(self.poses.shape[0]):
            if i % 1 == 0:
                print(self.poses[i][:3,3])
                visualizer.extrinsic2pyramid(self.poses[i])
        visualizer.show()
        exit()
        '''


        self.W = self.intrinsic_fisheye_02['image_width']
        self.H = self.intrinsic_fisheye_02['image_height']
        self.testset_index = np.array([4,9,14,18])
        self.trainset_index = np.setdiff1d(np.arange(num_images*2),self.testset_index)
        print("Data load end")

    def gen_random_rays_at(self, img_idx, batch_size):
        ## fisheye_02
        if img_idx % 2 == 0:
            k1 = self.intrinsic_fisheye_02['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_02['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_02['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_02['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_02['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_02['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_02['mirror_parameters']['xi']
            
        ## fisheye_03
        elif img_idx %2 == 1:
            k1 = self.intrinsic_fisheye_03['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_03['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_03['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_03['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_03['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_03['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_03['mirror_parameters']['xi']
        
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()
        
        iter = 10000
        map_dist = []
        z_dist = []
        ro2 = np.linspace(0.0, 1.0, iter)
        # for ro2 in np.linspace(0.0, 1.0, 200000):
        dis_cofficient = 1 + k1*ro2 + k2*ro2*ro2
        ro2_after = np.sqrt(ro2) * (1 + k1*ro2 + k2*ro2*ro2)  ## 畸变之后的 rou
        # map_dist.append([(1 + k1*ro2 + k2*ro2*ro2), ro2_after])
        map_dist = np.stack([dis_cofficient,ro2_after])
        map_dist = np.moveaxis(map_dist,-1,0)

        z = np.linspace(0.0, 1.0, iter)
        z_after = np.sqrt(1 - z**2) / (z + mirror)
        z_dist = np.stack([z,z_after])
        z_dist = np.moveaxis(z_dist,-1,0)

        map_dist = torch.from_numpy(map_dist).cpu()
        z_dist = torch.from_numpy(z_dist).cpu()

        ## 1. 将像素坐标系投影到归一化坐标系（畸变之后的坐标）
        x = (pixels_x - u0) / gamma1
        y = (pixels_y - v0) / gamma2
        dist = torch.sqrt( x*x + y*y ).cpu()
        indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)

        ##2. 除以畸变系数，得到畸变之前的 坐标
        x /= map_dist[indx, 0]
        y /= map_dist[indx, 0]

        ## 3.查找出 去畸变之后的 （x_undistortion,y_undistortion）对应的 Z_unitsphere 上的坐标,并根据Z_unitsphere 得到 投影前 x_unitsphere,y_unitsphere
        z_after = torch.sqrt(x*x + y*y).cpu()
        indx = torch.abs(z_dist[:, 1:] - z_after[None, :]).argmin(dim=0)
        x *= (z_dist[indx, 0] +mirror)
        y *= (z_dist[indx, 0] +mirror)
        xy = torch.stack((x, y))
        xys = xy.permute(1,0)

        ## 4. 根据 （x_undistortion,y_undistortion） 得到z 数值，即在球面坐标系的坐标。（Camera Coordinates）
        z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
        isnan = z.isnan()
        z[isnan] = 1.
        left_fisheye_grid = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)

        ## 5.筛选出在单位球上的有效点，因为并不是所有的点都是有效的
        valid = left_fisheye_grid[:, 3] < 0.5   
        left_valid = left_fisheye_grid[valid, :3]

        ## 产生有效的光线,将相机坐标系的光线ray_d 转化到 世界坐标系下面
        dirs = left_valid.to(self.device)
        dirs = dirs / torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(
        # [..., N_rays, 1, 3] * [..., 1, 3, 3]
        dirs[..., None, :] * self.poses[img_idx, None, :3, :3], -1
        ) 
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape)

        color = self.images[img_idx][(pixels_y, pixels_x)].reshape(-1,3)
        color = color[valid,:]
        mask = self.masks[img_idx][(pixels_y, pixels_x)].reshape(-1,3)
        mask = mask[valid,:]

        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()
    
    
    def loadCameraToPose(self,filename):
        # open file
        Tr = {}
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                lineData = list(line.strip().split())
                if lineData[0] == 'image_02:':
                    data = np.array(lineData[1:]).reshape(3, 4).astype(np.float)
                    data = np.concatenate((data, lastrow), axis=0)
                    Tr[lineData[0]] = data
                elif lineData[0] == 'image_03:':
                    data = np.array(lineData[1:]).reshape(3, 4).astype(np.float)
                    data = np.concatenate((data, lastrow), axis=0)
                    Tr[lineData[0]] = data
        return Tr
    
    def readYAMLFile(self,fileName):
        '''make OpenCV YAML file compatible with python'''
        ret = {}
        skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
        with open(fileName,'r') as fin:
            for i in range(skip_lines):
                fin.readline()
            yamlFileOut = fin.read()
            myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
            yamlFileOut = myRe.sub(r': \1', yamlFileOut)
            ret = yaml.safe_load(yamlFileOut)
        return ret
    
    def Normailize_T(self, poses):
        for i, pose in enumerate(poses):
            if i == 0:
                inv_pose = np.linalg.inv(pose)
                poses[i] = np.eye(4)
            else:
                poses[i] = np.dot(inv_pose, poses[i])

        '''New Normalization '''
        # scale = np.max(poses[:,:3,3])
        # print(f"scale:{scale}\n")
        # for i in range(poses.shape[0]):
        #     poses[i, :3, 3] = poses[i, :3, 3] / scale
            

        return poses
    
    def image_at(self, idx, resolution_level):
        return self.all_images[idx]

    def near_far_from_sphere(self, rays_o, rays_d):
        near, far = 0., 6.
        return near,far
    
    def gen_rays_at(self, img_idx, resolution_level=1):
        def chunk(grid):
            ## 1. 将图像坐标系上的 2 维度像素，投影到 归一化 相机坐标系
            x = grid[0, :]
            y = grid[1, :]

            x = (x - u0) / gamma1
            y = (y - v0) / gamma2

            dist = torch.sqrt(x*x + y*y)  ## 畸变之后的rou
            indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)
            x /= map_dist[indx, 0]
            y /= map_dist[indx, 0]
        
            z_after = torch.sqrt(x*x + y*y)
            indx = torch.abs(z_dist[:, 1:] - z_after[None, :]).argmin(dim=0)

            ## z_dist[indx, 0] +mirror 是对数值的修正， X*Z 表示从归一化坐标系到相机坐标系
            x *= (z_dist[indx, 0] +mirror)
            y *= (z_dist[indx, 0] +mirror)

            xy = torch.stack((x, y))
            return xy
        
        ## fisheye_02
        if img_idx % 2 == 0:
            k1 = self.intrinsic_fisheye_02['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_02['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_02['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_02['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_02['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_02['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_02['mirror_parameters']['xi']
            
        ## fisheye_03
        elif img_idx %2 == 1:
            k1 = self.intrinsic_fisheye_03['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_03['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_03['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_03['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_03['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_03['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_03['mirror_parameters']['xi']
        
        ## Test View 的时候 shape 设定为 
        
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W ).cpu()
        ty = torch.linspace(0, self.H - 1, self.H ).cpu()   
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.reshape(-1)
        pixels_y = pixels_y.reshape(-1)

        ## pixels 的第一维度应该为 表示行 pixels_y
        pixels = np.stack((pixels_y, pixels_x, np.ones_like(pixels_y)), axis=0)  
        grid = torch.from_numpy(pixels).cpu()
        
        iter = 10000  ## 采样的精度
        map_dist = []
        z_dist = []
        ro2 = np.linspace(0.0, 1.0, iter)
        # for ro2 in np.linspace(0.0, 1.0, 200000):
        dis_cofficient = 1 + k1*ro2 + k2*ro2*ro2
        ro2_after = np.sqrt(ro2) * (1 + k1*ro2 + k2*ro2*ro2)  ## 畸变之后的 rou
        # map_dist.append([(1 + k1*ro2 + k2*ro2*ro2), ro2_after])
        map_dist = np.stack([dis_cofficient,ro2_after])
        map_dist = np.moveaxis(map_dist,-1,0)

        z = np.linspace(0.0, 1.0, iter)
        z_after = np.sqrt(1 - z**2) / (z + mirror)
        z_dist = np.stack([z,z_after])
        z_dist = np.moveaxis(z_dist,-1,0)

        map_dist = torch.from_numpy(map_dist).cuda()
        z_dist = torch.from_numpy(z_dist).cuda()

        xys = []
        for i in range(self.H):
            xy = chunk(grid[:, i*self.W:(i+1)*self.W].cuda())
            xys.append(xy.permute(1, 0))  ##[1400,2]
        xys = torch.cat(xys, dim=0)

        
        z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
        isnan = z.isnan()
        z[isnan] = 1.
        left_fisheye_grid = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)

        
        ## 读取一帧图像，将color 信息和 相机的3D 点concat 在一起
        # img = self.image_at(img_idx,resolution_level=1).reshape(-1,3)
        # img = torch.from_numpy(img).cuda()
        # self.cam2img(left_fisheye_grid[:,:3],fisheye_idx='02',img=img)
        

        left_valid = left_fisheye_grid[:,:3]
        ## 产生有效的光线,将相机坐标系的光线ray_d 转化到 世界坐标系下面
        dirs = left_valid.to(self.device)
        dirs = dirs / torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(
        # [..., N_rays, 1, 3] * [..., 1, 3, 3]
        dirs[..., None, :] * self.poses[img_idx, None, :3, :3], -1
        ) 
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape)

        return rays_o.reshape(self.H,self.W,-1), rays_v.reshape(self.H,self.W,-1)
    
    def cam2img(self, points,fisheye_idx = None,img=None):
        if fisheye_idx == "02":
            k1 = self.intrinsic_fisheye_02['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_02['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_02['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_02['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_02['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_02['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_02['mirror_parameters']['xi']
            
        ## fisheye_03
        elif fisheye_idx == "03":
            k1 = self.intrinsic_fisheye_03['distortion_parameters']['k1']
            k2 = self.intrinsic_fisheye_03['distortion_parameters']['k2']
            gamma1 = self.intrinsic_fisheye_03['projection_parameters']['gamma1']
            gamma2 = self.intrinsic_fisheye_03['projection_parameters']['gamma2']
            u0 = self.intrinsic_fisheye_03['projection_parameters']['u0']
            v0 = self.intrinsic_fisheye_03['projection_parameters']['v0']
            mirror = self.intrinsic_fisheye_03['mirror_parameters']['xi']
    
        norm = torch.norm(points, dim=1, p=2)

        x = points[:,0] / norm
        y = points[:,1] / norm
        z = points[:,2] / norm

        x = x / (z+mirror)
        y = y / (z+mirror)


        ## 也就是说在Kitti360当中，仅仅考虑了 径向畸变，而没有考虑切向畸变
        ro2 = x*x + y*y
        x = x * (1 + k1*ro2 + k2*ro2*ro2)
        y = y * (1 + k1*ro2 + k2*ro2*ro2)

        x = gamma1*x + u0
        y = gamma2*y + v0

        x = torch.round(x).long()
        y = torch.round(y).long()      
        redner_img = torch.zeros(1400,1400,3)
        for i in np.arange(x.shape[0]):
            redner_img[x[i],y[i]] = img[i,:]

        cv.imwrite("orign_fisheye.png",img.reshape(1400,1400,-1).detach().cpu().numpy()*255)
        cv.imwrite("prject_fisheye.png",redner_img.detach().cpu().numpy()*255)
        exit()
        return

    
    
if __name__ == '__main__':
    from pyhocon import ConfigFactory
    conf_path = 'confs/kitti360.conf'
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    dataset = Kitti360_fisheye(conf['dataset'])

    rays = dataset.gen_random_rays_at(img_idx=0, batch_size=1024)
    rays = dataset.gen_rays_at(img_idx=0,resolution_level=1)