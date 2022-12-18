import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from camera_pose_visualizer import CameraPoseVisualizer

class Kitti360_Data:
    def __init__(self,conf):
        super(Kitti360_Data, self).__init__()
        self.device = 'cuda'
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.poses_dir = conf.get_string('poses_dir')
        self.intrinsic_dir = conf.get_string('intrinsic_param')
        self.image_dir = os.path.join(self.data_dir,'2013_05_28_drive_0000_sync')
        self.camera_dict = os.path.join(self.data_dir, self.poses_dir)

        self.came2world = {}
        self.start_index = 3353
        num_images = int(conf.get_string('num_images'))

        '''  load intrinstic   '''
        self.intrinsic = os.path.join(self.data_dir, self.intrinsic_dir)
        K = None
        with open(self.intrinsic, 'r') as f:
            lines = f.readlines()
            for line in lines:
                value = list(line.strip().split())
                if value[0] == 'P_rect_00:':
                    K = [float(x) for x in value[1:]]
                    K = np.array(K).reshape(3, 4)
                elif value[0] ==  'R_rect_01:':
                    R_rect_01 = np.eye(4)
                    R_rect_01[:3, :3] = np.array(value[1:]).reshape(3, 3).astype(np.float)


        '''Load extrinstic matrix'''
        CamPose_00 = {}
        CamPose_01 = {}
        extrinstic_file = os.path.join(self.data_dir, os.path.join('data_poses', '2013_05_28_drive_0000_sync'))
        cam2world_file_00 = os.path.join(extrinstic_file, 'cam0_to_world.txt')
        pose_file = os.path.join(extrinstic_file, 'poses.txt')

        ''' Camera_00  to world coordinate '''
        with open(cam2world_file_00, 'r') as f:
            lines = f.readlines()
            for line in lines:
                lineData = list(map(float, line.strip().split()))
                CamPose_00[lineData[0]] = np.array(lineData[1:]).reshape(4, 4)

        ''' Camera_01 to world coordiante '''
        CamToPose_01 = self.loadCameraToPose(os.path.join(os.path.join(self.data_dir, 'calibration'), 'calib_cam_to_pose.txt'))
        poses = np.loadtxt(pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
            pp = np.matmul(pose, CamToPose_01)
            CamPose_01[frame] = np.matmul(pp, np.linalg.inv(R_rect_01))


        ''' Load Image '''
        image_00 = os.path.join(self.image_dir, 'image_00/data_rect')
        image_01 = os.path.join(self.image_dir, 'image_01/data_rect')
        self.all_images = []
        self.all_poses = []
        for idx in range(self.start_index, self.start_index + num_images, 1):
            ## read image_00
            image = cv.imread(os.path.join(image_00, "{:010d}.png").format(idx)) / 256.0
            self.all_images.append(image)
            self.all_poses.append(CamPose_00[idx])

            ## read image_01
            image = cv.imread(os.path.join(image_01, "{:010d}.png").format(idx)) / 256.0
            self.all_images.append(image)
            self.all_poses.append(CamPose_01[idx])

        self.images = np.array(self.all_images).astype(np.float32)
        self.images = torch.from_numpy(self.images).cpu()
        self.masks = torch.ones_like(self.images)
        self.n_images = len(self.all_images)
        self.poses = np.stack(self.all_poses).astype(np.float32)
        ''' Normalize the pose matrix '''
        self.poses = self.Normailize_T(self.poses)
        self.poses = torch.from_numpy(self.poses).to(self.device)
        self.object_bbox_min = np.array([-0.1, -0.1, -2.01])
        self.object_bbox_max = np.array([0.1, 0.1, 2.01])


        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.focal = K[0][0]
        self.image_pixels = self.H * self.W
        self.intrinsics_all = np.array([
            [self.focal, 0, 0.5 * self.W, 0],
            [0, self.focal, 0.5 * self.H, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.intrinsics_all = np.stack([self.intrinsics_all.astype(np.float32) for i in range(self.images.shape[0])])
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all).to(self.device)
        self.intrinsics_inv = torch.inverse(self.intrinsics_all)

        '''     Visual Camera Pose 

        '''
        visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])
        for i in np.arange(self.poses.shape[0]):
            if i % 1 == 0:
                print(self.poses[i][:3,3])
                visualizer.extrinsic2pyramid(self.poses[i])
        visualizer.show()
        print("Data load end")

    def loadCameraToPose(self,filename):
        # open file
        Tr = {}
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                lineData = list(line.strip().split())
                if lineData[0] == 'image_01:':
                    data = np.array(lineData[1:]).reshape(3, 4).astype(np.float)
                    data = np.concatenate((data, lastrow), axis=0)
                    Tr[lineData[0]] = data

        return Tr['image_01:']

    def Normailize_T(self, poses):
        for i, pose in enumerate(poses):
            if i == 0:
                inv_pose = np.linalg.inv(pose)
                poses[i] = np.eye(4)
            else:
                poses[i] = np.dot(inv_pose, poses[i])

        '''New Normalization '''
        scale = poses[-1, 2, 3]
        print(f"scale:{scale}\n")
        for i in range(poses.shape[0]):
            poses[i, :3, 3] = poses[i, :3, 3] / scale
            print(poses[i])

        return poses

    def gen_random_rays_at(self, img_idx, batch_size, pixels_y=None, pixels_x=None):
        """
        Generate random rays at world space from one camera.
        """
        if pixels_y is not None:
            pixels_x = torch.from_numpy(pixels_x).to('cuda')
            pixels_y = torch.from_numpy(pixels_y).to('cuda')
        else:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3

        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_inv[img_idx, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.poses[img_idx, None, None, :3, :3],
                              rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.poses[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def image_at(self, idx,resolution_level):
        return self.all_images[idx]

    def near_far_from_sphere(self, rays_o, rays_d):
        near, far = 0., 6.
        return near,far

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.poses[idx_0, :3, 3] * (1.0 - ratio) + self.poses[idx_1, :3, 3] * ratio
        pose_0 = self.poses[idx_0].detach().cpu().numpy()
        pose_1 = self.poses[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
