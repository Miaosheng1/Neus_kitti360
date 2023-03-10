import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.load_kitti360 import Kitti360_Data
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from camera_pose_visualizer import CameraPoseVisualizer

_color_map_depths = np.array([
    [0, 0, 0],          # 0.000
    [0, 0, 255],        # 0.114
    [255, 0, 0],        # 0.299
    [255, 0, 255],      # 0.413
    [0, 255, 0],        # 0.587
    [0, 255, 255],      # 0.701
    [255, 255,  0],     # 0.886
    [255, 255,  255],   # 1.000
    [255, 255,  255],   # 1.000
]).astype(float)
_color_map_bincenters = np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000, # doesn't make a difference, just strictly higher than 1
])


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, render_pixel=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)  ## 将conf 文件里面的 “CASE_NAME” 换成 parser 里面的 case
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        # self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.img_dir = self.conf['general.img_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        self.dataset = Kitti360_Data(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)  ## warm_up = 5000
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.render_pixel = render_pixel
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        ## SDF 网络是全连接网络
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.img_dir,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        ## 如果是 is_continue 设置为True，找到最近的一次训练权重，并且加载进去
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug  备份代码和文件
        # if self.mode[:5] == 'train':
        #     self.file_backup()

    def train(self):
        torch.set_printoptions(precision=4, sci_mode=False)
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()  ## 在打乱的 N 张照片中，重新获取排列顺序

        ''' Visualize Camera Pose'''
        # poses = self.dataset.pose_all
        # visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 5])
        # for i in np.arange(poses.shape[0]):
        #     if i % 2 == 0:
        #         visualizer.extrinsic2pyramid(poses[i], 'y', 10)
        # visualizer.show()


        # if res_step == 0:
        #     self.validate_image(idx=45)

        for iter_i in tqdm(range(res_step)):

            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} psnr={}'.format(self.iter_step, loss, psnr))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        os.makedirs(self.img_dir, exist_ok=True)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        if self.render_pixel:
            ## 描述像素在orign_image
            orign_imgae = self.dataset.image_at(idx, resolution_level=1).astype(np.float32)
            color_cv = orign_imgae / 256.0
            pixel_x = np.array([100, 650, 720, 700,650])
            pixel_y = np.array([300, 230, 820, 800,300])
            p = np.stack([pixel_x,pixel_y],axis=-1).astype(np.int)
            for point in p:
                cv.circle(orign_imgae,point,radius=5,color=(0,0,255),thickness=4)
            cv.imwrite(os.path.join(self.img_dir,"draw_point.png"),orign_imgae)
            data = self.dataset.gen_random_rays_at(idx, self.batch_size, pixels_y=pixel_y, pixels_x=pixel_x)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            self.batch_size = 1
        else:
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level= self.validate_resolution_level)  ## 降采样4 倍生成的 tensor(300,400,3)
            H, W, _ = rays_o.shape

        ## 获取Ray_o 和 Ray_d
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)  ## 得到300*400=120000 个像素的原点（进行了拉直）
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []


        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)


            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out



        img_fine = None
        if len(out_rgb_fine) > 0 and not self.render_pixel:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        elif self.render_pixel:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([1, len(pixel_x), 3]) * 256).clip(0, 255)
            img_true = true_rgb.detach().cpu().numpy()[None, :, :].astype(np.float64) * 256.0

            cv.imwrite(os.path.join(self.img_dir,"render_pixel.png"), np.concatenate((img_fine, img_true), axis=0))

        # normal_img = None
        # if len(out_normal_fine) > 0:
        #     normal_img = np.concatenate(out_normal_fine, axis=0)
        #     rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
        #     normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
        #                   .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        '{}_{}.png'.format( i, idx)),
                           ## 这里的 concatenate 函数的操作是 将 inferenc 的图片img_fine[..., i] 和 原始的图像 竖着并在一起，并使用cv.imwrite 保存
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=self.validate_resolution_level)*256.0]))
        # print("finish")
        # if len(out_normal_fine) > 0:
        #     cv.imwrite(os.path.join(self.base_exp_dir,
        #                             'normals',
        #                             '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
        #                normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_depth =[]
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            out_depth.append(render_out['depth_map'].detach().cpu().numpy())

            # import matplotlib.pyplot as plt
            # random_ray = np.random.randint(0,100,10)
            #
            # for i in random_ray:
            #     plt.plot(render_out['weights'][i,:128].detach().cpu().numpy())
            # plt.savefig("weight.png")
            # plt.close("all")

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)

        pred_depth = np.concatenate(out_depth, axis=0).reshape([H, W]).astype(np.float)
        # dd =pred_depth/pred_depth.max()
        depth = cv.applyColorMap(cv.convertScaleAbs(((pred_depth/pred_depth.max()) * 255).astype(np.uint8),alpha=2), cv.COLORMAP_JET)
        # depth = self.color_depth_map(pred_depth)

        return img_fine,depth

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.2):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # if world_space:
        #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        depth_img = []
        n_frames = 10
        for i in range(n_frames):
            print(i)
            img,depth = self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=2)
            images.append(img)

            # depth = cv.applyColorMap(depth,cv.COLORMAP_JET)
            depth_img.append(depth)
        # for i in range(n_frames):
        #     images.append(images[n_frames - i - 1])
        render_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(render_dir, exist_ok=True)
        for i in range(n_frames):
            cv.imwrite(os.path.join(render_dir,"render{:03d}.png".format(i)),images[i])
            cv.imwrite(os.path.join(render_dir, "depth{:03d}.png".format(i)), depth_img[i])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')

        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                              '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                 fourcc, 30, (w, h))
        
        for image in images:
            writer.write(image)
       
        writer.release()

    def color_depth_map(self,depths, scale=None):
        """
        Color an input depth map.

        Arguments:
            depths -- HxW numpy array of depths
            [scale=None] -- scaling the values (defaults to the maximum depth)
        Returns:
            colored_depths -- HxWx3 numpy array visualizing the depths
        """
        # print(type(depths))
        if scale is None:
            scale = depths.max()

        values = np.clip(depths.flatten() / scale , 0, 1)
        # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
        lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
        lower_bin_value = _color_map_bincenters[lower_bin]
        higher_bin_value = _color_map_bincenters[lower_bin + 1]
        alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
        colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
            lower_bin + 1] * alphas.reshape(-1, 1)
        return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)


if __name__ == '__main__':
    print('Hello Miao, This is benchmark version!')
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")  ## 是否Find 之前训练的weight 继续训练
    parser.add_argument('--gpu', type=int, default=0)  ## 调用哪一个GPU
    parser.add_argument('--case', type=str, default='')  ## 数据集文件夹
    parser.add_argument("--render_pixel", default=False, action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.render_pixel)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
