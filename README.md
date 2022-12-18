# Neus_kitti360

## 双目版本
参考Nerf_kitti360 写了load_kitti.py 脚本，使用双目图像

## PE 的Normalize 方法
没有直接给pts/100 进行归一化，而是将选取最远的相机位姿Z的平移归一化到1
near=0 far=6

## Train 记录
选择num_frame =20 进行训练，psnr可以训练到26-27之间

## Bug 记录 
当使用Kitti360的单目版本时候，Neus无法recover correct geometry,所有的pixel 的深度值几乎一样，使用双目正确恢复出geometry

*单目recover出深度图：*

![image](https://user-images.githubusercontent.com/111415805/208279798-1e9b3dc1-fe99-4cd5-ab2c-a20b4d078eda.png)

*双目recover出深度图：*
![image](https://user-images.githubusercontent.com/111415805/208279815-74aa5a2a-7fb7-4210-916d-8381d611ff02.png)
