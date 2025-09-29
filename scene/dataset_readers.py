#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import imageio
import skimage
import cv2
from scene.gaussian_model import BasicPointCloud
from glob import glob
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    prcppoint: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    mono: np.array
    K: np.array =None
    s0: np.array=None
    s1: np.array=None
    s2: np.array=None
    azimuth: np.array = None
    dop: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    need_resize: bool = True

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normal=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz) if normal is None else normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    alpha = imageio.imread(path, pilmode='F')
    alpha = skimage.img_as_float32(alpha) / 255
    return alpha
def readSMVP3DSceneInfo(path, white_background, eval):
    print("Reading Training")
    train_cam_infos = readSMVP3D(path, white_background, "train")    
    if eval:
        train_cam_infos = train_cam_infos
        test_cam_infos = train_cam_infos
        # train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in train_cams]

        # test_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in test_cams]
    else:
        train_cam_infos = train_cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 100_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 1.2
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        xyz = normal * 0.5 #- rand_scale / 2

        rand_scale *= 2
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)



    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           need_resize = True)
    del train_cam_infos, test_cam_infos
    return scene_info

def readSMVP3D(path, white_background, name):
   
    image_dir = '{0}/train'.format(path)
    image_paths = sorted(glob_imgs(image_dir))
    cam_file = '{0}/cameras.npz'.format(path)

    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    try:
        scale_mats = [camera_dict['scale_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
        world_mats = [camera_dict['world_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
    except:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)


    rgb_images = []
    for i in image_paths:
        rgb = load_rgb(i)
        rgb_images.append(rgb)


    azimuth_path = sorted(os.listdir(os.path.join(path, name+"_input_azimuth_maps")))

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        t = t[:3, :] / t[3:, :]
        K = K / K[2, 2]

        T = -R @ t  
        T = T[:, 0]
        R = R.T
        R[:2,:3] *= -1


        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])
        image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255
        try:
            s0_file = os.path.join(path, 'train_images_stokes', f'{i+1:02d}_s0.hdr')
            s0p1_file = os.path.join(path,'train_images_stokes', f'{i+1:02d}_s0p1.hdr')
            s0p2_file = os.path.join(path,'train_images_stokes', f'{i+1:02d}_s0p2.hdr')
            s0 = imageio.imread(s0_file)
            s0p1 = imageio.imread(s0p1_file)
            s0p2 = imageio.imread(s0p2_file)
            s0 = skimage.img_as_float32(s0)
            s0p1 = skimage.img_as_float32(s0p1)
            s0p2 = skimage.img_as_float32(s0p2)
            s0 ,s1, s2 = s0/2, (s0p1-s0)/2, (s0p2-s0)/2
            s0 = s0.transpose(2,0,1)
            s1 = s1.transpose(2,0,1)
            s2 = s2.transpose(2,0,1)
        except:
            s0_file = os.path.join(path, "s0", f'{i:04d}.npy')
            s0 = np.load(s0_file) 
            s1_file = os.path.join(path, "s1", f'{i:04d}.npy')
            s1 = np.load(s1_file) 
            s2_file = os.path.join(path, "s2", f'{i:04d}.npy')
            s2 = np.load(s2_file) 
            s0,s1,s2 = s0.transpose(2,0,1),s1.transpose(2,0,1),s2.transpose(2,0,1)

        azimuth_file= os.path.join(path, name+"_input_azimuth_maps", azimuth_path[i])
        azimuth = imageio.imread(azimuth_file)
        azimuth = skimage.img_as_float32(azimuth)
        mask = azimuth[...,[-1]].transpose([2, 0, 1]).astype(np.float32)
        azimuth = (0.5*(np.arctan2(s2.mean(0),s1.mean(0)))[None,...] + np.pi/2).clip(0, np.pi)

        mono = None

        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if image.shape[-1] == 4:
            alpha = image[..., 3:] / 255
            image = image[..., :3]
        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        prcppoint = K[:2, 2] / image.size[:2]

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],s0=s0,s1=s1,s2=s2,K=K,  
                         mask = mask, mono=mono))
    return cam_infos


def readREALWORLDSceneInfo(path, white_background, eval):
    print("Reading Training")
    train_cam_infos = readREALWORLD(path, white_background, "train")    
    if eval:
        train_cam_infos = train_cam_infos
        test_cam_infos = train_cam_infos
        # train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in train_cams]

        # test_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in test_cams]
    else:
        train_cam_infos = train_cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 100_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 1.2
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        xyz = normal * 0.5 #- rand_scale / 2

        rand_scale *= 2
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)



    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           need_resize = True)
    del train_cam_infos, test_cam_infos
    return scene_info

def readREALWORLD(path, white_background, name):
   
    image_dir = '{0}/train'.format(path)
    image_paths = sorted(glob_imgs(image_dir))
    cam_file = '{0}/cameras.npz'.format(path)

    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    try:
        scale_mats = [camera_dict['scale_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
        world_mats = [camera_dict['world_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
    except:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)


    rgb_images = []
    for i in image_paths:
        rgb = load_rgb(i)
        rgb_images.append(rgb)

    azimuth_path = sorted(os.listdir(os.path.join(path, name+"_input_azimuth_maps")))

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        t = t[:3, :] / t[3:, :]
        K = K / K[2, 2]

        T = -R @ t  
        T = T[:, 0]
        R = R.T
        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])
        image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255
        try:
            s0_file = os.path.join(path, 'train_images_stokes', f'{i+1:02d}_s0.hdr')
            s0p1_file = os.path.join(path,'train_images_stokes', f'{i+1:02d}_s0p1.hdr')
            s0p2_file = os.path.join(path,'train_images_stokes', f'{i+1:02d}_s0p2.hdr')
            s0 = imageio.imread(s0_file)
            s0p1 = imageio.imread(s0p1_file)
            s0p2 = imageio.imread(s0p2_file)
            s0 = skimage.img_as_float32(s0)
            s0p1 = skimage.img_as_float32(s0p1)
            s0p2 = skimage.img_as_float32(s0p2)
            s0 ,s1, s2 = s0/2, (s0p1-s0)/2, (s0p2-s0)/2
            s0 = s0.transpose(2,0,1)
            s1 = s1.transpose(2,0,1)
            s2 = s2.transpose(2,0,1)
        except:
            s0_file = os.path.join(path, "s0", f'{i:04d}.npy')
            s0 = np.load(s0_file) 
            s1_file = os.path.join(path, "s1", f'{i:04d}.npy')
            s1 = np.load(s1_file) 
            s2_file = os.path.join(path, "s2", f'{i:04d}.npy')
            s2 = np.load(s2_file) 
            s0,s1,s2 = s0.transpose(2,0,1),s1.transpose(2,0,1),s2.transpose(2,0,1)

        azimuth_file= os.path.join(path, name+"_input_azimuth_maps", azimuth_path[i])
        azimuth = imageio.imread(azimuth_file)
        azimuth = skimage.img_as_float32(azimuth)
        mask = azimuth[...,[-1]].transpose([2, 0, 1]).astype(np.float32)
        azimuth = (0.5*(np.arctan2(s2.mean(0),s1.mean(0)))[None,...] + np.pi/2).clip(0, np.pi)
        # azimuth = azimuth[..., [0]].transpose([2, 0, 1]).astype(np.float32) * np.pi

        
        mono = None

        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if image.shape[-1] == 4:
            alpha = image[..., 3:] / 255
            image = image[..., :3]
        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        prcppoint = K[:2, 2] / image.size[:2]

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, prcppoint=prcppoint, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],s0=s0,s1=s1,s2=s2,K=K,  
                         mask = mask, mono=mono))
    return cam_infos


sceneLoadTypeCallbacks = {
    "SMVP3D": readSMVP3DSceneInfo,
    "REALWORLD": readREALWORLDSceneInfo
}


if __name__ == '__main__':
    None
    