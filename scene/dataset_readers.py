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
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    file_id: int
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    object_mask: np.array
    background_ignore_mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, background_ignore_mask_folder=None):
    all_filenames = sorted([f[:-4] for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')])
    # for cam in cameras:
    #     cam.filename_id = all_filenames.index(cam.image_name)
    
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[DATA] Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        #interactive debugging smaller dataset hack:
        if images_folder.endswith("garden_small/images"):
            if int(extr.name[3:8]) > 7985:
                continue
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))#R input is C2W or transposed(GLM) W2C ----> Colmap R is W2C but gets transposed
        T = np.array(extr.tvec)#t input is in camera (i.e. W2C pose)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, f"[DATA] Colmap camera model '{intr.model}' not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if images_folder.endswith("_rgba"):#RGB images (used in COLMAP) are combined with alpha. RGBA images must have .png but the previous RGB might have been .jpg
            image_path = os.path.join(images_folder, image_name+'.png')
        image = Image.open(image_path)
        #prevent large number of open files
        tmp = np.array(image)
        image.close()

        if background_ignore_mask_folder is not None:
            background_ignore_mask_path = os.path.join(background_ignore_mask_folder, image_name + '.png')
            if os.path.exists(background_ignore_mask_path):
                background_ignore_mask = Image.open(background_ignore_mask_path)
                # Load alpha channel
                background_ignore_mask = np.array(background_ignore_mask)[:, :, 3]
                # PIL
                background_ignore_mask = Image.fromarray(background_ignore_mask)

            else:
                background_ignore_mask = None
        else:
            background_ignore_mask = None
            
        if len(tmp.shape) == 3 and tmp.shape[2] == 4:#[H,W,C] and C is rgba
            #DTU/2DGS/scanXXX/images
            object_mask = Image.fromarray(tmp[:, :, 3])#alpha channel of RGBA image
        elif len(tmp.shape) == 2:#[H,W], which is grayscale
            tmp = np.stack([tmp, tmp, tmp], axis=-1)#make image have 3 channels
            object_mask = None
        else:
            object_mask = None

        image = Image.fromarray(tmp)#training image is everything (default)

        file_id = all_filenames.index(image_name)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, object_mask=object_mask, background_ignore_mask=background_ignore_mask,
                              image_path=image_path, image_name=image_name, width=width, height=height, file_id=file_id)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    # print('cam infos', cam_infos)
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, bkg_images=None, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    background_ignore_mask_folder = os.path.join(path, bkg_images) if bkg_images is not None else None
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), 
                                           background_ignore_mask_folder=background_ignore_mask_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("[DATA] Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("[DATA] Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("[DATA] Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"[DATA] Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSimpleCameras(cam_intrinsics, images_folder, background_ignore_mask_folder=None):
    all_filenames = sorted([f[:-4] for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')])

    cam_infos = []
    for idx, image_name in enumerate(all_filenames):
        sys.stdout.write('\r')
        sys.stdout.write("[DATA] Reading camera {}/{}".format(idx+1, len(all_filenames)))
        sys.stdout.flush()

        # Intrinsics (manually supplied)
        intr = cam_intrinsics  # same for all images

        height = intr['height']
        width = intr['width']
        uid = intr.get('id', 0)  # optional id, default 0

        # Extrinsics: identity pose (camera at origin)
        R = np.eye(3)
        T = np.zeros(3)

        # Intrinsics → FOV
        if intr['model'] == "SIMPLE_PINHOLE":
            focal_length_x = intr['params'][0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr['model'] == "PINHOLE":
            focal_length_x = intr['params'][0]
            focal_length_y = intr['params'][1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise ValueError(f"[DATA] Camera model '{intr['model']}' not handled!")

        # Load image
        image_path = os.path.join(images_folder, image_name + '.png')  # assume .png for RGBA; adapt if needed
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, image_name + '.jpg')
        image = Image.open(image_path)

        # Convert to numpy temporarily
        tmp = np.array(image)
        image.close()

        # Extract object mask if RGBA
        if len(tmp.shape) == 3 and tmp.shape[2] == 4:
            object_mask = Image.fromarray(tmp[:, :, 3])
        elif len(tmp.shape) == 2:
            tmp = np.stack([tmp, tmp, tmp], axis=-1)
            object_mask = None
        else:
            object_mask = None

        # Re-load the image (now clean)
        image = Image.fromarray(tmp)

        if background_ignore_mask_folder is not None:
            background_ignore_mask_path = os.path.join(background_ignore_mask_folder, image_name + '.png')
            if os.path.exists(background_ignore_mask_path):
                background_ignore_mask = Image.open(background_ignore_mask_path)
                # Load alpha channel
                background_ignore_mask = np.array(background_ignore_mask)[:, :, 3] / 255.0
                background_ignore_mask = Image.fromarray(background_ignore_mask)
            else:
                background_ignore_mask = None
        else:
            background_ignore_mask = None

        # CameraInfo (reuse your existing class)
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            object_mask=object_mask,
            background_ignore_mask=background_ignore_mask if background_ignore_mask_folder else None,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            file_id=idx
        )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readSimpleSceneInfo(path, cam_intrinsics, images=None, eval=False, bkg_images=None, llffhold=8):
    reading_dir = "images" if images is None else images
    images_folder = os.path.join(path, reading_dir)

    # Use the simple camera reader (no extrinsics)
    background_ignore_mask_folder = os.path.join(path, bkg_images) if bkg_images is not None else None
    cam_infos_unsorted = readSimpleCameras(cam_intrinsics=cam_intrinsics, images_folder=images_folder, 
                                           background_ignore_mask_folder=background_ignore_mask_folder) 
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Point cloud is skipped because there's no sparse folder / point cloud
    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # if os.path.exists(ply_path):
    #     try:
    #         pcd = fetchPly(ply_path)
    #     except:
    #         pcd = None
    # else:
    print("[DATA] No point cloud initialization for hands.")
    pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Hand" : readSimpleSceneInfo,
}