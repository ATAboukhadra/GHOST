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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_mano import GaussianModelMano as GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.hand_init = "canonical"

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("[SCENE] Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sfm_rescaled")) and args.images != "hand_rgba":
            # print(args.source_path, args.images)
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, bkg_images=args.background_ignore_mask)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("[SCENE] Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif args.images == "hand_rgba":
            print("[SCENE] Found hand_rgba images, assuming hand dataset!")
            cam_intrinsics = {'model': 'SIMPLE_PINHOLE', 'width': 1920, 'height': 1080, 'params': [915]}  # fx (same for x and y)
            scene_info = sceneLoadTypeCallbacks["Hand"](args.source_path, cam_intrinsics, args.images, args.eval, bkg_images=args.background_ignore_mask)
        else:
            assert False, "[SCENE] Could not recognize scene type!"

        if not self.loaded_iter and args.images != "hand_rgba":
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("[SCENE] Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("[SCENE] Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud_deformed.ply"))
        else:
            if args.hand == 'none':
                print("[SCENE] Creating Gaussians from SfM point cloud")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)#SfM pcd of object
                if args.use_obj_prior:
                    # print(f"[SCENE] Loading Gaussians of a trained object model from '{args.object_path}'")
                    self.gaussians.load_prior(os.path.join(args.source_path, 'icp/prior_obj_scaled.obj'))
            elif args.hand in ['right', 'left']:
                print(f"[SCENE] Creating Gaussians from {args.hand} hand mesh")
                self.gaussians.load_ply(os.path.join(args.source_path, self.hand_init, f"gaussians_{args.hand}_{args.gaussians_per_edge}.ply"), hand=True)#Gaussians from Mesh
                self.gaussians.load_transforms(os.path.join(args.source_path, f"{args.hand}_transformations.pth"), args.gaussians_per_edge)
                if not args.object_path == "":
                    print(f"[SCENE] Loading Gaussians of a trained object model from '{args.object_path}'")
                    self.gaussians.load_ply(args.object_path)#has to be the full path to .ply file
                self.gaussians.make_parameters()
            elif args.hand in ['both']:
                print(f"[SCENE] Creating Gaussians from both hand meshes")
                self.gaussians.load_ply(os.path.join(args.source_path, self.hand_init, f"gaussians_{'right'}_{args.gaussians_per_edge}.ply"), hand=True)#Gaussians from Mesh
                self.gaussians.load_transforms(os.path.join(args.source_path, f"{'right'}_transformations.pth"), args.gaussians_per_edge)
                self.gaussians.load_ply(os.path.join(args.source_path, self.hand_init, f"gaussians_{'left'}_{args.gaussians_per_edge}.ply"), hand=True)#Gaussians from Mesh
                self.gaussians.load_transforms(os.path.join(args.source_path, f"{'left'}_transformations.pth"), args.gaussians_per_edge)
                if not args.object_path == "":
                    print(f"[SCENE] Loading Gaussians of a trained object model from '{args.object_path}'")
                    self.gaussians.load_ply(args.object_path)#has to be the full path to .ply file
                self.gaussians.make_parameters()
            else:
                raise NotImplementedError(f"No handle for argument hand with value '{args.hand}'!")

    def save(self, iteration, frame_num=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), frame_num=frame_num)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]