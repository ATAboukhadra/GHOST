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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, matrix_to_quaternion, quaternion_multiply
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation, polar_decomposition
# from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import einops
from .angle_helpers import transform_shs_batched

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.seen = torch.empty(0)
        self.transforms = None
        # self.transforms2 = None
        self.right_hand_indices = None
        self.left_hand_indices = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.binding = None  # gaussian index to face index
        self.binding_counter = None  # number of points bound to each face

        # self.gaussian_visibility_count = None  # (N,) int: how many frames each Gaussian is visible in
        self.object_poses = {}
        # self.is_binding = False


        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("[GAUSS] Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gaussian_visibility_count = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.int32)

        if training_args.prune_unseen:
            self.seen = torch.ones(self.get_xyz.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)#init points are assumed to be all visible

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # xyz = self.update_gaussians_position().detach().cpu().numpy()#saving current deformed hand
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # rotation = self.update_gaussians_rotation().detach().cpu().numpy()#saving current deformed hand

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # print(attributes[:self.limit1].shape)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # saving only the canonical hand
        # cano_attributes = attributes[self.right_hand_indices]
        # elements = np.empty(cano_attributes.shape[0], dtype=dtype_full)
        # elements[:] = list(map(tuple, cano_attributes))
        # el = PlyElement.describe(elements, 'vertex')   
        # PlyData([el]).write(path[:-4]+"_canonical.ply")     
        
        #saving current deformed hand
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # Compute rotated sh

        # attributes = np.concatenate((self.get_gaussians_position().detach().cpu().numpy(), normals, f_dc, 
        #                              self.rotate_sh_batched().detach().cpu().numpy().reshape(len(self._xyz), -1), opacities, scale, 
        #                              self.get_gaussians_rotation().detach().cpu().numpy()), axis=1)
        # print(opacities[-30083:])
        attributes = np.concatenate((self.get_gaussians_position().detach().cpu().numpy(), normals, f_dc, f_rest, opacities, scale, self.get_gaussians_rotation().detach().cpu().numpy()), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path[:-4]+"_deformed.ply")#"point_cloud.ply"
        if self.binding is None:
            return

        def save_subset(mask, suffix):
            elements = np.empty(np.sum(mask), dtype=dtype_full)
            elements[:] = list(map(tuple, attributes[mask]))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(f"{path[:-4]}_{suffix}.ply")

        
        binding_cpu = self.binding.detach().cpu().numpy()
        num_faces_per_hand = 1538  # adjust based on MANO

        is_right = (binding_cpu >= 0) & (binding_cpu < num_faces_per_hand)
        # is_left = (binding_cpu >= num_faces_per_hand) & (binding_cpu < 2 * num_faces_per_hand)
        is_object = (binding_cpu == self.identity_binding_index)

        # save_subset(is_right, "right_hand")
        # save_subset(is_left, "left_hand")
        save_subset(is_object, "object")

        # Save binding list for reference
        np.save(f"{path[:-4]}_binding.npy", binding_cpu)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, hand=False):
        plydata = PlyData.read(path)
        # print(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # print(hand, "xyz shape", xyz.shape)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if not hand:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            #max_radii2D is not restored from checkpoint for now
            max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
            # Object binding
            num_object_gaussians = xyz.shape[0]
            object_binding = torch.full((num_object_gaussians,), self.identity_binding_index, device="cuda", dtype=torch.int32)
            self.binding = torch.cat((self.binding, object_binding), dim=0)

        else:
            features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            #max_radii2D cannot be restored from checkpoint
            max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")

        if not hand:
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # ##########################################################################TODO temp for when .ply is created with 3 scales instead of 2
            # scales = scales[:, :2]
            # ##########################################################################TODO temp for when .ply is created with 3 scales instead of 2
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(xyz).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2).cpu().numpy()

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        ##########################################################################TODO temp for when .ply is created with rotation matrix instead of quad
        if hand:
            rots = matrix_to_quaternion(torch.from_numpy(rots.reshape(-1, 3, 3))).numpy()
        ##########################################################################TODO temp for when .ply is created with rotation matrix instead of quad

        if len(self._xyz) == 0:
            self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
            self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
            self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
            self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
            self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
            self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
            self.max_radii2D = max_radii2D
        else:
            self._xyz = torch.cat([self._xyz, torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)], dim=0)
            self._features_dc = torch.cat([self._features_dc, torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)], dim=0)
            self._features_rest = torch.cat([self._features_rest, torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)], dim=0)
            self._opacity = torch.cat([self._opacity, torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)], dim=0)
            self._scaling = torch.cat([self._scaling, torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)], dim=0)
            self._rotation = torch.cat([self._rotation, torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)], dim=0)
            self.max_radii2D = torch.cat([self.max_radii2D, max_radii2D], dim=0)
        
        self._scaling_init = self._scaling.detach().clone()

        self.active_sh_degree = self.max_sh_degree#TODO how to handle for Gaussians and already trained object?

    def make_parameters(self):
        self._xyz = nn.Parameter(self._xyz)
        self._features_dc = nn.Parameter(self._features_dc)
        self._features_rest = nn.Parameter(self._features_rest)
        self._opacity = nn.Parameter(self._opacity)
        self._scaling = nn.Parameter(self._scaling)
        self._rotation = nn.Parameter(self._rotation)

    def load_transforms(self, path, gaussians_per_edge):
        num_hand_faces = 1538
        self.num_gaussians_per_face = (gaussians_per_edge * (gaussians_per_edge-1)) // 2 + gaussians_per_edge #assumed to always be the same Gaussian density
        self.limit1 = int(num_hand_faces * self.num_gaussians_per_face)
        self.limit2 = int(self.limit1 + num_hand_faces * self.num_gaussians_per_face)
        self.binding = torch.arange(self.limit1, device="cuda", dtype=torch.int32) // self.num_gaussians_per_face
        self.binding_counter = torch.ones(num_hand_faces, dtype=torch.int32, device="cuda") * int(self.num_gaussians_per_face)
    
        if self.transforms == None:
            self.transforms = torch.load(path)#[T, F, 4, 4], with T=#frames, f=#faces
        else:
            self.transforms = torch.cat((self.transforms, torch.load(path)), dim=1)
            self.binding = torch.cat((self.binding, torch.arange(self.limit1, self.limit2, device="cuda", dtype=torch.int32) // self.num_gaussians_per_face), dim=0)
            self.binding_counter = torch.cat((self.binding_counter, torch.ones(num_hand_faces, dtype=torch.int32, device="cuda") * int(self.num_gaussians_per_face)), dim=0)

        num_frames = self.transforms.shape[0]
        identity = torch.eye(4, device="cuda").unsqueeze(0).repeat(num_frames, 1, 1).unsqueeze(1)
        self.transforms = torch.cat((self.transforms, identity), dim=1)#[T, F+1, 4, 4]
        self.identity_binding_index = self.transforms.shape[1] - 1  # This is the index of the identity transform

    def set_image_transform(self, image_id, cam):
        W2C = np.eye(4)
        W2C[:3, :3] = np.transpose(cam.R)#R is C2W
        W2C[:3,  3] = cam.T#T is W2C
        C2W = torch.from_numpy(np.linalg.inv(W2C)).float().to(self.transforms.device)#transforms are float and on cuda

        #select the transformations corresponding to the current image id and assign to each Gaussian their transformation based on which face it belongs to
        self.transforms_image = self.transforms[image_id, self.binding.long()]#[T, F, 4, 4] -> [N, 4, 4], where T=num_images, F=num_faces, N=num_gaussians
        self.transforms_image = torch.matmul(C2W, self.transforms_image)
        # Replace the object transformation with identity
        object_gaussians_mask = self.binding == self.identity_binding_index
        self.transforms_image[object_gaussians_mask] = torch.eye(4, device=self.transforms_image.device).unsqueeze(0).repeat(object_gaussians_mask.sum(), 1, 1)
        # print(self.transforms_image)
        self.transforms_image_quat = matrix_to_quaternion(self.transforms_image[:, :3, :3])
        
        self.is_grasping = False
        if self.transforms is not None and len(self.object_poses) > 0:
            self.is_grasping = self.detect_grasping(image_id, cam)

    def detect_grasping(self, image_id, cam):
        # This assumes a static-camera setup where the object can only move when the hand is grasping it
        # To extend this to non-stationary camera, we need to compute the relative change in pose for both the hand and the object 
        # and check if the object is moving relative to the hand

        if int(image_id) not in self.object_poses.keys():
            transformation = torch.eye(4, device="cuda")
            transformation[:3, :3] = torch.from_numpy(cam.R).float().to(self.transforms.device)
            transformation[:3, 3] = torch.from_numpy(cam.T).float().to(self.transforms.device)
            self.object_poses[int(image_id)] = transformation
            return False
        else:
            if int(image_id) - 1 not in self.object_poses.keys():
                return False
            else:
                transformation = self.object_poses[int(image_id)]
                prev_transformation = self.object_poses[int(image_id) - 1]
                # Compute difference between transformations
                diff_transformation = torch.matmul(transformation, torch.inverse(prev_transformation))
                r_diff = diff_transformation[:3, :3]
                trace = torch.trace(r_diff)
                cos_theta = (trace - 1) / 2
                cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
                angle_rad = torch.acos(cos_theta_clamped)
                angle_deg = torch.rad2deg(angle_rad)

                if angle_deg > 1.0:
                    # print(f'object is moving {image_id}:', angle_deg)
                    return True
                else:
                    # print(f'object is not moving {image_id}:', angle_deg)
                    return False

                # print(f'difference in transformation {image_id}:', diff_transformation)
                # Check if the difference is significant i.e. if the object is moving
                # compute how close to identity the transformation is

    def get_sh(self):
        # print(self.active_sh_degree)
        if self.transforms is not None and self.active_sh_degree <= 3:
            features = self.rotate_sh_batched()
            features = torch.cat((self._features_dc, features), dim=1)
            return features
        
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
        
    def get_gaussians_position(self):
        if self.transforms is not None:
            Ms, translations = self.transforms_image[:, :3, :3], self.transforms_image[:, :3, 3:]  # [N, 3, 3], [N, 3, 1]
            xyz = (Ms @ self._xyz.unsqueeze(-1) + translations).squeeze(-1)
            # Compute the hand center using the binding
            self.object_center = torch.mean(xyz[self.binding == self.identity_binding_index], dim=0, keepdim=True)
            self.hand_center = torch.mean(xyz[self.binding != self.identity_binding_index], dim=0, keepdim=True)
            # print(hand_center, object_center)

            return xyz
        return self._xyz
    
    def get_gaussians_rotation(self):
        if self.transforms is not None:
            quad = quaternion_multiply(self.transforms_image_quat, self._rotation)
            return quad
        return self._rotation

    def rotate_sh_batched(self):
        # Convert quaternions to rotation matrices
        R_matrices = np.stack([
            R.from_quat(q.detach().cpu().numpy()).as_matrix()
            for q in self.transforms_image_quat
        ], axis=0)  # (N, 3, 3)

        R_matrices = torch.from_numpy(R_matrices).to(self._features_rest.device, dtype=self._features_rest.dtype)

        rotated = transform_shs_batched(self._features_rest, R_matrices)
        return rotated

    # def get_gaussians_sh(self):
    #     if self.transforms is not None:
    #         sh = torch.zeros((self._features_dc.shape[0], 3, (self.max_sh_degree + 1) ** 2), device="cuda")
    #         sh[:, :, 0] = self._features_dc
    #         sh[:, :, 1:] = self._features_rest
    #         sh = torch.matmul(self.transforms_image, sh)
    #         return sh
    #     return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    # def get_gaussians_opacity(self):
    #     if self.transforms is not None:
    #         opacities = torch.zeros((self._opacity.shape[0], 1), device="cuda")
    #         opacities[:, 0] = self._opacity
    #         opacities = torch.matmul(self.transforms_image, opacities)
    #         return opacities
    #     return self._opacity

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):

        if self.binding is not None:
        
            binding_to_prune = self.binding[mask]
            hand_mask = binding_to_prune != self.identity_binding_index
            binding_to_prune_hand = binding_to_prune[hand_mask]

            counter_prune = torch.zeros_like(self.binding_counter)
            counter_prune.scatter_add_(0, binding_to_prune_hand.long(), torch.ones_like(binding_to_prune_hand, dtype=torch.int32, device="cuda"))
            mask_redundant = (self.binding_counter - counter_prune) > 0

            # Get absolute indices to update
            mask_indices = mask.nonzero(as_tuple=False).squeeze(1)
            hand_indices_in_mask = mask_indices[hand_mask]

            mask_clone = mask.clone()
            mask_clone[hand_indices_in_mask] = mask_redundant[binding_to_prune_hand]

            mask = mask_clone

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.binding is not None:
            self.binding = self.binding[valid_points_mask]
            # print number of hand points and object points
            num_hand_points = torch.sum(self.binding != self.identity_binding_index)
            num_object_points = torch.sum(self.binding == self.identity_binding_index)
            # print(f"Number of hand points: {num_hand_points}, Number of object points: {num_object_points}")

        if self.seen.numel():#seen is not empty -> we are tracking seen status
            self.seen = self.seen[valid_points_mask]

    def prune_unseen(self):
        if self.seen.sum() < self.seen.numel():#save many operations if all gaussians were visible
            self.prune_points(~self.seen)
        # self.seen = torch.zeros(self.seen.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)
        self.clear_seen_status()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.seen.numel():#seen is not empty -> we are tracking seen status
            self.seen = torch.cat((self.seen, torch.ones(new_opacities.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)), dim=0)#points added during densification have been seen

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if self.binding is not None:

            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding.long(), torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self.binding is not None:

            new_binding = self.binding[selected_pts_mask]
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding.long(), torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        if self.transforms is not None: # In case of HO
            # print(self.binding == self.identity_binding_index, self.binding.shape)
            # grads[self.binding == self.identity_binding_index] = 0.0
            grads[:] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # if self.limit2 == 0:
        #     prune_mask[:self.limit1] = False
        # else:
        #     prune_mask[:self.limit2] = False

        if self.transforms is not None: # In case of HO
            prune_mask = torch.zeros_like(prune_mask, dtype=torch.bool) # keep all points for HO

        # Keep object points
        if self.binding is not None:
            prune_mask[self.binding == self.identity_binding_index] = False

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_seen_status(self, update_filter):
        self.seen[update_filter] = True

    def clear_seen_status(self):
        self.seen = torch.zeros(self.seen.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)

