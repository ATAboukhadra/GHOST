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
# from .graphics_utils import compute_face_transformation_optimized
from ho_preprocessing_utils.transformation_utils import compute_face_transformation_optimized_batched
from pathlib import Path
# import mano
# from manotorch.manolayer import ManoLayer as TorchManoLayer
import mano_torch
import trimesh


def map_deform2hold(verts, scale, _normalize_shift):
    conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normalize_shift = _normalize_shift.copy()
    normalize_shift[0] *= -1
    verts = verts.cpu().numpy()
    verts -= normalize_shift
    verts /= scale
    verts = np.dot(verts, conversion_matrix)
    return torch.tensor(verts, dtype=torch.float32)


np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.str = np.str_
np.unicode = np.unicode_
np.object = np.object_
np.complex = np.complex_

from manopth.manolayer import ManoLayer
torch.autograd.set_detect_anomaly(True)


class GaussianModelMano:

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


    def __init__(self, sh_degree : int, num_pose_params : int = 45):
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

        # MANO Stuff
        self.use_pca = True if num_pose_params < 45 else False
        self.num_pose_params = num_pose_params
        self.mano_layer = ManoLayer(mano_root='ho_preprocessing_utils/mano_v1_2/models', use_pca=self.use_pca, ncomps=num_pose_params, flat_hand_mean=True).cuda()
        self.pose_param_cano = torch.zeros(1, 48).cuda()
        self.shape_param_cano = torch.zeros(1, 10).cuda()
        # self.transl = torch.zeros(1, 3).cuda()
        self.transl_cano = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.pose_param_cano.device).unsqueeze(0)
        self.scale_factor = 1000
        self.num_object_gaussians = 0

        self.canonical_verts, _ = self.mano_layer(self.pose_param_cano, self.shape_param_cano, self.transl_cano)
        self.canonical_verts = self.canonical_verts[0] / self.scale_factor  # Canonical (flat) hand vertices

        self.canonical_verts_left, _ = self.mano_layer(self.pose_param_cano, self.shape_param_cano, self.transl_cano)
        self.canonical_verts_left[:, :, 0] *= -1  # Mirror the canonical hand for left hand
        self.canonical_verts_left = self.canonical_verts_left[0] / self.scale_factor  # Canonical (flat) left hand vertices

        self.faces = self.mano_layer.th_faces  # Faces of the mesh
        self.iou_hand = None
        self._pose_params = None
        self._shape_params = None
        self.optimize_left = False
        self.hold_pc = None

        # self.spawned_points_binding = None

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
        return self.scaling_activation(self._scaling)
        
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

        if self._pose_params is not None:
            print("Optimizing MANO Params ..")
            # l.append({'params': [self._pose_params], 'lr': training_args.pose_lr, "name": "mano_pose"})
            # l.append({'params': [self._shape_params], 'lr': training_args.shape_lr, "name": "mano_shape"})
            l.append({'params': [self._transl], 'lr': training_args.transl_lr, "name": "mano_transl"})

            if self.optimize_left:
                # l.append({'params': [self._pose_params_left], 'lr': training_args.pose_lr, "name": "mano_pose_left"})
                l.append({'params': [self._transl_left], 'lr': training_args.transl_lr, "name": "mano_transl_left"})

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

    def save_ply(self, path, save_cano=False, frame_num=-1):
        # print(os.path.dirname(path))
        mkdir_p(os.path.dirname(path))
        # return
        if frame_num >= 0:
            # just write a txt file with the frame number
            with open(f"{path[:-4]}_{frame_num}.txt", 'w') as f:
                f.write(str(frame_num))

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
        if save_cano:
            cano_attributes = attributes[self.right_hand_indices]
            elements = np.empty(cano_attributes.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, cano_attributes))
            el = PlyElement.describe(elements, 'vertex')   
            PlyData([el]).write(path[:-4]+"_canonical.ply")     
        
        #saving current deformed hand
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # Compute rotated sh

        # attributes = np.concatenate((self.get_gaussians_position().detach().cpu().numpy(), normals, f_dc, 
        #                              self.rotate_sh_batched().detach().cpu().numpy().reshape(len(self._xyz), -1), opacities, scale, 
        #                              self.get_gaussians_rotation().detach().cpu().numpy()), axis=1)
        attributes = np.concatenate((self.get_gaussians_position().detach().cpu().numpy(), normals, f_dc, f_rest, opacities, scale, self.get_gaussians_rotation().detach().cpu().numpy()), axis=1)

        elements[:] = list(map(tuple, attributes))

        if self.binding is None:
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path[:-4]+"_object.ply")
            return

        def save_subset(mask, suffix):
            elements = np.empty(np.sum(mask), dtype=dtype_full)
            elements[:] = list(map(tuple, attributes[mask]))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(f"{path[:-4]}_{suffix}.ply")

        binding_cpu = self.binding.detach().cpu().numpy()
        num_faces_per_hand = 1538  # adjust based on MANO

        is_right = (binding_cpu >= 0) & (binding_cpu < num_faces_per_hand)
        is_left = (binding_cpu >= num_faces_per_hand) & (binding_cpu < 2 * num_faces_per_hand)
        is_object = (binding_cpu == self.identity_binding_index)

        save_subset(is_right, "right_hand")
        save_subset(is_left, "left_hand")
        save_subset(is_object, "object")
        save_subset(is_right | is_left | is_object, "all")

        # Save binding list for reference
        np.save(f"{path[:-4]}_binding.npy", binding_cpu)

        # Save mano parameters if they exist
        if self._pose_params is not None:
            mano_params = {
                "pose": self._pose_params.detach().cpu(),
                "shape": self._shape_params.detach().cpu(),
                "transl": self._transl.detach().cpu(),
                "pose_left": self._pose_params_left.detach().cpu() if self.optimize_left else None,
                "transl_left": self._transl_left.detach().cpu() if self.optimize_left else None,
            }
            for key, value in mano_params.items():
                if value is not None:
                    mano_path = Path(path).parent / f"{Path(path).stem}_{key}.pt"
                    torch.save(value, mano_path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, hand=False):
        plydata = PlyData.read(path)
        self.spatial_lr_scale = 1.0

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
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
            self.num_object_gaussians = xyz.shape[0]
            object_binding = torch.full((self.num_object_gaussians,), self.identity_binding_index, device="cuda", dtype=torch.int32)
            self.binding = torch.cat((self.binding, object_binding), dim=0)
            self.object_gaussians_mask = self.binding == self.identity_binding_index

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

        if hand:
            rots = matrix_to_quaternion(torch.from_numpy(rots.reshape(-1, 3, 3))).numpy()

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
            self.load_mano_params(path, "right")
        else:
            self.transforms = torch.cat((self.transforms[:, :-1], torch.load(path)), dim=1)
            self.load_mano_params(path, "left")

            self.binding = torch.cat((self.binding, torch.arange(self.limit1, self.limit2, device="cuda", dtype=torch.int32) // self.num_gaussians_per_face), dim=0)
            self.binding_counter = torch.cat((self.binding_counter, torch.ones(num_hand_faces, dtype=torch.int32, device="cuda") * int(self.num_gaussians_per_face)), dim=0)

        num_frames = self.transforms.shape[0]
        identity = torch.eye(4, device="cuda").unsqueeze(0).repeat(num_frames, 1, 1).unsqueeze(1)
        self.transforms = torch.cat((self.transforms, identity), dim=1)#[T, F+1, 4, 4]
        self.identity_binding_index = self.transforms.shape[1] - 1  # This is the index of the identity transform

    def set_image_transform(self, image_id, cam):

        # W2C = np.eye(4)
        # W2C[:3, :3] = np.transpose(cam.R)#R is C2W
        # W2C[:3,  3] = cam.T#T is W2C
        # C2W = torch.from_numpy(np.linalg.inv(W2C)).float().to(self.transforms.device)#transforms are float and on cuda

        # #select the transformations corresponding to the current image id and assign to each Gaussian their transformation based on which face it belongs to
        # transforms_image = self.transforms[image_id, self.binding.long()]#[T, F, 4, 4] -> [N, 4, 4], where T=num_images, F=num_faces, N=num_gaussians
        # transforms_image = torch.matmul(C2W, transforms_image)

        # # Replace the object transformation with identity

        # eye_stack = torch.eye(4, device=transforms_image.device).unsqueeze(0).repeat(self.num_object_gaussians, 1, 1)
        # transforms_image = transforms_image.clone()  # Avoid modifying view
        # # with torch.no_grad():
        # #     transforms_image[self.object_gaussians_mask] = eye_stack

        # self.transforms_image = transforms_image
        # self.transforms_image_quat = matrix_to_quaternion(self.transforms_image[:, :3, :3])
        
        # self.is_grasping = False
        # if self.transforms is not None and len(self.object_poses) > 0:
        #     self.is_grasping = self.detect_grasping(image_id, cam)

        W2C = np.eye(4)
        W2C[:3, :3] = np.transpose(cam.R)#R is C2W
        W2C[:3,  3] = cam.T#T is W2C
        C2W = torch.from_numpy(np.linalg.inv(W2C)).float().to(self.transforms.device)#transforms are float and on cuda

        #select the transformations corresponding to the current image id and assign to each Gaussian their transformation based on which face it belongs to
        self.transforms_image = self.transforms[image_id, self.binding.long()]#[T, F, 4, 4] -> [N, 4, 4], where T=num_images, F=num_faces, N=num_gaussians
        self.transforms_image = torch.matmul(C2W, self.transforms_image)
        # Replace the object transformation with identity
        object_gaussians_mask = self.binding == self.identity_binding_index

        with torch.no_grad():
            num_object_gaussians = int(object_gaussians_mask.sum().item())
            self.transforms_image[object_gaussians_mask] = torch.eye(4, device=self.transforms_image.device).unsqueeze(0).repeat(num_object_gaussians, 1, 1)

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
                    return True
                else:
                    return False

                # print(f'difference in transformation {image_id}:', diff_transformation)
                # Check if the difference is significant i.e. if the object is moving
                # compute how close to identity the transformation is

    def get_sh(self):
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
            # self.object_center = torch.mean(xyz[self.binding == self.identity_binding_index], dim=0, keepdim=True)
            # self.hand_center = torch.mean(xyz[self.binding != self.identity_binding_index], dim=0, keepdim=True)

            return xyz
        
        return self._xyz
    
    def get_gaussians_rotation(self):
        if self.transforms is not None:
            quad = quaternion_multiply(self.transforms_image_quat, self._rotation)
            return quad
        return self._rotation

    # def rotate_sh_batched(self):
    #     # Convert quaternions to rotation matrices
    #     R_matrices = np.stack([
    #         R.from_quat(q.detach().cpu().numpy()).as_matrix()
    #         for q in self.transforms_image_quat
    #     ], axis=0)  # (N, 3, 3)

    #     R_matrices = torch.from_numpy(R_matrices).to(self._features_rest.device, dtype=self._features_rest.dtype)

    #     rotated = transform_shs_batched(self._features_rest, R_matrices)
    #     return rotated

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
            if 'mano' in group["name"]:
                # These parameters are not extended, so we skip them
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

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
            if 'mano' in group["name"]:
                # These parameters are not extended, so we skip them
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

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

        # if self.spawned_points_binding is not None:
        #     self.spawned_points_binding = self.spawned_points_binding[valid_points_mask]

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
            if 'mano' in group["name"]:
                # These parameters are not extended, so we skip them
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

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
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }

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
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_gaussians_position()[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if self.binding is not None:

            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding.long(), torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        # if self.spawned_points_binding is not None:
        #     new_binding = torch.zeros_like(new_xyz[:, 0], dtype=torch.int32, device="cuda")
        #     self.spawned_points_binding = torch.cat((self.spawned_points_binding, new_binding))

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
        
        # if self.spawned_points_binding is not None:
        #     new_binding = torch.zeros_like(new_xyz[:, 0], dtype=torch.int32, device="cuda")
        #     self.spawned_points_binding = torch.cat((self.spawned_points_binding, new_binding))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        if self.transforms is not None: # In case of HO
            # print(self.binding == self.identity_binding_index, self.binding.shape)
            grads[self.binding == self.identity_binding_index] = 0.0
            # grads[:] = 0.0

        # Only if number of gaussians is less than 100k
        if self.get_xyz.shape[0] < 100000:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if self.transforms is not None: # In case of HO
            prune_mask = torch.zeros_like(prune_mask, dtype=torch.bool) # keep all points for HO

        # Keep object points
        if self.binding is not None:
            prune_mask[self.binding == self.identity_binding_index] = False
        
        if self.get_xyz.shape[0] > 1000:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_seen_status(self, update_filter):
        self.seen[update_filter] = True

    def clear_seen_status(self):
        self.seen = torch.zeros(self.seen.shape[0], device="cuda", dtype=torch.bool, requires_grad=False)

    # MANO Stuff
    def load_mano_params(self, path, side="right"):
        # replace {side}_transformations.pth with pose_params_{side}.pt
        pose_path = path.replace(f"{side}_transformations.pth", f"pose_params_{side}_final.pt")
        mano_pose_param = torch.load(str(pose_path)).float().cuda()#.requires_grad_()
        mano_pose_param = mano_pose_param.clone().detach().requires_grad_()#.cuda()
        num_frames = mano_pose_param.shape[0]

        shape_path = path.replace(f"{side}_transformations.pth", f"shape_params_right_final.pt") # The same shape params for both hands
        shape_param = torch.load(str(shape_path)).float().cuda()#.requires_grad_()
        shape_param_init = shape_param.clone().detach()
        shape_param = shape_param.clone().detach().requires_grad_()#.cuda()

        transl_path = path.replace(f"transformations", f"translations")
        transl = torch.load(str(transl_path)).reshape(-1, 3).float().cuda()#.requires_grad_()
        transl = transl.clone().detach().requires_grad_()#.cuda()
        
        if side == "right":
            self._pose_params = nn.Parameter(mano_pose_param)
            self._transl = nn.Parameter(transl)
            # Reinitialize canonical verts with the loaded shape params
            self.canonical_verts, _ = self.mano_layer(self.pose_param_cano, shape_param_init, self.transl_cano) 
            self.canonical_verts = self.canonical_verts[0] / self.scale_factor  # Canonical (flat) hand vertices

        else:
            self._pose_params_left = nn.Parameter(mano_pose_param)
            self._transl_left = nn.Parameter(transl)
            self.canonical_verts_left, _ = self.mano_layer(self.pose_param_cano, shape_param_init, self.transl_cano)
            self.canonical_verts_left[:, :, 0] *= -1  # Mirror the canonical hand for left hand
            self.canonical_verts_left = self.canonical_verts_left[0] / self.scale_factor  # Canonical (flat) left hand vertices
            self.optimize_left = True
        
        # Use the same shape params for both hands
        self._shape_params = nn.Parameter(shape_param)

    def update_hand_transformations(self):

        num_frames = self.transforms.shape[0]
        deformed_verts, _ = self.mano_layer(self._pose_params, self._shape_params.repeat(num_frames, 1), self._transl)
        deformed_verts = deformed_verts / self.scale_factor  # Canonical (flat) hand vertices

        transformations = compute_face_transformation_optimized_batched(self.canonical_verts, deformed_verts, self.faces)

        if self.optimize_left:
            deformed_verts_left, _ = self.mano_layer(self._pose_params_left, self._shape_params.repeat(num_frames, 1), self._transl_left)
            deformed_verts_left[:, :, 0] = -deformed_verts_left[:, :, 0]  # Mirror the canonical hand for left hand
            deformed_verts_left = deformed_verts_left / self.scale_factor  # Canonical (flat) left hand vertices
            transformations_left = compute_face_transformation_optimized_batched(self.canonical_verts_left, deformed_verts_left, self.faces)
            transformations = torch.cat((transformations, transformations_left), dim=1)  # [T, F*2, 4, 4]

        identity = torch.eye(4, device="cuda").unsqueeze(0).repeat(num_frames, 1, 1).unsqueeze(1)

        # compare this with old transforms
        self.transforms = torch.cat((transformations, identity), dim=1) # [T, F*2+1, 4, 4]

    @torch.no_grad()
    def export_submission_outputs(self, cam, seq_name):

        SEQUENCE_TO_SHIFT_SCALE = { # HOLD related values required for submission
            "arctic_s03_box_grab_01_1": (np.array([-0.07795634, 0.22807714, 2.49349689]), 1.2525596618652344),
            "arctic_s03_notebook_grab_01_1": (np.array([-0.03239906, 0.15758559, 1.84788918]), 0.9274397492408752),
            "arctic_s03_laptop_grab_01_1": (np.array([-0.04400319, 0.20229244, 0.30538869]), 0.18447273969650269),
            "arctic_s03_ketchup_grab_01_1": (np.array([-0.09286293, 0.11386055, 0.58659148]), 0.3023563027381897),
            "arctic_s03_espressomachine_grab_01_1": (np.array([-0.13647562, 0.1287374, 3.27741623]), 1.6413909196853638),
            "arctic_s03_microwave_grab_01_1": (np.array([0.02537658, 0.15653569, 2.42339349]), 1.2142882347106934),
            "arctic_s03_waffleiron_grab_01_1": (np.array([-0.08380565, 0.11460939, 2.25544786]), 1.1299561262130737),
            "arctic_s03_mixer_grab_01_1": (np.array([-0.06304927, 0.13679582, 3.52100658]), 1.7621134519577026),
            "arctic_s03_capsulemachine_grab_01_1": (np.array([-0.11495478, 0.15422642, 1.283813]), 0.6490716934204102),
        }

        if seq_name not in SEQUENCE_TO_SHIFT_SCALE:
            # print(f"Sequence {seq_name} not found in SEQUENCE_TO_SHIFT_SCALE mapping.")
            scale = None
            # raise ValueError(f"Sequence {seq_name} not found in SEQUENCE_TO_SHIFT_SCALE mapping.")
        else:
            normalize_shift, inverse_scale = SEQUENCE_TO_SHIFT_SCALE[seq_name]
            scale = 1.0 / inverse_scale


        image_path = f"./data/{seq_name}/build/image/{cam.image_name}.png"
        frame_num = int(cam.image_name)

        self.set_image_transform(frame_num, cam)

        # Right hand (already in camera space)
        verts_right, joints_right = self.mano_layer(self._pose_params[frame_num:frame_num+1],
                                                    self._shape_params.repeat(1, 1),
                                                    self._transl[frame_num:frame_num+1])
        verts_right = verts_right[0] / self.scale_factor
        joints_right = joints_right[0] / self.scale_factor
        joints_right_centered = joints_right - joints_right[0:1]
        # verts_right_centered = verts_right - joints_right[0:1]
        if scale is not None:
            verts_right_hold = map_deform2hold(verts_right, scale, normalize_shift)

        # Left hand (already in camera space)
        if self.optimize_left:
            verts_left, joints_left = self.mano_layer(self._pose_params_left[frame_num:frame_num+1],
                                                    self._shape_params.repeat(1, 1),
                                                    self._transl_left[frame_num:frame_num+1])
            verts_left[:, :, 0] *= -1
            joints_left[:, :, 0] *= -1 

            verts_left = verts_left[0] / self.scale_factor
            joints_left = joints_left[0] / self.scale_factor
            joints_left_centered = joints_left - joints_left[0:1]
            # verts_left_centered = verts_left - joints_left[0:1]
        else:
            verts_left = torch.zeros((778, 3), device=verts_right.device)
            joints_left_centered = torch.zeros((21, 3), device=verts_right.device)
            # verts_left_centered = torch.zeros((778, 3), device=verts_right.device)

        # Convert object Gaussians from world to camera space
        obj_mask = self.binding == self.identity_binding_index
        # Number of object Gaussians before filtering
        num_object_gaussians = obj_mask.sum().item()
        
        opacity = self.get_opacity.squeeze()  # shape: [N]
        # obj_mask &= opacity > 0.05  # or 0.1 depending on tuning

        obj_world = self.get_gaussians_position()[obj_mask]
        # obj_world = self.get_gaussians_position()

        W2C = torch.eye(4, device="cuda")
        W2C[:3, :3] = torch.from_numpy(cam.R.T).float().to(W2C.device)
        W2C[:3, 3] = torch.from_numpy(cam.T).float().to(W2C.device)

        ones = torch.ones((obj_world.shape[0], 1), device=obj_world.device)
        obj_world_homo = torch.cat([obj_world, ones], dim=-1)  # (N, 4)
        obj_cam = (W2C @ obj_world_homo.T).T[:, :3]  # (N, 3)

        if scale is not None:
            obj_cam_hold = map_deform2hold(obj_cam, scale, normalize_shift)

        # Relative to hand roots
        v3d_right_object = obj_cam - joints_right[0:1]
        v3d_left_object = obj_cam - (joints_left[0:1] if self.optimize_left else joints_right[0:1])

        # Extra required outputs
        faces_np_left = self.mano_layer.th_faces[:, [0, 2, 1]].cpu().numpy()  # Convert to numpy for trimesh
        faces_np_right = self.mano_layer.th_faces.cpu().numpy()
        canonical_verts_left, _ = self.mano_layer(self.pose_param_cano, self._shape_params.repeat(1, 1), self.transl_cano)
        canonical_verts_left[:, :, 0] *= -1  # Mirror the canonical hand for left hand
        canonical_verts_left = canonical_verts_left[0] / self.scale_factor  # Canonical (flat) left hand vertices
        verts_np = canonical_verts_left.cpu().numpy()
        
        root_object = obj_cam.mean(dim=0)
        obj_cam_centered = obj_cam - root_object

        # Export verts_right, verts_left and obj_cam as point clouds
        import open3d as o3d
        def save_ply(name, points):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            o3d.io.write_point_cloud(name, pcd)

        if frame_num == 0:
            out_dir = f"./pointclouds/{seq_name}"
            os.makedirs(out_dir, exist_ok=True)

            save_ply(os.path.join(out_dir, "right_hand.ply"), verts_right)
            save_ply(os.path.join(out_dir, "left_hand.ply"), verts_left)
            save_ply(os.path.join(out_dir, "object.ply"), obj_cam)

            # Save left canonical as mesh
            save_path = os.path.join(out_dir, "left_hand_canonical.obj")
            # Create mesh
            mesh_left = trimesh.Trimesh(vertices=verts_np, faces=faces_np_left, process=False)
            # Export to OBJ
            mesh_left.export(save_path, file_type='obj')
            # Save right world vertices as mesh
            # save_path = os.path.join(out_dir, "right_hand_hold.obj")
            # # Create mesh
            # mesh_right = trimesh.Trimesh(vertices=verts_right_hold.cpu().numpy(), faces=faces_np_right, process=False)
            # # Export to OBJ
            # mesh_right.export(save_path, file_type='obj')

            print(f"[EXPORT] Saved PLY point clouds to {out_dir}")
            print(f"Number of object Gaussians before filtering: {num_object_gaussians}")
            print(f"Number of object Gaussians after filtering: {obj_mask.sum().item()}")


        final_dict = {
            'image_path': image_path,
            'v_posed.left': canonical_verts_left.unsqueeze(0),  # Posed left hand vertices

            'v3d_c.right': verts_right.unsqueeze(0),
            'v3d_c.object': obj_cam.unsqueeze(0),
            'j3d_c.right': joints_right.unsqueeze(0),

            'root.right': joints_right[0:1],  # Root joint for right hand
            'j3d_ra.right': joints_right_centered.unsqueeze(0),
            'root.object': root_object.unsqueeze(0),  # Root joint for the object
            'v3d_ra.object': obj_cam_centered.unsqueeze(0),  # Relative to root joint of the object
            
            'v3d_right.object': v3d_right_object.unsqueeze(0),
            
            'faces': {
                'left': torch.tensor(faces_np_left, dtype=torch.int16),  # Left hand faces
                'right': torch.tensor(faces_np_right, dtype=torch.int16),  # Right hand faces
                'object': torch.tensor([[0, 1, 2]], dtype=torch.int16)  # No object faces, but can be added if needed
            }    
        }

        if self.optimize_left:
            final_dict['v3d_c.left'] = verts_left.unsqueeze(0)  # Canonical left hand vertices
            final_dict['j3d_c.left'] = joints_left.unsqueeze(0)  # Canonical left hand joints
            final_dict['root.left'] = joints_left[0:1]  # Root joint for left hand
            final_dict['j3d_ra.left'] = joints_left_centered.unsqueeze(0)
            final_dict['v3d_left.object'] = v3d_left_object.unsqueeze(0)

        if scale is not None:
            final_dict['verts.right'] = verts_right_hold.unsqueeze(0)  # Right hand vertices in hold space
            final_dict['verts.object'] = obj_cam_hold.unsqueeze(0)  # Object vertices in hold space

        return final_dict

    def load_prior(self, path):
        # Load and .obj prior point cloud
        mesh = trimesh.load(path, force='mesh')
        if mesh.vertices.shape[0] < 5000:
            pts_np, face_idx = trimesh.sample.sample_surface(mesh, 5000)
            trimesh_points = trimesh.PointCloud(pts_np)
            trimesh_points.export(path.replace('.obj', '_sampled.ply'), file_type='ply')
            verts = torch.tensor(pts_np, dtype=torch.float32, device="cuda")
        else:
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")

        # Export the sampled points to a .ply file
        self.obj_prior_pc = verts
        # self.spawned_points_binding = torch.zeros((self._xyz.shape[0]), dtype=torch.int32, device="cuda")  # Initialize binding for prior points
        
        seq_name = path.split('/')[-4]
        if os.path.exists(f'hold_stuff/hold_{seq_name}_meshes/v3d_ra_object_0000.obj'):
            self.hold_pc = trimesh.load(f'hold_stuff/hold_{seq_name}_meshes/v3d_ra_object_0000.obj', force='mesh').vertices
            self.hold_pc = torch.tensor(self.hold_pc, dtype=torch.float32, device="cuda")

    @torch.no_grad()
    def spawn_at_positions(
        self,
        new_centers_xyz: torch.Tensor,   # (K,3) positions in the SAME coord frame as get_gaussians_position()
        init_scale_frac: float = 0.6,    # shrink copied scales a bit
        init_alpha_frac: float = 0.5     # slightly lower starting opacity
    ):
        """
        Spawn new gaussians at given positions by copying internal params from
        the nearest existing *object* gaussian (binding==identity_binding_index).
        Falls back to global nearest if object set is empty.
        """

        device = self._xyz.device
        K = new_centers_xyz.shape[0]
        if K == 0:
            return

        # --- choose donor set (object gaussians preferred)
        if getattr(self, "binding", None) is not None:
            obj_mask = (self.binding == self.identity_binding_index)
            donor_idx = torch.nonzero(obj_mask, as_tuple=False).squeeze(1)
        else:
            donor_idx = torch.arange(self.get_xyz.shape[0], device=device)

        if donor_idx.numel() == 0:
            donor_idx = torch.arange(self.get_xyz.shape[0], device=device)
            if donor_idx.numel() == 0:
                return  # nothing to copy from

        # --- nearest neighbor donors
        donors_pos = self.get_gaussians_position()[donor_idx]           # (Nd,3)
        d = torch.cdist(new_centers_xyz.to(device), donors_pos)         # (K,Nd)
        nn = donor_idx[d.argmin(dim=1)]                                 # (K,)

        # --- build new tensors by copying internal (parameter-space) values
        new_xyz          = new_centers_xyz.to(device)                    # same space as get_gaussians_position()
        new_features_dc  = self._features_dc[nn].clone()
        new_features_rest= self._features_rest[nn].clone()
        new_opacities    = self._opacity[nn].clone() * init_alpha_frac   # param-space opacity copied & reduced
        # new_opacities[:] = 0.0
        new_scaling      = self._scaling[nn].clone()
        new_rotation     = self._rotation[nn].clone()

        # shrink scales a bit (in param space if you prefer, but safe to do in activated space then inverse)
        # simplest: scale param directly (empirically fine)
        new_scaling = new_scaling * init_scale_frac

        # new_binding = torch.ones((K,), device=device, dtype=self.spawned_points_binding.dtype)
        # self.spawned_points_binding = torch.cat((self.spawned_points_binding, new_binding), dim=0)

        # --- commit
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
        )

    # @torch.no_grad()
    # def spawn_at_positions(
    #     self,
    #     new_centers_xyz: torch.Tensor,   # (K,3) positions in same coord frame as get_gaussians_position()
    #     init_scale_frac: float = 0.6,    # shrink copied scales a bit
    #     init_alpha_frac: float = 0.5,    # lower starting opacity
    #     k_neighbors: int = 5             # number of donors to blend
    # ):
    #     """
    #     Spawn new gaussians at given positions by copying/averaging parameters
    #     from nearest existing object gaussians. Falls back to global set if empty.
    #     Colors are averaged, scales set to median, rotations aligned to local normals.
    #     """
    #     device = self._xyz.device
    #     K = new_centers_xyz.shape[0]
    #     if K == 0:
    #         return

    #     # --- donor set (prefer object binding)
    #     if getattr(self, "binding", None) is not None:
    #         obj_mask = (self.binding == self.identity_binding_index)
    #         donor_idx = torch.nonzero(obj_mask, as_tuple=False).squeeze(1)
    #     else:
    #         donor_idx = torch.arange(self.get_xyz.shape[0], device=device)

    #     if donor_idx.numel() == 0:
    #         donor_idx = torch.arange(self.get_xyz.shape[0], device=device)
    #         if donor_idx.numel() == 0:
    #             return  # nothing to copy from

    #     # --- nearest neighbors
    #     donors_pos = self.get_gaussians_position()[donor_idx]     # (Nd,3)
    #     dists = torch.cdist(new_centers_xyz.to(device), donors_pos)  # (K,Nd)

    #     knn_dists, knn_idx = torch.topk(dists, k=min(k_neighbors, donor_idx.numel()), dim=1, largest=False)
    #     nn = donor_idx[knn_idx]  # (K,k)

    #     # --- features: distance-weighted average
    #     weights = 1.0 / (knn_dists + 1e-6)
    #     weights = weights / weights.sum(dim=1, keepdim=True)

    #     new_features_dc   = (self._features_dc[nn]   * weights[..., None]).sum(dim=1)
    #     new_features_rest = (self._features_rest[nn] * weights[..., None]).sum(dim=1)
    #     new_opacities     = (self._opacity[nn]       * weights).sum(dim=1) * init_alpha_frac

    #     # --- scales: median of neighbors
    #     new_scaling = self._scaling[nn].median(dim=1).values * init_scale_frac

    #     # --- rotations: align to local normal (PCA of neighbors)
    #     new_rotation = []
    #     for k in range(K):
    #         neigh = donors_pos[knn_idx[k]]  # (k,3)
    #         cov = torch.matmul((neigh - neigh.mean(0)).T, (neigh - neigh.mean(0)))  # (3,3)
    #         eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order
    #         normal = eigvecs[:, 0]  # smallest eigenvector
    #         # create rotation aligning z-axis to normal
    #         z_axis = torch.tensor([0, 0, 1.0], device=device)
    #         v = torch.cross(z_axis, normal)
    #         c = torch.dot(z_axis, normal)
    #         if torch.norm(v) < 1e-6:
    #             R = torch.eye(3, device=device)
    #         else:
    #             vx = torch.tensor([[0, -v[2], v[1]],
    #                             [v[2], 0, -v[0]],
    #                             [-v[1], v[0], 0]], device=device)
    #             R = torch.eye(3, device=device) + vx + vx @ vx * (1 / (1 + c + 1e-6))
    #         new_rotation.append(R.unsqueeze(0))
    #     new_rotation = torch.cat(new_rotation, dim=0)

    #     # --- commit new gaussians
    #     if getattr(self, "binding", None) is not None:
    #         new_binding = torch.full((K,), int(self.identity_binding_index),
    #                                 device=device, dtype=self.binding.dtype)
    #         self.binding = torch.cat([self.binding, new_binding], dim=0)
    #         self.binding_counter.scatter_add_(0, new_binding.long(),
    #                                         torch.ones_like(new_binding, dtype=torch.int32, device=device))

    #     self.densification_postfix(
    #         new_centers_xyz.to(device),
    #         new_features_dc,
    #         new_features_rest,
    #         new_opacities,
    #         new_scaling,
    #         new_rotation
    #     )
