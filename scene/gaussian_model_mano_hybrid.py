# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
import numpy as np
import torch
from pathlib import Path
from plyfile import PlyData
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from simple_knn._C import distCUDA2

from utils.graphics_utils import compute_face_transformation_optimized
from utils.general_utils import matrix_to_quaternion, quaternion_multiply
from .gaussian_model_mano import GaussianModelMano


class GaussianModelManoHybrid(GaussianModelMano):
    """
    Hybrid viewer model that keeps the hand/object binding logic from GaussianModelMano
    while also supporting loading per-frame MANO pose/shape sequences for interactive viewing.
    """

    def __init__(self, sh_degree: int, num_pose_params: int = 45):
        super().__init__(sh_degree, num_pose_params)

        # Simple per-frame parameters used by the viewer controls
        self.pose_param = torch.zeros(1, 48).cuda()
        self.shape_param = torch.zeros(1, 10).cuda()
        self.transl = torch.zeros(1, 3).cuda()

        # Canonical verts used for per-face transforms in the viewer
        self.verts_cano = self.canonical_verts.unsqueeze(0)
        self.timestep = 0
        self.num_timesteps = None

    def update_mesh_by_param_dict(self, pose_param=None, shape_param=None):
        pose = pose_param if pose_param is not None else self.pose_param
        shape = shape_param if shape_param is not None else self.shape_param

        verts, _ = self.mano_layer(pose, shape, self.transl)
        verts = verts / self.scale_factor

        self.update_mesh_properties(verts, self.verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep

        if hasattr(self, "pose_param_seq"):
            pose = self.pose_param_seq[timestep].unsqueeze(0)
            shape = self.shape_param_seq[timestep].unsqueeze(0) if hasattr(self, "shape_param_seq") else self.shape_param
        else:
            pose, shape = self.pose_param, self.shape_param

        verts, _ = self.mano_layer(pose, shape, self.transl)
        verts = verts / self.scale_factor

        self.update_mesh_properties(verts, self.verts_cano)

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.mano_layer.th_faces.to(verts.device)
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        self.face_transform, self.face_orien_mat, self.face_scaling, self.face_center = compute_face_transformation_optimized(
            verts_cano[0], verts[0], faces
        )

        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano

        # drive Gaussian transforms with per-face motions
        self._update_transforms_image()

    def _update_transforms_image(self):
        if self.binding is None or not hasattr(self, "face_transform"):
            return

        face_tf = self.face_transform.to(self.binding.device)
        binding = self.binding.long()
        self.transforms_image = face_tf[binding]
        self.transforms_image_quat = matrix_to_quaternion(self.transforms_image[:, :3, :3])

    def _load_ply_basic(self, path: Path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        if len(extra_f_names) == features_extra.shape[1] * features_extra.shape[2]:
            flat_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                flat_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = flat_extra.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        if len(scale_names) > 0:
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(xyz).float().cuda()), 1e-7)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2).cpu().numpy()

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Allow for rotation matrices stored instead of quaternions
        if rots.shape[1] == 9:
            rots = matrix_to_quaternion(torch.from_numpy(rots.reshape(-1, 3, 3))).numpy()

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        self._features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = (
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._scaling_init = self._scaling.detach().clone()

    def load_ply(self, path, has_target=False, motion_path=None, disable_fid=None):
        path = Path(path)

        # If we have training-time transforms loaded use the original loader (keeps bindings/identity index)
        if hasattr(self, "identity_binding_index"):
            super().load_ply(str(path), hand=False, has_target=has_target)
        else:
            self._load_ply_basic(path)

            # initialize binding for the viewer (no identity transform available)
            num_faces = len(self.mano_layer.th_faces)
            num_gaussians_per_face = max(1, self._xyz.shape[0] // num_faces)
            binding = torch.arange(num_faces).repeat_interleave(num_gaussians_per_face)
            if binding.shape[0] < self._xyz.shape[0]:
                # Pad to cover all points
                pad = torch.full((self._xyz.shape[0] - binding.shape[0],), num_faces - 1, dtype=binding.dtype)
                binding = torch.cat([binding, pad], dim=0)
            self.binding = binding[: self._xyz.shape[0]].cuda()
            self.binding_counter = torch.ones_like(self.binding, dtype=torch.int32).cuda()
            self.identity_binding_index = -1

        if disable_fid:
            mask = (self.binding[:, None] != torch.tensor(disable_fid, device=self.binding.device)[None, :]).all(-1)
            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
            if self.max_radii2D is not None:
                self.max_radii2D = self.max_radii2D[mask]

        if motion_path is not None and not has_target:
            self.load_motion_sequence(motion_path)

    def load_motion_sequence(self, motion_path):
        motion_path = Path(motion_path)
        mano_pose_param = torch.load(str(motion_path)).cuda().float()
        shape_param_path = motion_path.with_name(motion_path.name.replace("pose", "shape"))
        shape_param = torch.load(str(shape_param_path)).cuda().float()

        self.pose_param_seq = mano_pose_param
        self.shape_param_seq = shape_param
        self.num_timesteps = self.pose_param_seq.shape[0]

        # keep the interactive params in sync with the loaded sequence
        self.pose_param = self.pose_param_seq[:1].clone()
        self.shape_param = self.shape_param_seq[:1].clone()

    # Override to always use transforms_image when available (viewer setting)
    def get_gaussians_position(self):
        if hasattr(self, "transforms_image"):
            Ms = self.transforms_image[:, :3, :3]
            translations = self.transforms_image[:, :3, 3:]
            return (Ms @ self._xyz.unsqueeze(-1) + translations).squeeze(-1)
        return super().get_gaussians_position()

    def get_gaussians_rotation(self):
        if hasattr(self, "transforms_image"):
            return quaternion_multiply(self.transforms_image_quat, self._rotation)
        return super().get_gaussians_rotation()
