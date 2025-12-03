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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def polar_decomposition(M):
    """
    Perform polar decomposition on a matrix M to extract rotation (R) and scale/shear (S).
    """
    U, S, Vt = torch.linalg.svd(M)
    R = U @ Vt
    S_matrix = Vt.T @ torch.diag(S) @ Vt
    return R, S_matrix

def batch_polar_decomposition(M):
    """
    Perform polar decomposition on a batch of 3x3 matrices M.
    Input: M of shape (b, 3, 3)
    Output: R of shape (b, 3, 3) - rotation matrices
            S of shape (b, 3) - scale/shear values
    """
    U, S, Vt = torch.linalg.svd(M)  # U: (b, 3, 3), S: (b, 3), Vt: (b, 3, 3)
    # print(S.shape, S[0])
    R = U @ Vt  # Compute batch rotation matrices
    # S_matrix = Vt.transpose(-2, -1) @ torch.diag_embed(S) @ Vt  # Compute batch scale/shear matrices
    return R, S

def compute_face_transformation_optimized(canonical_verts, deformed_verts, faces):
    """
    Compute per-face transformations (translation, rotation, scale)
    from canonical (flat hand) to deformed space.
    Optimized to remove loops and use tensor operations.
    """
    device = canonical_verts.device
    # print(canonical_verts.shape, deformed_verts.shape, faces.shape)

    # Extract vertices for each face (batched operation)
    v0_c, v1_c, v2_c = [canonical_verts[faces[:, i]] for i in range(3)]
    v0_d, v1_d, v2_d = [deformed_verts[faces[:, i]] for i in range(3)]

    # Compute edges for canonical and deformed triangles
    # e1_c, e2_c = safe_normalize(v1_c - v0_c), safe_normalize(v2_c - v0_c)
    # e1_d, e2_d = safe_normalize(v1_d - v0_d), safe_normalize(v2_d - v0_d)

    e1_c, e2_c = v1_c - v0_c, v2_c - v0_c
    e1_d, e2_d = v1_d - v0_d, v2_d - v0_d

    # Construct basis matrices for canonical and deformed triangles
    # T_c = torch.stack([e1_c, e2_c, safe_normalize(torch.cross(e1_c, e2_c, dim=-1))], dim=-1)  # Shape: (num_faces, 3, 3)
    # T_d = torch.stack([e1_d, e2_d, safe_normalize(torch.cross(e1_d, e2_d, dim=-1))], dim=-1)  # Shape: (num_faces, 3, 3)
    T_c = torch.stack([e1_c, e2_c, torch.cross(e1_c, e2_c, dim=-1)], dim=-1)  # Shape: (num_faces, 3, 3)
    T_d = torch.stack([e1_d, e2_d, torch.cross(e1_d, e2_d, dim=-1)], dim=-1)  # Shape: (num_faces, 3, 3)
    

    # Compute transformation matrices (M = T_d @ T_c^-1)
    T_c_inv = torch.linalg.inv(T_c)  # Shape: (num_faces, 3, 3)
    M = T_d @ T_c_inv  # Shape: (num_faces, 3, 3)

    R, S = batch_polar_decomposition(M)

    # Compute translations
    translation = v0_d - torch.einsum("bij,bj->bi", M, v0_c)  # Shape: (num_faces, 3)

    # Construct full affine transformation matrices (4x4)
    num_faces = faces.shape[0]
    affine_matrices = torch.eye(4, device=device).unsqueeze(0).repeat(num_faces, 1, 1)
    affine_matrices[:, :3, :3] = M
    affine_matrices[:, :3, 3] = translation

    # print(affine_matrices.min(), affine_matrices.max())

    # compute average scale
    scale = S.mean(dim=-1).unsqueeze(-1)
    # print(scale.min(), scale.max())

    # scale = torch.linalg.det(R).abs().pow(1/3).unsqueeze(-1)
    # scale = 
    # print(S.shape, S[0])

    # print(scale.shape, R.shape, translation.shape)

    # orientation = affine_matrices[:, :3, :3]
    # scale = torch.linalg.det(orientation).abs().pow(1/3)

    return affine_matrices, R, scale, translation

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)
    
    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2

    return orientation, scale

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals
