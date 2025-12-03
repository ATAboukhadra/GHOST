import numpy as np
import torch
import os
import trimesh
from plyfile import PlyData, PlyElement

def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialize_gaussians_evenly(verts, faces, num_gaussians_per_face=5, rescale_factor=1000):
    device = verts.device
    gaussians = []

    for face in faces:
        face_verts = verts[face]  # Vertices of the face
        v0, v1, v2 = face_verts   # Triangle vertices

        # Compute face normal and area
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = torch.cross(edge1, edge2)
        area = 0.5 * torch.norm(normal)  # Area of the triangle
        normal = normal / normal.norm()  # Normalize the normal

        # Compute Gaussian size based on the area and number of Gaussians
        num_actual_gaussians_per_face = (num_gaussians_per_face * (num_gaussians_per_face - 1)) // 2 + num_gaussians_per_face

        gaussian_scale = (area / num_actual_gaussians_per_face)**0.5 / rescale_factor
        # gaussian_scale_vector = torch.tensor([gaussian_scale, gaussian_scale], device=device)
        gaussian_scale_vector = torch.log(torch.tensor([gaussian_scale, gaussian_scale, gaussian_scale], device=device))

        # Create barycentric grid for even distribution
        for i in range(num_gaussians_per_face):
            for j in range(num_gaussians_per_face - i):
                u = (i + 0.5) / num_gaussians_per_face
                v = (j + 0.5) / num_gaussians_per_face
                w = 1 - u - v
                gaussian_center = u * v0 + v * v1 + w * v2

                # Align Gaussian rotation to face normal
                z_axis = normal
                x_axis = edge1 / edge1.norm()  # Use edge1 as x-axis
                y_axis = torch.cross(z_axis, x_axis)
                y_axis = y_axis / y_axis.norm()
                rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1).flatten()
                # print(rotation_matrix)

                gaussians.append({
                    "center": gaussian_center,
                    "normal": normal,
                    "scale": gaussian_scale_vector,
                    "rotation": rotation_matrix,
                })

    return gaussians

def save_gaussians_as_ply(gaussians, features_dc, features_rest, opacities, path):
    mkdir_p(os.path.dirname(path))
    
    xyz = torch.stack([g["center"] for g in gaussians]).cpu().detach().numpy()
    normals = torch.stack([g["normal"] for g in gaussians]).cpu().detach().numpy()
    scales = torch.stack([g["scale"] for g in gaussians]).cpu().detach().numpy()
    rotations = torch.stack([g["rotation"] for g in gaussians]).cpu().detach().numpy()
    features_dc = features_dc.detach().cpu().numpy()
    features_rest = features_rest.detach().cpu().numpy()
    opacities = opacities.detach().cpu().numpy().flatten()
    # scales = scales.detach().cpu().numpy()
    
    attributes = np.concatenate([xyz, normals, features_dc, features_rest, 
                                 opacities[:, None], scales, rotations], axis=1)
    
    attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attribute_names += [f'f_dc_{i}' for i in range(features_dc.shape[1])]
    attribute_names += [f'f_rest_{i}' for i in range(features_rest.shape[1])]
    attribute_names += ['opacity', 'scale_0', 'scale_1', 'scale_2']
    # attribute_names += ['opacity', 'scale_0', 'scale_1']

    attribute_names += [f'rot_{i}' for i in range(9)]
    
    dtype_full = [(name, 'f4') for name in attribute_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    # print(f"PLY file saved as {path}")

def simulate_hand_texture(num_gaussians):
    """
    Simulate realistic skin-tone textures for the Gaussians.
    """
    # Skin tone ranges in RGB (normalized to [0, 1])
    # base_skin_tone = torch.tensor([0.8, 0.6, 0.5])  # Example light skin tone
    # tone_variation = torch.rand(num_gaussians, 3) * 0.05  # Small variations for realism

    # Dark skin tone ranges in RGB (normalized to [0, 1])
    base_skin_tone = torch.tensor([0.4, 0.2, 0.1])  # Example dark skin tone
    tone_variation = torch.rand(num_gaussians, 3) * 0.05  # Small variations for realism


    # Features for diffuse color (skin tone)
    features_dc = base_skin_tone + tone_variation
    features_dc = features_dc.clamp(0, 1)  # Ensure values are within [0, 1]

    # Features for fine details (specular, wrinkles, etc.)
    # features_rest = torch.rand(num_gaussians, 3) * 0.2  # Add random highlights
    features_rest = torch.rand(num_gaussians, 3) * 0.1  # Add subtle highlights

    return features_dc, features_rest


def load_mesh_from_obj(file_path, device='cpu'):
    """
    Load vertices and faces from an .obj file.
    """
    mesh = trimesh.load(file_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return verts, faces
