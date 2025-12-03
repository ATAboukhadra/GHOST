import torch

def compute_face_transformation_optimized(canonical_verts, deformed_verts, faces):
    """
    Compute per-face transformations (translation, rotation, scale)
    from canonical (flat hand) to deformed space.
    Optimized to remove loops and use tensor operations.
    """
    device = canonical_verts.device

    # Extract vertices for each face (batched operation)
    v0_c, v1_c, v2_c = [canonical_verts[faces[:, i]] for i in range(3)]
    v0_d, v1_d, v2_d = [deformed_verts[faces[:, i]] for i in range(3)]

    # Compute edges for canonical and deformed triangles
    e1_c, e2_c = v1_c - v0_c, v2_c - v0_c
    e1_d, e2_d = v1_d - v0_d, v2_d - v0_d

    # Construct basis matrices for canonical and deformed triangles
    T_c = torch.stack([e1_c, e2_c, torch.cross(e1_c, e2_c, dim=-1)], dim=-1)  # Shape: (num_faces, 3, 3)
    T_d = torch.stack([e1_d, e2_d, torch.cross(e1_d, e2_d, dim=-1)], dim=-1)  # Shape: (num_faces, 3, 3)

    # Compute transformation matrices (M = T_d @ T_c^-1)
    T_c_inv = torch.linalg.inv(T_c)  # Shape: (num_faces, 3, 3)
    M = T_d @ T_c_inv  # Shape: (num_faces, 3, 3)

    # Compute translations
    translation = v0_d - torch.einsum("bij,bj->bi", M, v0_c)  # Shape: (num_faces, 3)

    # Construct full affine transformation matrices (4x4)
    num_faces = faces.shape[0]
    affine_matrices = torch.eye(4, device=device).unsqueeze(0).repeat(num_faces, 1, 1)
    affine_matrices[:, :3, :3] = M
    affine_matrices[:, :3, 3] = translation

    return affine_matrices

def compute_affine_transformation(v0_c, v1_c, v2_c, v0_d, v1_d, v2_d):
    """
    Compute the affine transformation from a canonical triangle to a deformed triangle.
    """
    # Compute edges in canonical and deformed space
    e1_c = v1_c - v0_c
    e2_c = v2_c - v0_c
    e1_d = v1_d - v0_d
    e2_d = v2_d - v0_d
    
    # Construct basis matrices
    T_c = torch.stack([e1_c, e2_c, torch.cross(e1_c, e2_c)], dim=1)  # Canonical basis
    T_d = torch.stack([e1_d, e2_d, torch.cross(e1_d, e2_d)], dim=1)  # Deformed basis
    
    # Compute the transformation matrix
    M = T_d @ torch.linalg.inv(T_c)  # M: affine transformation
    
    return M

def polar_decomposition(M):
    """
    Perform polar decomposition on a matrix M to extract rotation (R) and scale/shear (S).
    """
    U, S, Vt = torch.linalg.svd(M)
    R = U @ Vt
    S_matrix = Vt.T @ torch.diag(S) @ Vt
    return R, S_matrix

def update_gaussians(gaussians, transformations, num_gaussians_per_face):
    """
    Update Gaussian attributes based on the per-face transformations.
    """
    updated_gaussians = []
    for i, gaussian in enumerate(gaussians):
        transformation = transformations[i // num_gaussians_per_face]  # Transformation for the corresponding face
        M, translation = transformation[:3, :3], transformation[:3, 3]
        # Decompose M into rotation (R) and scale-shear (S)
        R, S = polar_decomposition(M)
        
        # Update Gaussian center (apply rotation and translation)
        updated_center = M @ gaussian["center"] + translation
        
        # Update Gaussian normal (rotate normal vector)
        updated_normal = R @ gaussian["normal"]
        # import ipdb;ipdb.set_trace()
        # Update Gaussian rotation matrix
        updated_rotation = R.flatten()
        # print(updated_rotation.shape)
        # updated_rotation = R.flatten()
        
        # Update Gaussian scale (extract diagonal of S for scale)
        # updated_scale = torch.diag(S)

        updated_gaussians.append({
            "center": updated_center,
            "normal": updated_normal,
            "rotation": updated_rotation,
            # "scale": updated_scale,
            "scale": gaussian["scale"],
        })
    return updated_gaussians

def apply_transformations(vertices, transformations, faces):
    """
    Apply per-face transformations to the vertices.
    """
    updated_vertices = vertices.clone()
    vertex_updated = torch.zeros_like(vertices[:, 0]).bool()
    for i, face in enumerate(faces):
        M, translation = transformations[i][:3, :3], transformations[i][:3, 3]
        updated_vertices[face] = torch.bmm(M.unsqueeze(0).repeat_interleave(3, dim=0), vertices[face].unsqueeze(-1)).squeeze(-1) + translation
        vertex_updated[face] = torch.ones_like(face).bool()
    return updated_vertices

def compute_face_transformation_optimized_batched(canonical_verts, deformed_verts, faces):
    """
    Batched version for computing per-face affine transformations
    from canonical_verts (V, 3) to deformed_verts (T, V, 3).
    
    Returns:
        affine_matrices: (T, F, 4, 4)
    """
    device = canonical_verts.device
    T = deformed_verts.shape[0]
    F = faces.shape[0]

    # Canonical triangle edges (same for all frames)
    v0_c, v1_c, v2_c = [canonical_verts[faces[:, i]] for i in range(3)]  # Each: (F, 3)
    e1_c = v1_c - v0_c  # (F, 3)
    e2_c = v2_c - v0_c
    n_c = torch.cross(e1_c, e2_c, dim=-1)  # (F, 3)
    T_c = torch.stack([e1_c, e2_c, n_c], dim=-1)  # (F, 3, 3)
    T_c_inv = torch.linalg.inv(T_c)  # (F, 3, 3)

    # Deformed triangle vertices for each frame
    v0_d, v1_d, v2_d = [deformed_verts[:, faces[:, i], :] for i in range(3)]  # Each: (T, F, 3)
    e1_d = v1_d - v0_d  # (T, F, 3)
    e2_d = v2_d - v0_d
    n_d = torch.cross(e1_d, e2_d, dim=-1)
    T_d = torch.stack([e1_d, e2_d, n_d], dim=-1)  # (T, F, 3, 3)

    # Compute transformation matrices M = T_d @ T_c^-1 (batched matmul)
    M = T_d @ T_c_inv.unsqueeze(0)  # (T, F, 3, 3)

    # Compute translation: t = v0_d - M @ v0_c
    v0_c_exp = v0_c.unsqueeze(0)  # (1, F, 3)
    translation = v0_d - torch.einsum("tfij,fj->tfi", M, v0_c)  # (T, F, 3)

    # Assemble affine matrices (T, F, 4, 4)
    affine_matrices = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(T, F, 1, 1)  # (T, F, 4, 4)
    affine_matrices[:, :, :3, :3] = M
    affine_matrices[:, :, :3, 3] = translation

    return affine_matrices


def compute_face_areas(verts, faces):
    # verts: (V, 3) or (T, V, 3), faces: (F, 3)
    v0, v1, v2 = [verts[..., faces[:, i], :] for i in range(3)]  # shape: (T, F, 3) or (F, 3)
    cross_prod = torch.cross(v1 - v0, v2 - v0, dim=-1)
    area = 0.5 * torch.norm(cross_prod, dim=-1)  # shape: (T, F) or (F,)
    return area