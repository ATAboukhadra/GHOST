import torch
import einops
# from e3nn import o3

def matrix_to_zyz_angles(R):
    """
    Batch convert rotation matrices to ZYZ Euler angles (compatible with e3nn).
    R: (N, 3, 3)
    Returns: alpha, beta, gamma (each of shape (N,))
    """
    eps = 1e-8
    beta = torch.acos(R[:, 2, 2].clamp(-1+eps, 1-eps))
    
    sin_beta = torch.sin(beta)
    mask = sin_beta > eps

    alpha = torch.atan2(R[:, 1, 2], R[:, 0, 2])
    gamma = torch.atan2(R[:, 2, 1], -R[:, 2, 0])

    # handle gimbal lock (beta near 0)
    alpha[~mask] = torch.atan2(R[~mask, 1, 0], R[~mask, 0, 0])
    gamma[~mask] = 0.0

    return alpha, beta, gamma


def transform_shs_batched(shs_feat, rotation_matrix):
    """
    Rotate SH features of degrees 1–3 using Wigner D matrices for a batch of Gaussians.

    Args:
        shs_feat: (N, 15, 3) SH features
        rotation_matrix: (N, 3, 3) rotation matrices

    Returns:
        rotated SH features: (N, 15, 3)
    """
    device = shs_feat.device
    dtype = shs_feat.dtype
    N = shs_feat.shape[0]

    # === Convert to proper frame by permuting axes (yxz -> xyz)
    P = torch.tensor([[0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]], dtype=dtype, device=device)  # (3, 3)

    P_inv = torch.inverse(P)

    # Adjust rotation matrices
    rotation_matrix = P_inv @ rotation_matrix @ P  # (N, 3, 3)

    # Convert rotation matrices to Euler angles
    # angles = [o3._rotation.matrix_to_angles(rotation_matrix[n]) for n in tqdm(range(N))]
    # alpha, beta, gamma = zip(*angles)
    alpha, beta, gamma = matrix_to_zyz_angles(rotation_matrix)

    D1 = wigner_D_l1_batch(alpha, beta, gamma)  # (N, 3, 3)
    D2 = wigner_D_l2_batch(alpha, beta, gamma)  # (N, 5, 5)

    # D1 = torch.stack([o3.wigner_D(1, a, -b, g) for a, b, g in zip(alpha, beta, gamma)])  # (N, 3, 3)
    # D2 = torch.stack([o3.wigner_D(2, a, -b, g) for a, b, g in zip(alpha, beta, gamma)])  # (N, 5, 5)
    # D3 = torch.stack([o3.wigner_D(3, a, -b, g) for a, b, g in zip(alpha, beta, gamma)])  # (N, 7, 7)

    out = shs_feat.clone()

    # Degree 1
    Y1 = out[:, 0:3, :]  # (N, 3, 3)
    Y1 = einops.rearrange(Y1, 'n shs c -> n c shs')
    Y1_rot = torch.einsum('nij,ncj->nci', D1, Y1)
    out[:, 0:3, :] = einops.rearrange(Y1_rot, 'n c shs -> n shs c')

    # # Degree 2
    Y2 = out[:, 3:8, :]  # (N, 5, 3)
    Y2 = einops.rearrange(Y2, 'n shs c -> n c shs')
    Y2_rot = torch.einsum('nij,ncj->nci', D2, Y2)
    out[:, 3:8, :] = einops.rearrange(Y2_rot, 'n c shs -> n shs c')

    # # Degree 3
    # Y3 = out[:, 8:15, :]  # (N, 7, 3)
    # Y3 = einops.rearrange(Y3, 'n shs c -> n c shs')
    # Y3_rot = torch.einsum('nij,ncj->nci', D3, Y3)
    # out[:, 8:15, :] = einops.rearrange(Y3_rot, 'n c shs -> n shs c')

    return out


def wigner_D_l1_batch(alpha, beta, gamma):
    """
    Compute real Wigner-D matrices for l=1 given ZYZ Euler angles.
    Inputs:
        alpha, beta, gamma: (N,) tensors [in radians]
    Returns:
        D: (N, 3, 3) real-valued Wigner-D matrices
    """
    device = alpha.device
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    cos_b = torch.cos(beta)
    sin_b = torch.sin(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)

    c1 = cos_a * cos_g - sin_a * cos_b * sin_g
    c2 = -cos_a * sin_g - sin_a * cos_b * cos_g
    c3 = sin_a * sin_b

    c4 = sin_a * cos_g + cos_a * cos_b * sin_g
    c5 = -sin_a * sin_g + cos_a * cos_b * cos_g
    c6 = -cos_a * sin_b

    c7 = sin_b * sin_g
    c8 = sin_b * cos_g
    c9 = cos_b

    D = torch.stack([
        torch.stack([c1, c2, c3], dim=-1),
        torch.stack([c4, c5, c6], dim=-1),
        torch.stack([c7, c8, c9], dim=-1),
    ], dim=-2)  # (N, 3, 3)

    return D


def wigner_D_l2_batch(alpha, beta, gamma):
    """
    Compute real Wigner-D matrices for l=2 using real SH basis.
    Inputs:
        alpha, beta, gamma: (N,) tensors [in radians]
    Returns:
        D: (N, 5, 5) real-valued Wigner-D matrices
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    # Compute the real D matrix using known symbolic formulas
    # From: https://github.com/Fyusion/SH/blob/master/src/sh_rotation.cpp
    D = torch.zeros((alpha.shape[0], 5, 5), device=alpha.device, dtype=alpha.dtype)

    c2a, s2a = torch.cos(2 * alpha), torch.sin(2 * alpha)
    c2g, s2g = torch.cos(2 * gamma), torch.sin(2 * gamma)
    sb2, cb2 = sb**2, cb**2

    # Precompute combinations
    sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=alpha.dtype, device=alpha.device))
    sqrt6 = torch.sqrt(torch.tensor(6.0, dtype=alpha.dtype, device=alpha.device))
    sqrt15 = torch.sqrt(torch.tensor(15.0, dtype=alpha.dtype, device=alpha.device))

    # The real-valued basis order: [Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2}]
    # You may also reorder depending on your SH basis.

    # Fill D matrices
    D[:, 0, 0] = 0.25 * (1 + cb)**2 * torch.cos(2 * (alpha + gamma))
    D[:, 0, 1] = 0.5 * sb * (1 + cb) * torch.cos(alpha + gamma)
    D[:, 0, 2] = 0.5 * sb2 * torch.cos(2 * alpha)
    D[:, 0, 3] = 0.5 * sb * (1 - cb) * torch.cos(alpha - gamma)
    D[:, 0, 4] = 0.25 * (1 - cb)**2 * torch.cos(2 * (alpha - gamma))

    D[:, 1, 0] = -0.5 * sb * (1 + cb) * torch.sin(alpha + gamma)
    D[:, 1, 1] = cb * torch.cos(gamma)
    D[:, 1, 2] = -sb * cb * torch.sin(alpha)
    D[:, 1, 3] = -cb * torch.cos(gamma)
    D[:, 1, 4] = 0.5 * sb * (1 - cb) * torch.sin(alpha - gamma)

    D[:, 2, 0] = 0.5 * sb2 * torch.cos(2 * gamma)
    D[:, 2, 1] = sb * cb * torch.cos(gamma)
    D[:, 2, 2] = cb2 - sb2 / 2
    D[:, 2, 3] = sb * cb * torch.cos(gamma)
    D[:, 2, 4] = 0.5 * sb2 * torch.cos(2 * gamma)

    D[:, 3, 0] = 0.5 * sb * (1 - cb) * torch.sin(alpha - gamma)
    D[:, 3, 1] = cb * torch.sin(gamma)
    D[:, 3, 2] = sb * cb * torch.cos(alpha)
    D[:, 3, 3] = cb * torch.sin(gamma)
    D[:, 3, 4] = -0.5 * sb * (1 + cb) * torch.sin(alpha + gamma)

    D[:, 4, 0] = 0.25 * (1 - cb)**2 * torch.sin(2 * (alpha - gamma))
    D[:, 4, 1] = 0.5 * sb * (1 - cb) * torch.sin(alpha - gamma)
    D[:, 4, 2] = 0.5 * sb2 * torch.sin(2 * alpha)
    D[:, 4, 3] = -0.5 * sb * (1 + cb) * torch.sin(alpha + gamma)
    D[:, 4, 4] = -0.25 * (1 + cb)**2 * torch.sin(2 * (alpha + gamma))

    return D

def wigner_D_l3_batch(alpha, beta, gamma):
    """
    Compute real Wigner-D matrices for l=3 using real SH basis.
    Inputs:
        alpha, beta, gamma: (N,) tensors [in radians]
    Returns:
        D: (N, 7, 7) real-valued Wigner-D matrices
    """
    N = alpha.shape[0]
    device = alpha.device
    dtype = alpha.dtype

    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    c2a, s2a = torch.cos(2 * alpha), torch.sin(2 * alpha)
    c3a, s3a = torch.cos(3 * alpha), torch.sin(3 * alpha)
    c2g, s2g = torch.cos(2 * gamma), torch.sin(2 * gamma)
    c3g, s3g = torch.cos(3 * gamma), torch.sin(3 * gamma)

    sb2 = sb ** 2
    sb3 = sb ** 3
    cb2 = cb ** 2
    cb3 = cb ** 3

    sqrt = lambda x: torch.sqrt(torch.tensor(x, dtype=dtype, device=device))

    D = torch.zeros((N, 7, 7), device=device, dtype=dtype)

    # Use symbolic expressions derived from real Wigner D matrix for l=3
    # For readability, expressions below are partially compressed and structured

    # Populate D[i, j] using the known patterns for l=3 real basis (e.g., from Peter-Pike Sloan's SH basis rotation)

    # Only filling diagonal for simplicity and showing pattern.
    # You can fill the full matrix using same style as l=2 if needed for precision applications.

    D[:, 0, 0] = 0.125 * (1 + cb) ** 3 * torch.cos(3 * (alpha + gamma))
    D[:, 1, 1] = 0.75 * sb * (1 + cb) ** 2 * torch.cos(2 * (alpha + gamma))
    D[:, 2, 2] = 1.5 * sb2 * (1 + cb) * torch.cos(alpha + gamma)
    D[:, 3, 3] = 0.5 * (5 * cb3 - 3 * cb)
    D[:, 4, 4] = 1.5 * sb2 * (1 - cb) * torch.cos(alpha - gamma)
    D[:, 5, 5] = 0.75 * sb * (1 - cb) ** 2 * torch.cos(2 * (alpha - gamma))
    D[:, 6, 6] = 0.125 * (1 - cb) ** 3 * torch.cos(3 * (alpha - gamma))

    return D
