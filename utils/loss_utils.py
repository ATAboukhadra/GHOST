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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    if mask is None:
        return _ssim(img1, img2, window, window_size, channel, size_average)
    else:
        return _ssim_masked(img1*mask, img2*mask, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim_masked(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map[mask.repeat(1, 3, 1, 1)>0].mean()#This includes windows on the mask boundaries, which reach into areas outside of the mask
    else:
        raise NotImplementedError()

def sky_loss(alpha, sky_mask, lambda_):
    #mask==1 -> sky
    #alpha==0 -> background
    l_sky = torch.mean(alpha*sky_mask)#if sky, alpha*1 (encourages to use background), if not sky, 0 (can be anything)
    l_sky *= lambda_
    return l_sky

def obj_loss(alpha, obj_mask, lambda_):
    #mask==0 -> not the object
    #alpha==0 -> background
    non_obj_mask = torch.ones_like(obj_mask) - obj_mask
    l_sky = torch.mean(alpha*non_obj_mask)#if not object, alpha*1 (encourages to use background), if object, 0 (can be anything)
    l_sky *= lambda_
    return l_sky


# def geo_consistency_loss(
#     G_xyz: torch.Tensor,          # (Ng,3) object gaussian centers (same space as prior)
#     prior_xyz: torch.Tensor,      # (Np,3) sampled prior point cloud
#     conf: torch.Tensor = None,    # (Ng,) confidence in [0,1]; e.g., visibility or EMA
#     tau_fill: float = 0.01,       # small gaps threshold (fill occlusions)
#     tau_out:  float = 0.03,       # outlier threshold (prune strays)
#     gamma:    float = 2.0,        # protects high‑confidence gaussians: (1 - conf)^gamma
#     w_fill:   float = 1.0,        # weight for hole filling (prior→gaussians)
#     w_out:    float = 0.5,        # weight for outlier penalty (gaussians→prior)
#     k_fill:   int   = 1,          # use 1 (nearest) or small k (e.g., 3) for thin gaps
# ) -> torch.Tensor:
#     """
#     L = w_fill * mean_p   [ ReLU(dist_p_to_G - tau_fill)^2 ]
#       + w_out  * mean_g   [ ReLU(dist_g_to_P - tau_out )^2 * (1 - conf_g)^gamma ]

#     - Fill small occlusion gaps from the prior (P→G).
#     - Penalize outlier gaussians but protect visible ones via `conf` (G→P).
#     """
#     if G_xyz.numel() == 0 or prior_xyz.numel() == 0:
#         return G_xyz.new_tensor(0.0)

#     # Pairwise distances once
#     D = torch.cdist(prior_xyz, G_xyz)    # (Np, Ng)

#     # ---- P→G (fill holes)
#     if k_fill > 1 and D.shape[1] >= k_fill:
#         d_p = torch.topk(D, k=k_fill, dim=1, largest=False).values.mean(dim=1)  # (Np,)
#     else:
#         d_p = D.min(dim=1).values                                             # (Np,)
#     r_p = F.relu(d_p - tau_fill)
#     L_fill = (r_p * r_p).mean()

#     # ---- G→P (penalize outliers, protect visible)
#     d_g = D.min(dim=0).values                                                 # (Ng,)
#     r_g = F.relu(d_g - tau_out)
#     if conf is not None:
#         w_conf = (1.0 - conf.clamp(0, 1)).pow(gamma)
#         r_g = r_g * w_conf
#     L_out = (r_g * r_g).mean()

#     return w_fill * L_fill + w_out * L_out

# def geo_consistency_loss(
#     G_xyz: torch.Tensor,          # (Ng,3) gaussian centers
#     prior_xyz: torch.Tensor,      # (Np,3) prior point cloud (outer surface)
#     conf: torch.Tensor = None,    # (Ng,) confidence [0,1]
#     tau_out: float = 0.03,        # distance threshold for outliers
#     gamma: float = 2.0,           # protects visible gaussians
#     w_out: float = 1.0            # weight
# ) -> torch.Tensor:
#     """
#     Penalize Gaussians that are far from the prior surface.

#     L = mean_g [ ReLU(dist_g_to_P - tau_out)^2 * (1 - conf_g)^gamma ]

#     - Keeps Gaussians close to the prior.
#     - Protects high-confidence (visible) Gaussians from being pulled back.
#     - No hole-filling (that’s handled separately via densify_from_prior).
#     """
#     if G_xyz.numel() == 0 or prior_xyz.numel() == 0:
#         return G_xyz.new_tensor(0.0)

#     # Distance from each Gaussian to nearest prior point
#     d_g = torch.cdist(G_xyz, prior_xyz).min(dim=1).values   # (Ng,)

#     # Only penalize if beyond tau_out
#     r_g = F.relu(d_g - tau_out)

#     # Confidence weighting
#     if conf is not None:
#         w_conf = (1.0 - conf.clamp(0, 1)).pow(gamma)
#         r_g = r_g * w_conf

#     return w_out * (r_g * r_g).mean()

def geo_consistency_loss(
    G_xyz: torch.Tensor,          # (Ng,3) gaussian centers
    prior_xyz: torch.Tensor,      # (Np,3) prior surface points
    conf: torch.Tensor = None,    # (Ng,) confidence [0,1]
    tau_out: float = 0.03,
    tau_fill: float = 0.01,
    gamma: float = 2.0,
    w_out: float = 1.0,
    w_fill: float = 1.0
):
    if G_xyz.numel() == 0 or prior_xyz.numel() == 0:
        return G_xyz.new_tensor(0.0)

    # Gaussian→prior (outlier suppression)
    d_g = torch.cdist(G_xyz, prior_xyz).min(dim=1).values
    r_g = F.relu(d_g - tau_out)
    # r_g = torch.log1p(torch.exp((d_g - tau_out) * 10.0)) / 10.0

    if conf is not None:
        w_conf = (1.0 - conf.clamp(0, 1)).pow(gamma)
        r_g = r_g * w_conf
    L_out = (r_g * r_g).mean()

    # Prior→Gaussian (hole filling)
    d_p = torch.cdist(prior_xyz, G_xyz).min(dim=1).values
    r_p = F.relu(d_p - tau_fill)
    L_fill = (r_p * r_p).mean()

    # return chamfer distance in mm between gaussians and prior
    cd = (d_g.mean()) * 1000.0
    cd = round(cd.detach().cpu().item(), 2)
    # print(cd)

    return w_out * L_out + w_fill * L_fill, cd

# import torch
# import torch.nn.functional as F

# def geo_consistency_loss(
#     G_xyz: torch.Tensor,          # (Ng,3) gaussian centers
#     prior_xyz: torch.Tensor,      # (Np,3) prior surface points
#     # conf: torch.Tensor = None,    # (Ng,) optional confidence
#     tau: float = 0.02,            # tolerance (same for both directions)
#     gamma: float = 2.0
# ):
#     """
#     Simpler geometric consistency loss:
#     - Encourages Gaussians to lie near prior surface (outlier suppression).
#     - Encourages prior surface to be covered by Gaussians (hole filling).
#     - Uses one intuitive tolerance (tau).
#     """

#     if G_xyz.numel() == 0 or prior_xyz.numel() == 0:
#         return G_xyz.new_tensor(0.0)

#     # Gaussian → Prior
#     d_g = torch.cdist(G_xyz, prior_xyz).min(dim=1).values
#     r_g = F.softplus(d_g - tau, beta=10.0)   # smooth margin
#     # if conf is not None:
#     #     w_conf = (1.0 - conf.clamp(0, 1)).pow(gamma)
#     #     r_g = r_g * w_conf
#     L_out = (r_g * r_g).mean()

#     # Prior → Gaussian
#     d_p = torch.cdist(prior_xyz, G_xyz).min(dim=1).values
#     r_p = F.softplus(d_p - tau, beta=10.0)
#     L_fill = (r_p * r_p).mean()

#     # Balanced sum
#     return 0.5 * (L_out + L_fill)


@torch.no_grad()
def densify_from_prior(
    gaussians,
    prior_surface,               # (Np,3) sampled surface points
    conf=None,                   # (Ng,) confidence [0,1]
    tau_fill=0.01,               # fill threshold (~1 cm at object scale)
    max_new=2000,
):
    G = gaussians.get_gaussians_position()
    if G.numel() == 0 or prior_surface.numel() == 0:
        return

    # use only low-confidence gaussians as reference
    if conf is not None:
        low = conf < 0.6
        G_ref = G[low] if low.any() else G
    else:
        G_ref = G

    # distances from prior surface to current gaussians
    d = torch.cdist(prior_surface, G_ref).min(dim=1).values  # (Np,)
    holes = d > tau_fill
    if not holes.any():
        return

    # pick farthest hole points directly from the surface
    hole_pts = prior_surface[holes]
    hole_d   = d[holes]
    K = min(hole_pts.shape[0], max_new)
    new_xyz = hole_pts[torch.topk(hole_d, k=K, largest=True).indices]

    # spawn gaussians exactly on surface (no jitter needed)
    gaussians.spawn_at_positions(new_xyz, init_scale_frac=1.0, init_alpha_frac=0.5)

    return new_xyz.shape[0]