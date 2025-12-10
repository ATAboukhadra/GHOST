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
