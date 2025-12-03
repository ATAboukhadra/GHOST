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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, obj_loss, geo_consistency_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, psnr_masked, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if opt.prune_unseen:
        num_views = len(scene.getTrainCameras())
        if opt.prune_unseen_interval == -1:
            if num_views*2 > opt.iterations*0.1:
                opt.prune_unseen_interval = int(num_views*1.5)
                print(f"[INIT] Pruning interval was automatically set to {opt.prune_unseen_interval}. (1.5x{num_views} training views)")
            else:
                opt.prune_unseen_interval = int(num_views*2)
                print(f"[INIT] Pruning interval was automatically set to {opt.prune_unseen_interval}. (2x{num_views} training views)")
        else:
            print(f"[INIT] Pruning interval is set to {opt.prune_unseen_interval}. ({num_views} training views)")
    
    if dataset.use_object_mask:
        if scene.getTrainCameras()[0].object_mask is None:
            raise ValueError("Running with 'use_object_mask' option but could not load object masks from dataset!")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_bg_for_log = 0.0
    ema_psnr_for_log = 0.0

    render_dir = os.path.join(dataset.model_path, 'rendered_views')
    logging_dir = os.path.join(dataset.model_path, "logs")

    print("\n[TRAIN] Starting optimization.")
    print("--------------------------------", flush=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    xyz_prev = None
    seen_accum = None
    best_icp_error = 1e10
    best_iter = -1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if dataset.use_object_mask:
            if viewpoint_cam.object_mask.sum() == 0:
                continue
        
        if dataset.hand in ['right', 'left', 'both']:
            gaussians.set_image_transform(viewpoint_cam.file_id, viewpoint_cam)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rend_alpha"] 
        rgba = torch.cat([image, alpha], dim=0)     # shape: (4, H, W)
        
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_object_mask:
            object_mask = viewpoint_cam.object_mask.cuda()
            gt_image = gt_image * object_mask#Mask RGB of ground truth image
            image = image * object_mask#Mask RGB of rendered image

        Ll1 = l1_loss(image, gt_image)
        loss_base = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        lambda_interact = 0.01

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        loss_normal = lambda_normal * (normal_error).mean()
        loss_dist = lambda_dist * (rend_dist).mean()
        
        if gaussians.transforms is not None and gaussians.is_grasping:
            object_center, hand_center = gaussians.object_center, gaussians.hand_center
            # Compute distance between object and hand center
            dist = torch.norm(object_center - hand_center, dim=1)
            # Compute the loss as the mean distance
            loss_interact = lambda_interact * dist.mean()
            # print(loss_dist)
        else:
            loss_interact = 0.0
        # loss
        total_loss = loss_base + loss_dist + loss_normal
        # print(loss_interact, total_loss)
        # lambda_scale_reg = 0.1
        # scale_reg = lambda_scale_reg * torch.mean((gaussians._scaling - gaussians._scaling_init) ** 2)
        # total_loss += scale_reg
        # lambda_scaling_reg = 0.05
        # scaling_diff = gaussians._scaling - gaussians._scaling_init
        # scale_increase = torch.clamp(scaling_diff, min=0.0)  # Only positive changes
        # scale_reg = lambda_scaling_reg * torch.mean(scale_increase ** 2)
        # total_loss += scale_reg
        # scale_penalty = gaussians.get_scaling[hand_mask].max(dim=1).values - max_scale
        # scale_loss = torch.mean(torch.clamp(scale_penalty, min=0.0) ** 2)
        # total_loss += 0.1 * scale_loss

        rend_alpha = render_pkg['rend_alpha']
        if dataset.use_object_mask:
            if viewpoint_cam.background_ignore_mask is not None:
                background_ignore_mask = viewpoint_cam.background_ignore_mask.cuda()
                # logical or with object mask
                background_ignore_mask = torch.clamp(background_ignore_mask + object_mask, 0, 1)  
                loss_background = obj_loss(rend_alpha, background_ignore_mask, opt.lambda_background)
            else:
                loss_background = obj_loss(rend_alpha, object_mask, opt.lambda_background)#False -> make transparent

                # print("weighting frame {} with iou {}".format(frame_id, gaussians.iou_hand[frame_id]))
            total_loss += loss_background
        
        if dataset.use_obj_prior and iteration > 500:
            # Make sure both are in the SAME space (e.g., camera space for the current view)
            # obj_mask = gaussians.binding == gaussians.identity_binding_index

            G = gaussians.get_gaussians_position()        # (Ng, 3)
            P = gaussians.obj_prior_pc                   # (Np, 3) subsample to ~10–30k

            # if gaussians.hold_pc is not None and iteration % 5000 == 0 and iteration == 30000:
            #     H = gaussians.hold_pc
            #     # sample 1000 points from H and G
            #     indices = torch.randperm(H.shape[0])[:1000]
            #     icp_error, _, _ = compute_icp_metrics_from_pcd(H[indices], G[indices], num_iters=600)
            #     # round to 2 decimal places
            #     icp_error = icp_error.round(2)
            #     print(f"[ITER {iteration}] ICP error to hold object: {icp_error:.2f}cm")
            #     if icp_error < best_icp_error:
            #         best_iter = iteration
            #         best_icp_error = icp_error
                    # best_found = Tr

            # Optional confidence: 1 if visible this frame, else 0 (simple & effective)
            # conf = render_pkg["seen"].float()             # (Ng,)
            conf = None

            loss_geo, cd = geo_consistency_loss(G, P, conf=conf, tau_out=opt.tau_out, w_out=opt.w_out, tau_fill=opt.tau_fill, w_fill=opt.w_fill)
            lambda_geo = opt.lambda_geo
            total_loss = total_loss + lambda_geo * loss_geo

        # if gaussians.iou_hand is not None:
        #     frame_id = int(viewpoint_cam.image_name)
        #     total_loss = torch.round(gaussians.iou_hand[frame_id]) * total_loss

        # opacity_target = 0.05
        # opacity_reg = torch.mean((gaussians.get_opacity - opacity_target).clamp(min=0.0) ** 2)
        # total_loss += 0.01 * opacity_reg

        total_loss.backward()

        if opt.optimize_mano:
            gaussians.update_hand_transformations()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_base.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * loss_dist.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * loss_normal.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                if dataset.use_object_mask:
                    ema_bg_for_log = 0.4 * loss_background.item() + 0.6 * ema_bg_for_log
                    ema_psnr_for_log = 0.4 * psnr_masked(image, gt_image, object_mask).mean().item() + 0.6 * ema_psnr_for_log
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "psnr_masked": f"{ema_psnr_for_log:.{2}f}",
                        "bg": f"{ema_bg_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz):_}"
                    }
                else:
                    ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().item() + 0.6 * ema_psnr_for_log
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "psnr_full": f"{ema_psnr_for_log:.{2}f}",
                        "bg": "disabled",
                        "Points": f"{len(gaussians.get_xyz):_}"
                    }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                elapsed_time = progress_bar.format_dict["elapsed"]
                progress_bar.close()
                with open(os.path.join(logging_dir, "elapsed time.txt"), 'w') as log_file:
                    log_file.write(f"elapsed time: {elapsed_time/60.0} minutes\n")
                    log_file.write(f"Gaussians: {len(gaussians.get_xyz):_}\n")


            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            if iteration == args.iterations:
                psnr_dict = training_report(tb_writer, iteration, Ll1, loss_base, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), False)
            else:
                psnr_dict = training_report(tb_writer, iteration, Ll1, loss_base, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if best_iter == iteration:
                best_psnr = psnr_dict

            if (iteration in saving_iterations) or iteration > opt.iterations - 20:
                if iteration == args.iterations:
                    print("[ITER {}] Saving Gaussians".format(iteration))#No line break because end of progress bar
                else:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                
                frame_id = int(viewpoint_cam.image_name)
                scene.save(iteration, frame_num=frame_id)


            if opt.prune_unseen:
                gaussians.add_seen_status(render_pkg["seen"])#seen/unseen according to rendering (bool tensor)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # gaussians.add_densification_stats(viewspace_point_tensor, render_pkg["seen"])#use seen status to add stats

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # import ipdb;ipdb.set_trace()
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if dataset.hand == 'none':
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                        # conf = gaussians.seen.float()

            if opt.prune_unseen:
                if iteration % opt.prune_unseen_interval == 0:
                    gaussians.prune_unseen()
            
            # Optimizer step
            if iteration < opt.iterations:

                if dataset.hand in ['right', 'left', 'both']:
                    if not gaussians._xyz.grad is None:

                        object_gaussians_mask = (gaussians.binding == gaussians.identity_binding_index)
                        # print(object_gaussians_mask.sum(), "object gaussians")
                        gaussians._xyz.grad[object_gaussians_mask] = 0 #don't change position
                        gaussians._scaling.grad[object_gaussians_mask] = 0 #don't change scale
                        gaussians._rotation.grad[object_gaussians_mask] = 0 #don't change rotation
                        gaussians._features_dc.grad[object_gaussians_mask] = 0 #don't change features
                        gaussians._features_rest.grad[object_gaussians_mask] = 0 #don't change features
                        # gaussians._opacity.grad[object_gaussians_mask] = 0 #don't change features

                        if opt.disable_hand_psr:
                            print("[TRAIN] Disabling hand geometry optimization")
                            gaussians._xyz.grad[:] = 0#don't change position
                            gaussians._scaling.grad[:] = 0#don't change scale
                            gaussians._rotation.grad[:] = 0#don't change rotation
                
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                if iteration in saving_iterations or iteration > opt.iterations - 10:
                    print("[ITER {}] Saving Checkpoint".format(iteration))#No line break due to line break for saving
                else:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), dataset.model_path + "/chkpnt" + str(iteration) + ".pth")

        # with torch.no_grad():
        #     if dataset.use_gui:
        #         if network_gui.conn == None:
        #             network_gui.try_connect(dataset.render_items)
        #         while network_gui.conn != None:
        #             try:
        #                 net_image_bytes = None
        #                 custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
        #                 if custom_cam != None:
        #                     if dataset.hand in ['right', 'left', 'both']:
        #                         scene.gaussians.set_image_transform(custom_cam.file_id, custom_cam)

        #                     render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
        #                     net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
        #                     net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #                 metrics_dict = {
        #                     "#": gaussians.get_opacity.shape[0],
        #                     "loss": ema_loss_for_log
        #                     # Add more metrics as needed
        #                 }
        #                 # Send the data
        #                 network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
        #                 if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #                     break
        #             except Exception as e:
        #                 # raise e
        #                 network_gui.conn = None


    print("--------------------------------")
    print("[TRAIN] Completed optimization.")

    if opt.prune_unseen:
        print("\n[TRAIN] Pruning unseen of final model.")
        print(f"[TRAIN] Gaussians before pruning: {len(gaussians.get_xyz)}")
        gaussians.clear_seen_status()
        viewpoint_stack = scene.getTrainCameras().copy()
        for viewpoint_cam in viewpoint_stack:
            if dataset.hand in ['right', 'left', 'both']:
                gaussians.set_image_transform(viewpoint_cam.file_id, viewpoint_cam)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            gaussians.add_seen_status(render_pkg["seen"])
        gaussians.prune_unseen()
        print(f"[TRAIN] Gaussians after pruning: {len(gaussians.get_xyz)}")

        print("[Train] Saving Gaussians")
        scene.save("pruned")

    print("\n[TRAIN] Optimized " + dataset.source_path)
    print("[TRAIN] Output folder " + dataset.model_path)

    # Export unique frames if requested
    # if dataset.export_unique_frames:
    export_outputs(scene, render, (pipe, background), os.path.join(dataset.model_path, "all_rendered_frames"), dataset)

    if best_iter != -1:
        metrics = {
            "ICP_best": best_icp_error,
            "best_iter": best_iter,
            "best_PSNR_train": best_psnr["train"],
            "best_PSNR_test": best_psnr["test"],
            # "ICP_last_iter": icp_error,
            "cd_last_iter": cd,
            "PSNR_train": psnr_dict["train"],
            "PSNR_test": psnr_dict["test"]
        }
        log_results(args.csv_file, args, metrics)

import csv, os

def log_results(csv_file, args, metrics):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "sequence","lambda_geo","tau_out","tau_fill","w_fill","w_out","run_id",
                "ICP_best","best_iter","best_PSNR_train","best_PSNR_test",
                "ICP_last_iter","cd_last_iter","PSNR_train","PSNR_test"
            ])
        writer.writerow([
            args.source_path.split("/")[1],  # sequence name
            args.lambda_geo, args.tau_out, args.tau_fill, args.w_fill, args.w_out, args.run_id,
            metrics["ICP_best"], metrics["best_iter"], metrics["best_PSNR_train"], metrics["best_PSNR_test"],
            metrics["ICP_last_iter"], metrics["cd_last_iter"], metrics["PSNR_train"], metrics["PSNR_test"]
        ])

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("[INIT] Creating folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    render_dir = os.path.join(args.model_path, "rendered_views")
    print(f"[INIT] Creating folder: {render_dir}")
    os.makedirs(render_dir, exist_ok = True)

    logging_dir = os.path.join(args.model_path, "logs")
    print(f"[INIT] Creating folder: {logging_dir}")
    os.makedirs(logging_dir, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[INIT] Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, with_line_break=True):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    psnr_dict = {}
    if iteration in testing_iterations:
        logging_dir = os.path.join(scene.model_path, "logs")
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_mask_list = []
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if viewpoint.object_mask is not None:
                        object_mask = viewpoint.object_mask.to("cuda")
                    else:
                        object_mask = torch.zeros([0]).to("cuda")
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if not (object_mask.sum() == 0):
                        psnr_mask_list.append(psnr_masked(image, gt_image, object_mask).mean().double())

                if len(psnr_mask_list) > 0:
                    psnr_mask_sum = 0.0
                    for psnr_ in psnr_mask_list:
                        psnr_mask_sum += psnr_
                    psnr_mask = psnr_mask_sum / len(psnr_mask_list)
                else:
                    psnr_mask = 0.0
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                # round to 2 float places
                psnr_dict[config['name']] = round(psnr_mask.detach().cpu().numpy().item(), 2)
                
                if with_line_break:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_masked {}".format(iteration, config['name'], l1_test, psnr_test, psnr_mask))
                    with_line_break = False
                else:
                    print("[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_masked {}".format(iteration, config['name'], l1_test, psnr_test, psnr_mask))
                with open(os.path.join(logging_dir, f"metrics_{config['name']}.txt"), 'w') as log_file:
                    log_file.write(f"L1: {l1_test}\n")
                    log_file.write(f"PSNR (full): {psnr_test}\n")
                    log_file.write(f"PSNR (mask): {psnr_mask}\n")
                    log_file.write(f"Total images: {len(config['cameras'])}\n")
                    log_file.write(f"Images with non-zero mask: {len(psnr_mask_list)}\n")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_masked', psnr_mask, iteration)

        torch.cuda.empty_cache()

    return psnr_dict

# @torch.no_grad()
# def export_unique_frames(scene, render_func, render_args, output_dir, dataset):
#     print(f"[EXPORT] Saving all unique training and testing RGBA frames to: {output_dir}")
#     os.makedirs(output_dir, exist_ok=True)

#     # Render and save all unique training frames
#     train_cams = scene.getTrainCameras()
#     for cam in train_cams:
#         if dataset.hand in ['right', 'left', 'both']:
#             scene.gaussians.set_image_transform(cam.file_id, cam)
        
#         render_pkg = render_func(cam, scene.gaussians, *render_args)
#         image = render_pkg["render"]
#         alpha = render_pkg["rend_alpha"]
#         rgba = torch.cat([image, alpha], dim=0)
#         rgba_img = (rgba.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
#         Image.fromarray(rgba_img, mode='RGBA').save(os.path.join(output_dir, f"{cam.image_name}.png"))

#     # Render and save all test frames
#     test_cams = scene.getTestCameras()
#     for cam in test_cams:
#         if dataset.hand in ['right', 'left', 'both']:
#             scene.gaussians.set_image_transform(cam.file_id, cam)

#         render_pkg = render_func(cam, scene.gaussians, *render_args)
#         image = render_pkg["render"]
#         alpha = render_pkg["rend_alpha"]
#         rgba = torch.cat([image, alpha], dim=0)
#         rgba_img = (rgba.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
#         Image.fromarray(rgba_img, mode='RGBA').save(os.path.join(output_dir, f"{cam.image_name}.png"))
#         if not os.path.exists(output_dir.replace('all', 'test')):
#             os.makedirs(output_dir.replace('all', 'test'), exist_ok=True)
#         Image.fromarray(rgba_img, mode='RGBA').save(os.path.join(output_dir.replace('all', 'test'), f"{cam.image_name}.png"))

#     print(f"[EXPORT] Done. Saved {len(train_cams)} training and {len(test_cams)} testing frames.")

@torch.no_grad()
def export_outputs(scene, render_func, render_args, output_dir, dataset):
    print(f"[EXPORT] Saving all unique training/testing RGBA frames + submission outputs to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    fnames = []
    results = {
        'v_posed.left': [],
        'verts.right': [],
        'verts.object': [],

        'v3d_c.left': [],
        'v3d_c.right': [],
        'v3d_c.object': [],
        'j3d_c.left': [],
        'j3d_c.right': [],

        'root.left': [],
        'j3d_ra.left': [],
        'root.right': [],
        'j3d_ra.right': [],
        'root.object': [],
        'v3d_ra.object': [],

        'v3d_right.object': [],
        'v3d_left.object': [],
    }

    all_cams = scene.getTrainCameras() + scene.getTestCameras()
    # sequence_name = os.path.basename(os.path.dirname(os.path.dirname(dataset.model_path)))
    sequence_name = dataset.model_path.split('/')[-4]
    print(f"[EXPORT] Sequence name: {sequence_name}")

    for cam in all_cams:
        # frame_idx = int(cam.image_name)
        image_path = f"./data/{sequence_name}/build/image/{cam.image_name}.png"
        fnames.append(image_path)

        if dataset.hand in ['right', 'left', 'both']:
            scene.gaussians.set_image_transform(cam.file_id, cam)

        render_pkg = render_func(cam, scene.gaussians, *render_args)
        image = render_pkg["render"]
        alpha = render_pkg["rend_alpha"]
        rgba = torch.cat([image, alpha], dim=0)
        rgba_img = (rgba.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(rgba_img, mode='RGBA').save(os.path.join(output_dir, f"{cam.image_name}.png"))

        # === export tensors ===
        if dataset.hand in ['right', 'left', 'both']:
            export = scene.gaussians.export_submission_outputs(cam, sequence_name)
            keys_to_delete = []
            for k in results:
                if k in export.keys():
                    results[k].append(export[k].cpu())
                else:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del results[k]
                
    if dataset.hand not in ['right', 'left', 'both']:
        return

    # Sort all frame outputs by fnames (image path)
    zipped = list(zip(fnames, *[results[k] for k in results]))
    zipped.sort(key=lambda x: x[0])  # sort by image path string

    # Unzip
    sorted_fnames = [x[0] for x in zipped]
    sorted_tensors = {k: [x[i + 1] for x in zipped] for i, k in enumerate(results)}

    # Print shapes
    tensors_dict = {k: torch.cat(sorted_tensors[k], dim=0) for k in sorted_tensors}
    
    # for k in tensors_dict:
    #     print(f"[EXPORT] {k} shape: {tensors_dict[k].shape}")

    save_dict = {
        "fnames": sorted_fnames,
        "full_seq_name": sequence_name,
        **tensors_dict
    }

    # Extend save_dict with subdirectory for faces {right, left, object}
    save_dict['faces'] = {
        'right': export['faces']['right'],
        'left': export['faces']['left'],
        'object': export['faces']['object']
    }

    #scan all tensors and print shapes
    # for k, v in save_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"[EXPORT] {k} shape: {v.shape}")
    #     elif isinstance(v, list):
    #         print(f"[EXPORT] {k} length: {len(v)}")
    #     elif isinstance(v, dict):
    #         for sub_k, sub_v in v.items():
    #             if isinstance(sub_v, torch.Tensor):
    #                 print(f"[EXPORT] {k}.{sub_k} shape: {sub_v.shape}")
    #             elif isinstance(sub_v, list):
    #                 print(f"[EXPORT] {k}.{sub_k} length: {len(sub_v)}")

    save_path = os.path.join(dataset.model_path, f"{sequence_name}.pt")
    torch.save(save_dict, save_path)
    print(f"[EXPORT] Submission file saved to {save_path}")
    print(f"[EXPORT] Done. Saved {len(all_cams)} frames.")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("[INIT] Optimizing " + args.source_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.use_gui:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\n[TRAIN] Training terminated successfully.")