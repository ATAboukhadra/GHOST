import torch
import os
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
from utils.grasping_utils import compute_grasping
from utils.colmap_readmodel import *

def project_points(points, cam_intr):
    points_2d = torch.matmul(points, cam_intr.t())
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d


def load_mesh_from_obj(file_path, device='cpu'):
    """
    Load vertices and faces from an .obj file.
    """
    mesh = trimesh.load(file_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    return verts, faces


def compute_loss(hand_proj_2d, target_2d, hand_verts, object_points, translations, is_grasping, rendered_mask, hand_mask_gt, prior_verts=None, frame_confidences=None, bb_box_thresh=20000):
    # is_grasping: 0 = no grasp, 1 = right, 2 = left

    if target_2d.shape[0] > 21:

        right_target_2d = target_2d[:21]
        left_target_2d = target_2d[21:]
        conf_thresh = 0.95
        proj_loss = torch.tensor(0.0, device=hand_proj_2d.device)
        right_bb = (torch.min(right_target_2d, dim=0)[0], torch.max(right_target_2d, dim=0)[0])
        right_bb_area = (right_bb[1][0] - right_bb[0][0]) * (right_bb[1][1] - right_bb[0][1])
        if right_bb_area > bb_box_thresh:
            proj_loss += torch.nn.functional.l1_loss(hand_proj_2d[:21], right_target_2d)

        # keypoints that have higher than 0.9 confidence
        # if len(frame_confidences['right']) > 0:
        #     high_conf_indices = [i for i, conf in enumerate(frame_confidences['right']) if conf > conf_thresh]
        #     if len(high_conf_indices) > 0:
        #         proj_loss += torch.nn.functional.l1_loss(hand_proj_2d[:21][high_conf_indices], right_target_2d[high_conf_indices])
        
        left_bb = (torch.min(left_target_2d, dim=0)[0], torch.max(left_target_2d, dim=0)[0])
        left_bb_area = (left_bb[1][0] - left_bb[0][0]) * (left_bb[1][1] - left_bb[0][1])
        if left_bb_area > bb_box_thresh:
            proj_loss += torch.nn.functional.l1_loss(hand_proj_2d[21:], left_target_2d)

        # if len(frame_confidences['left']) > 0:
        #     high_conf_indices = [i for i, conf in enumerate(frame_confidences['left']) if conf > conf_thresh]
        #     if len(high_conf_indices) > 0:
        #         proj_loss += torch.nn.functional.l1_loss(hand_proj_2d[21:][high_conf_indices], left_target_2d[high_conf_indices])

    else:
        proj_loss = torch.nn.functional.l1_loss(hand_proj_2d, target_2d)

    smoothness_loss = torch.mean((translations[1:] - translations[:-1])**2)

    loss_dict = {
        'proj_loss': proj_loss.item(),
        'smoothness_loss': smoothness_loss.item(),
    }
    if is_grasping:
        # print(is_grasping)
        if prior_verts is None:
            dist_matrix = torch.cdist(hand_verts, object_points)
            proximity_loss = torch.mean(torch.min(dist_matrix, dim=1)[0])
        else:
            dist_matrix_prior = torch.cdist(hand_verts, prior_verts)
            proximity_loss = torch.mean(torch.min(dist_matrix_prior, dim=1)[0])

        # proximity_loss 
        loss_dict['proximity_loss'] = proximity_loss.item() * 1000
        # print("Losses: ", proj_loss.item(), proximity_loss.item(), smoothness_loss.item())
        return proj_loss * 0.1 + proximity_loss + 10 * smoothness_loss, loss_dict
    else:
        return proj_loss * 0.1 + 10 * smoothness_loss, loss_dict

def filter_pc(point_cloud, device):

    # point_cloud = torch.tensor([p.xyz for p in points3D.values()], dtype=torch.float32, device=device)
    # Compute the mean and standard deviation of the point cloud
    mean = torch.mean(point_cloud, dim=0)
    std = torch.std(point_cloud, dim=0)
    # Filter points that are more than 2 standard deviations away from the mean
    mask = torch.all(torch.abs(point_cloud - mean) < 2 * std, axis=1)
    filtered_point_cloud = point_cloud[mask]

    return filtered_point_cloud

def visualize_grasping(preprocess_dir, base_path, is_grasping_flags):
    image_dir = os.path.join(base_path, "images")
    # print(len(is_grasping_flags), "frames to visualize grasping flags for.")
    # image list is only .png files in sorted order of
    images_list = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    for idx, image in tqdm(enumerate(images_list)):
        # print(idx)
        grasping_flag = is_grasping_flags[idx]
        img_path = os.path.join(image_dir, image)
        img = cv2.imread(img_path)

        # Draw the grasping flag on the image
        cv2.putText(img, f"Grasping: {grasping_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the modified image
        export_path = os.path.join(base_path, f"{preprocess_dir}/grasping/", f"{idx}_grasping.png")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        cv2.imwrite(export_path, img)

def load_and_compute_grasp(seq_name, hand_mesh_files, optimized_list, images, 
                           filtered_point_cloud,
                           hand_path, device, 
                           hand_masks, obj_masks,
                           translations, hand_proj_targets, hand_verts_seq, 
                           frames, is_grasping_flags, 
                           left_translations, hand_proj_targets_left, hand_verts_seq_left,
                           mano_rot_params, mano_rot_left_params, faces_right, faces_left
                           ):

    prev_point_cloud = None
    for i, mesh_path_r in tqdm(enumerate(hand_mesh_files)):
        frame_id_str = os.path.basename(mesh_path_r).split('_')[0]
        frame_id = int(frame_id_str)
        matched_img = None
        if 'object' in optimized_list:
            for img_id, img in images.items():
                if int(img.name.split('.')[0]) == frame_id:
                    matched_img = img
                    break
        
        # frame_id already extracted above
        mesh_path_r = os.path.join(hand_path, "hand_meshes", f"{frame_id}_1.obj")
        joints_3d_r_path = os.path.join(hand_path, "hand_joints", f"{frame_id}_1.xyz")
        joints_2d_r_path = os.path.join(hand_path, "hand_joints2d", f"{frame_id}_1.xyz")
        
        hand_mask_path = os.path.join(hand_path, "hand_rgba_right", f"{frame_id:04d}.png")
        obj_mask_path = os.path.join(hand_path, "obj_rgba", f"{frame_id:04d}.png")

        if not os.path.exists(mesh_path_r):
            continue

        hand_mask_rgba = cv2.imread(hand_mask_path, cv2.IMREAD_UNCHANGED)
        img_dim = hand_mask_rgba.shape[:2]
        hand_mask = hand_mask_rgba[:, :, 3]  # Extract alpha channel
        hand_masks.append(hand_mask)

        if os.path.exists(obj_mask_path) and 'object' in optimized_list:
            obj_mask_rgba = cv2.imread(obj_mask_path, cv2.IMREAD_UNCHANGED)
            obj_mask = obj_mask_rgba[:, :, 3]
            obj_masks.append(obj_mask)

        verts_r, _ = load_mesh_from_obj(mesh_path_r, device)
        #compute scale of the hand by finding furthest vertex from the wrist (first vertex)
        scale_hand = torch.norm(verts_r - verts_r[0], dim=1).max()

        joints_r = torch.tensor(np.loadtxt(joints_3d_r_path), dtype=torch.float32, device=device)
        joints2d_r = torch.tensor(np.loadtxt(joints_2d_r_path), dtype=torch.float32, device=device)

        transl_r = torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True)

        translations.append(transl_r)
        hand_proj_targets.append(joints2d_r)
        hand_verts_seq.append((verts_r, joints_r, faces_right, transl_r, 1))
        frames.append((frame_id, matched_img, transl_r))

        if matched_img is not None:
            R_obj = torch.tensor(qvec2rotmat(matched_img.qvec), dtype=torch.float32, device=device)
            T_obj = torch.tensor(matched_img.tvec, dtype=torch.float32, device=device) 
            transformed_point_cloud = torch.matmul(R_obj, filtered_point_cloud.T).T + T_obj

            T_obj = transformed_point_cloud.mean(dim=0)
            # compute scale of the object by finding furthest distance
            scale_obj = torch.norm(transformed_point_cloud - transformed_point_cloud[0], dim=1).max()
            # compute acceleration vectors
            if prev_point_cloud is not None:
                acc = torch.mean(torch.norm(transformed_point_cloud - prev_point_cloud, dim=1)).item()
                # print(f"Frame {frame_id}: Object Acceleration: {acc}")
            
            prev_point_cloud = transformed_point_cloud.clone()
            
        else:
            R_obj = None
            T_obj = None

        # Load left hand if available
        mesh_path_l = os.path.join(hand_path, "hand_meshes", f"{frame_id}_0.obj")
        joints_3d_l_path = os.path.join(hand_path, "hand_joints", f"{frame_id}_0.xyz")

        if 'left' in optimized_list and os.path.exists(mesh_path_l) and os.path.exists(joints_3d_l_path):

            verts_l, _ = load_mesh_from_obj(mesh_path_l, device)
            joints_l = torch.tensor(np.loadtxt(joints_3d_l_path), dtype=torch.float32, device=device)
            joints2d_l = torch.tensor(np.loadtxt(joints_2d_r_path.replace('_1.xyz', '_0.xyz')), dtype=torch.float32, device=device)
            transl_l = torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True)

            left_translations.append(transl_l)
            hand_proj_targets_left.append(joints2d_l)
            hand_verts_seq_left.append((verts_l, joints_l, faces_left, transl_l, 0))
            R_hand_l = mano_rot_left_params[i]
            T_hand_l = joints_l[0]  # Use wrist joint as reference

        else:
            R_hand_l = torch.zeros(3, dtype=torch.float32, device=device)
            T_hand_l = torch.zeros(3, dtype=torch.float32, device=device)

        is_grasping = 0

        R_hand_r = mano_rot_params[i]
        T_hand_r = joints_r[0]  # Use wrist joint as reference

        if 'object' in optimized_list:
            t_thres=1e-3 if 'arctic' in seq_name else 1e-4 # more lenient in single hand case because any object movement is likely due to the right hand
            r_thres=1e-2 if 'arctic' in seq_name else 1e-3
            th_thres=5e-3 if 'arctic' in seq_name else 1e-4
            if i > 0:
                is_grasping = compute_grasping(R_hand_r, R_hand_r_prev, 
                                            R_hand_l, R_hand_l_prev,
                                            T_hand_r, T_hand_r_prev,
                                            T_hand_l, T_hand_l_prev,
                                            R_obj, R_obj_prev,
                                            T_obj, T_obj_prev,
                                            t_thres=t_thres,
                                            r_thres=r_thres,
                                            th_thres=th_thres
                                            )

            R_hand_r_prev = R_hand_r
            R_hand_l_prev = R_hand_l
            T_hand_r_prev = T_hand_r  
            T_hand_l_prev = T_hand_l
            R_obj_prev = R_obj
            T_obj_prev = T_obj

        is_grasping_flags.append(is_grasping)
    
    return scale_hand, scale_obj, img_dim
