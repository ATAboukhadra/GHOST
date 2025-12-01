import numpy as np
import torch
import os
import cv2
import glob
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from utils.pytorch3d_renderer import project_3D_points
from pytorch3d.transforms import matrix_to_axis_angle

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def compute_area(bbox):
    """Compute area of a bounding box (x1, y1, x2, y2)."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def filter_by_area(bboxes, is_right, kps, confs, min_area, max_area):
    """Remove hands with too small or too large bounding box area."""
    results = [
        (bbox, right, kp, conf)
        for bbox, right, kp, conf in zip(bboxes, is_right, kps, confs)
        if min_area <= compute_area(bbox) <= max_area
    ]
    if not results:
        return [], [], [], []
    b, r, k, c = zip(*results)
    return list(b), list(r), list(k), list(c)

def filter_by_threshold(bboxes, is_right, kps, confs, thresh):
    """Remove low-confidence hands."""
    results = [
        (bbox, right, kp, conf)
        for bbox, right, kp, conf in zip(bboxes, is_right, kps, confs)
        if np.mean(conf) > thresh
    ]
    if not results:
        return [], [], [], []
    b, r, k, c = zip(*results)
    return list(b), list(r), list(k), list(c)

def select_best_hand(bboxes, is_right, kps, confs, hand_flag):
    """Select the best (highest confidence) hand of a given type."""
    hand_data = [
        (bbox, kp, conf)
        for bbox, right, kp, conf in zip(bboxes, is_right, kps, confs)
        if right == hand_flag
    ]
    if not hand_data:
        return [], [], []
    # Pick the one with highest mean confidence
    best_idx = np.argmax([np.mean(conf) for _, _, conf in hand_data])
    bbox, kp, conf = hand_data[best_idx]
    return [bbox], [kp], [conf]

def filter_hands(bboxes, is_right, kps, confs, conf_thresh=0.5, min_area=5000, max_area=100000):
    """
    Filter detected hands to keep at most one left and one right hand 
    with the highest average confidence score.

    Args:
        bboxes (list[np.ndarray]): Bounding boxes of detected hands (x1, y1, x2, y2).
        is_right (list[bool]): Flags indicating whether each hand is right-hand.
        kps (list[np.ndarray]): Keypoints for each detected hand.
        confs (list[np.ndarray]): Confidence scores for keypoints per hand.
        conf_thresh (float): Minimum mean confidence threshold to keep a detection.

    Returns:
        (bboxes, is_right, kps, confs): Filtered lists with at most one left and one right hand.
    """

    # Step 1: filter low-confidence detections
    bboxes, is_right, kps, confs = filter_by_threshold(bboxes, is_right, kps, confs, conf_thresh)

    # Step 2: filter small bounding boxes
    bboxes, is_right, kps, confs = filter_by_area(bboxes, is_right, kps, confs, min_area, max_area)

    # Step 2: select best right and best left hand
    right_bboxes, right_kps, right_confs = select_best_hand(bboxes, is_right, kps, confs, True)
    left_bboxes, left_kps, left_confs = select_best_hand(bboxes, is_right, kps, confs, False)

    conf_dict = {
        'right': right_confs,
        'left': left_confs
    }

    # Step 3: merge results
    final_bboxes = right_bboxes + left_bboxes
    final_kps = right_kps + left_kps
    final_confs = right_confs + left_confs
    final_is_right = [True] * len(right_bboxes) + [False] * len(left_bboxes)

    return final_bboxes, final_is_right, final_kps, final_confs, conf_dict



def load_kps(file_path, remap=True):
    # load the .xyz file
    kps = np.loadtxt(file_path)
    # mapping = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    # mapping = np.argsort(mapping)
    mapping = [ 0,  5,  6,  7,  9, 10, 11, 17, 18, 19, 13, 14, 15,  1,  2,  3,  4,  8, 12, 16, 20]
    if remap:
        kps = kps[mapping]

    return kps

def center_kps(kps):
    # subtract by root (first vertex)
    root = kps[0]
    kps -= root
    return kps

def compare_kps(kp1, kp2):
    # compute mpjpe between the two keypoints using torch
    kp1 = torch.tensor(kp1, dtype=torch.float32)
    kp2 = torch.tensor(kp2, dtype=torch.float32)

    if kp1.shape != kp2.shape:
        print(f"Keypoint shapes are different: {kp1.shape} vs {kp2.shape}")
        return None
    else:
        # print('----', mesh1, mesh2, '---')
        error = torch.mean(torch.norm(kp1 - kp2, dim=1))

    return error


def compare_with_hold(seq_name, aligned=False):
    ref_root = f'../hold_stuff/hold_{seq_name}_meshes/'
    dir_name = 'hand_joints_aligned' if aligned else 'hand_joints'
    our_predictions_root = f'../data/{seq_name}/gs_preprocessing/{dir_name}/'

    # Paths to meshes and predictions
    right_mesh_files = sorted(glob.glob(os.path.join(ref_root, 'v3d_c_right_*.obj')))
    left_mesh_files = sorted(glob.glob(os.path.join(ref_root, 'v3d_c_left_*.obj')))

    hold_right_preds = os.path.join(ref_root, 'j3d_ra_right.pt')
    hold_left_preds = os.path.join(ref_root, 'j3d_ra_left.pt')

    hold_kps_right = torch.load(hold_right_preds)
    hold_kps_left = torch.load(hold_left_preds)

    # ---- Right Hand ----
    errors_right, c_right = 0, 0
    for i, right_mesh_file in enumerate(right_mesh_files):
        frame_id = int(os.path.basename(right_mesh_file).split('_')[3].split('.')[0])
        pred_file = os.path.join(our_predictions_root, f'{frame_id}_1.xyz')

        pred_kps = load_kps(pred_file)
        pred_kps = center_kps(pred_kps)

        error = compare_kps(hold_kps_right[i], pred_kps)
        if error is not None:
            errors_right += (error * 1000)
            c_right += 1

    mpjpe_right = errors_right / max(c_right, 1)
    print(f"Total Right hand MPJPE: {mpjpe_right:.2f} mm")

    # ---- Left Hand ----
    errors_left, c_left = 0, 0
    for i, left_mesh_file in enumerate(left_mesh_files):
        frame_id = int(os.path.basename(left_mesh_file).split('_')[3].split('.')[0])
        pred_file = os.path.join(our_predictions_root, f'{frame_id}_0.xyz')

        pred_kps = load_kps(pred_file)
        pred_kps = center_kps(pred_kps)

        error = compare_kps(hold_kps_left[i], pred_kps)
        if error is not None:
            errors_left += (error * 1000)
            c_left += 1

    mpjpe_left = errors_left / max(c_left, 1)
    print(f"Total Left hand MPJPE: {mpjpe_left:.2f} mm")

    # ---- Save Final Results ----
    final_out_path = f'../data/{seq_name}/gs_preprocessing/mpjpe_errors.txt'
    with open(final_out_path, 'w') as f:
        f.write(f"MPJPE Right: {mpjpe_right:.2f} mm\n")
        f.write(f"MPJPE Left:  {mpjpe_left:.2f} mm\n")

    print(f"Saved error summary to {final_out_path}")

def compare_with_gt(seq_name, aligned=False):
    ref_root = f'../data/{seq_name}/build/gt/'
    if not os.path.exists(ref_root):
        print(f"GT folder does not exist: {ref_root}")
        return 0, 0, 0, 0
    
    dir_name = 'hand_joints_aligned' if aligned else 'hand_joints'
    our_predictions_root = f'../data/{seq_name}/gs_preprocessing/{dir_name}/'

    # Paths to meshes and predictions
    right_joints_files = sorted(glob.glob(os.path.join(ref_root, '*_joints_right_1.xyz')))
    left_joints_files = sorted(glob.glob(os.path.join(ref_root, '*_joints_left_0.xyz')))

    # ---- Right Hand ----
    errors_right, c_right = 0, 0
    for i, right_joints_file in enumerate(right_joints_files):

        frame_id = int(os.path.basename(right_joints_file).split('_')[0])
        pred_file = os.path.join(our_predictions_root, f'{frame_id}_1.xyz')

        gt_kps = load_kps(right_joints_file, remap=False)
        gt_kps = center_kps(gt_kps)

        pred_kps = load_kps(pred_file)
        pred_kps = center_kps(pred_kps)

        error = compare_kps(gt_kps, pred_kps)
        if error is not None:
            errors_right += (error * 1000)
            c_right += 1

    mpjpe_right = errors_right / max(c_right, 1)
    print(f"Total Right hand MPJPE: {mpjpe_right:.2f} mm")

    # ---- Left Hand ----
    errors_left, c_left = 0, 0
    for i, left_joints_file in enumerate(left_joints_files):
        frame_id = int(os.path.basename(left_joints_file).split('_')[0])
        pred_file = os.path.join(our_predictions_root, f'{frame_id}_0.xyz')

        gt_kps = load_kps(left_joints_file, remap=False)
        gt_kps = center_kps(gt_kps)

        pred_kps = load_kps(pred_file)
        pred_kps = center_kps(pred_kps)

        error = compare_kps(gt_kps, pred_kps)
        if error is not None:
            errors_left += (error * 1000)
            c_left += 1

    mpjpe_left = errors_left / max(c_left, 1)
    print(f"Total Left hand MPJPE: {mpjpe_left:.2f} mm")

    # ---- Save Final Results ----
    final_out_path = f'../data/{seq_name}/gs_preprocessing/mpjpe_errors.txt'
    with open(final_out_path, 'w') as f:
        f.write(f"MPJPE Right: {mpjpe_right:.2f} mm\n")
        f.write(f"MPJPE Left:  {mpjpe_left:.2f} mm\n")

    print(f"Saved error summary to {final_out_path}")

    return mpjpe_right, mpjpe_left, c_right, c_left

def remove_jitter(missing_frames, found_frames, translations, mano_orient, mano_pose, mano_shape, 
                  pose_thresh=1.0, orient_thresh=1.0, transl_thresh=2.0, shape_thresh=4.0):
    """
    Remove frames where pose, orientation, translation, or shape parameters
    deviate too much from local neighbors or global statistics.
    """

    indices_to_remove = []
    frame_numbers_to_remove = []

    # Convert to arrays
    translations = np.array(translations)
    mano_orient = np.array(mano_orient)
    mano_pose   = np.array(mano_pose)
    mano_shape  = np.array(mano_shape)

    # Global stats for shape (should not change much at all)
    shape_median = np.median(mano_shape, axis=0)
    shape_std = np.std(mano_shape, axis=0) + 1e-6

    for i in range(len(found_frames)):
        if i > 0 and i < len(found_frames) - 1:
            # --- Pose jitter check ---
            diff_prev = np.linalg.norm(mano_pose[i] - mano_pose[i-1])
            diff_next = np.linalg.norm(mano_pose[i] - mano_pose[i+1])
            pose_outlier = (diff_prev > pose_thresh and diff_next > pose_thresh)
            pose_outlier = pose_outlier or ((diff_prev > pose_thresh * 2 and i-1 not in indices_to_remove) or diff_next > pose_thresh * 2)

            # --- Orientation jump ---
            orient_diff_prev = np.linalg.norm(mano_orient[i] - mano_orient[i-1])
            orient_diff_next = np.linalg.norm(mano_orient[i] - mano_orient[i+1])
            orient_outlier = (orient_diff_prev > orient_thresh and orient_diff_next > orient_thresh) 
            orient_outlier = orient_outlier or (orient_diff_prev > orient_thresh * 2 and i-1 not in indices_to_remove) or orient_diff_next > orient_thresh * 2

            # --- Translation jitter (relative spike, x–y only) ---
            transl_prev_xy = np.linalg.norm(translations[i, :2] - translations[i-1, :2])
            transl_next_xy = np.linalg.norm(translations[i, :2] - translations[i+1, :2])
            transl_outlier = (transl_prev_xy > transl_thresh and transl_next_xy > transl_thresh) 

        else:
            pose_outlier, orient_outlier, transl_outlier = False, False, False

        # --- Shape deviation ---
        shape_z = np.abs((mano_shape[i] - shape_median) / shape_std)
        shape_outlier = np.any(shape_z > shape_thresh)

        # Decide removal
        if pose_outlier or orient_outlier or transl_outlier or shape_outlier:
            indices_to_remove.append(i)
            frame_numbers_to_remove.append(found_frames[i])
            print(f"Removing frame {found_frames[i]} "
                f"(pose={pose_outlier}, orient={orient_outlier}, "
                f"transl={transl_outlier}, shape={shape_outlier})")


    print(f"Removing {len(frame_numbers_to_remove)} jitter frames.")

    # Update lists
    updated_found_frames = [f for f in found_frames if f not in frame_numbers_to_remove]
    updated_missing_frames = sorted(missing_frames + frame_numbers_to_remove)

    # Filter arrays
    translations = np.delete(translations, indices_to_remove, axis=0)
    mano_orient  = np.delete(mano_orient, indices_to_remove, axis=0)
    mano_pose    = np.delete(mano_pose, indices_to_remove, axis=0)
    mano_shape   = np.delete(mano_shape, indices_to_remove, axis=0)

    return updated_missing_frames, updated_found_frames, translations, mano_orient, mano_pose, mano_shape

def postprocess_sequence(missing_frames, found_frames, translations, mano_orient, mano_pose, mano_shape, model, renderer, pred_mano, is_right=1, out_folder=''):
    # interpolate missing frames

    mano_orient = np.array(mano_orient)
    mano_pose = np.array(mano_pose)
    mano_shape = np.array(mano_shape)
    translations = np.array(translations)

    if len(missing_frames) > 0:
        start_time = found_frames[0]
        end_time = found_frames[-1]
        start_time_query = missing_frames[0]
        end_time_query = missing_frames[-1]

        if start_time_query < start_time:
            start_orient = mano_orient[:1]
            start_pose = mano_pose[:1]
            start_shape = mano_shape[:1]
            start_transl = translations[:1]

            mano_orient = np.concatenate((start_orient, mano_orient), axis=0)
            mano_pose = np.concatenate((start_pose, mano_pose), axis=0)
            mano_shape = np.concatenate((start_shape, mano_shape), axis=0)

            translations = np.concatenate((start_transl, translations), axis=0)
            found_frames = np.concatenate(([start_time_query], found_frames), axis=0)

        if end_time < end_time_query:
            end_orient = mano_orient[-1:]
            end_pose = mano_pose[-1:]
            end_shape = mano_shape[-1:]
            end_transl = translations[-1:]

            mano_orient = np.concatenate((mano_orient, end_orient), axis=0)
            mano_pose = np.concatenate((mano_pose, end_pose), axis=0)
            mano_shape = np.concatenate((mano_shape, end_shape), axis=0)
            translations = np.concatenate((translations, end_transl), axis=0)
            found_frames = np.concatenate((found_frames, [end_time_query]), axis=0)

        mano_orient_matrix = R.from_matrix(mano_orient)
        slerp = Slerp(found_frames, mano_orient_matrix)
        interp_orient = slerp(missing_frames).as_matrix()

        interp_angles = []
        for i in range(mano_pose.shape[1]):
            mano_pose_matrix = R.from_matrix(mano_pose[:, i, :])
            slerp = Slerp(found_frames, mano_pose_matrix)
            interp_pose = slerp(missing_frames).as_matrix()
            interp_angles.append(interp_pose)
        
        interp_pose = np.array(interp_angles).transpose(1, 0, 2, 3)

        interp_transl_x = np.interp(missing_frames, found_frames, translations[:, 0])
        interp_transl_y = np.interp(missing_frames, found_frames, translations[:, 1])
        interp_transl_z = np.interp(missing_frames, found_frames, translations[:, 2])
        interp_transl = np.vstack([interp_transl_x, interp_transl_y, interp_transl_z]).T

        interp_shapes = []
        for i in range(mano_shape.shape[1]):
            interp_shape = np.interp(missing_frames, found_frames, mano_shape[:, i])
            interp_shapes.append(interp_shape)
        interp_shapes = np.array(interp_shapes).T

        missing_pred_mano_params = {}
        # print(torch.tensor(interp_orient).to(model.device).unsqueeze(1).shape)
        missing_pred_mano_params['global_orient'] = torch.tensor(interp_orient).to(model.device).unsqueeze(1)
        missing_pred_mano_params['hand_pose'] = torch.tensor(interp_pose).to(model.device)
        missing_pred_mano_params['betas'] = torch.tensor(interp_shapes).to(model.device)

        pred_orient_axis = matrix_to_axis_angle(missing_pred_mano_params['global_orient'])
        pred_mano_pose_axis = matrix_to_axis_angle(missing_pred_mano_params['hand_pose'])
        pred_mano_shape = missing_pred_mano_params['betas']
        pred_mano_pose_axis = torch.cat([pred_orient_axis, pred_mano_pose_axis], dim=1).view(-1, 48)
        # print('missing_pred_mano_params', pred_mano_pose_axis.shape, pred_orient_axis.shape, pred_mano_shape.shape)

        mano_output = model.mano(**{k: v.float() for k,v in missing_pred_mano_params.items()}, pose2rot=False)
        
        verts = mano_output.vertices.detach().cpu().numpy()
        joints = mano_output.joints.detach().cpu().numpy()

        if not is_right:
            verts[:, :, 0] *= -1
            joints[:, :, 0] *= -1

        # pred mano meshes
        for i, frame_num in tqdm(enumerate(missing_frames)):
            # save the interpolated meshes
            camera_translation = interp_transl[i]
            joints[i] += camera_translation
            # print(camera_translation.shape)
            np.savetxt(os.path.join(out_folder, f'hand_joints/{frame_num}_{is_right}.xyz'), joints[i])
            
            joints_copy = joints[i].copy()
            joints_copy[:, :2] *= -1
            joints2d = project_3D_points(renderer.cam_int, joints_copy.reshape(1, -1, 3).copy())
            np.savetxt(os.path.join(out_folder, f'hand_joints2d/{frame_num}_{is_right}.xyz'), joints2d[0])
            # print('inside loop', verts[i].shape)
            tmesh = renderer.vertices_to_trimesh(verts[i], camera_translation, LIGHT_BLUE, is_right=is_right)
            tmesh.vertices[:, :2] *= -1
            # print('tmesh', tmesh.vertices.shape)
            tmesh.export(os.path.join(out_folder, f'hand_meshes/{frame_num}_{is_right}.obj'))

            pred_mano[frame_num] = (pred_mano_pose_axis[i].cpu(), pred_mano_shape[i].detach().cpu())

    # covnert the dict into a tensor for pose and tensor for betas but keep in mind 
    # that the frames that were added at the end should be in the right relative order
    # sort the values based on sorted keys
    pred_mano = dict(sorted(pred_mano.items()))
    # save the values as a tensor
    pose_params = torch.stack([v[0] for v in pred_mano.values()])
    shape_params = torch.stack([v[1] for v in pred_mano.values()])
    # print(pose_params.shape, shape_params.shape)
    is_right_str = 'right' if is_right else 'left'
    torch.save(pose_params, os.path.join(out_folder, f'pose_params_{is_right_str}.pt'))
    torch.save(shape_params, os.path.join(out_folder, f'shape_params_{is_right_str}.pt'))

def rerender_frames(missing_frames, out_folder, renderer, img_folder):

    for frame_num in missing_frames:
        all_verts, all_right, all_cam_t = [], [], []
        # zfill 4
        frame_num_str = str(frame_num).zfill(4)
        # print(frame_num_str)
        img_cv2 = cv2.imread(os.path.join(img_folder, f'{frame_num_str}.png'))
        for is_right in [0, 1]:
            obj_file = os.path.join(out_folder, f'hand_meshes/{frame_num}_{is_right}.obj')
            if os.path.exists(obj_file):
                verts = trimesh.load_mesh(obj_file).vertices
                # print(verts.shape)
                all_verts.append(verts)
                all_right.append(is_right)
                all_cam_t.append(np.zeros(3))

        # print(frame_num, len(all_verts), all_right, all_verts[0].shape)
        cam_view = renderer.fast_render_rgb_frame_pytorch3d(all_verts, cam_t=all_cam_t, is_right=all_right)
        # convert it to a segmentation mask
        visibility_mask = cam_view[:, :, 3] > 0.5
        # save the segmentation mask
        cv2.imwrite(os.path.join(out_folder, f'hand_masks/{frame_num_str}.jpg.png'), (255 * visibility_mask).astype(np.uint8))
        # Overlay image
        input_img = img_cv2.astype(np.float32)[:,:,::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :,: 1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:, :, :3] * (1-cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        # print fps on image
        output_img = 255 * input_img_overlay[:, :, ::-1]
        # flip image
        output_img = output_img.astype(np.uint8)
        cv2.imwrite(os.path.join(out_folder, f'hand_rendered_new/{frame_num_str}.jpg'), output_img)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area + 1e-6

    return inter_area / union


def visualize_detections(bboxes, is_right, kps, confs, img_cv2, frame_num_str, args):
    img_kps_copy = img_cv2.copy()
    os.makedirs(os.path.join(args.out_folder, 'hand_detections'), exist_ok=True)

    for bbox, right, kp, c in zip(bboxes, is_right, kps, confs):
        x1, y1, x2, y2 = map(int, bbox)
        label = 'R' if right else 'L'
        cv2.rectangle(img_kps_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_kps_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # Put area of bb in k px
        area = (x2 - x1) * (y2 - y1) / 1000
        cv2.putText(img_kps_copy, f"{int(area)}k px", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        # print average confidence
        conf_mean = np.mean(c)
        cv2.putText(img_kps_copy, f"{conf_mean:.2f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        for ki, ((x, y), conf) in enumerate(zip(kp, c)):
            cv2.circle(img_kps_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(img_kps_copy, f"{conf:.2f}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            # cv2.putText(img_kps_copy, f"{ki}", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imwrite(os.path.join(args.out_folder, f'hand_detections/{frame_num_str}.png'), img_kps_copy)


