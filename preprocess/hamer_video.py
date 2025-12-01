import torch
import argparse
import os
import cv2
import numpy as np

np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.str = np.str_
np.unicode = np.unicode_
np.object = np.object_
np.complex = np.complex_
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from pathlib import Path
from utils.demo_detector_with_tracking import create_detector
from utils.pytorch3d_renderer import MeshPyTorch3DRenderer, project_3D_points
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import cam_crop_to_full
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_axis_angle
from manopth.manolayer import ManoLayer

from utils.hamer_utils import filter_hands, compare_with_gt, compare_with_hold, iou, remove_jitter, postprocess_sequence, rerender_frames, visualize_detections
import numpy as np

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='../data/HO3D_v3/train/MC1/rgb/', help='Root folder for frames')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--max_batch_size', type=int, default=8, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--model', type=str, default='hamer', choices=['hamer', 'wilor'], help='Which model to use: hamer or wilor')
    parser.add_argument('--visualize', action='store_true', default=False, help='If set, visualize the fitting results')

    # Detector parameters
    parser.add_argument('--square', action='store_true', default=False, help='If set, switch rectangular bbox to square')
    parser.add_argument('--bb_pad', type=float, default=0.0, help='Padding to add to bbox')
    parser.add_argument('--bb_conf', type=float, default=0.5, help='Min confidence for bbox detection')
    parser.add_argument('--bb_area', type=float, default=0.002, help='Min area for bbox detection as fraction of image area')
    parser.add_argument('--bb_iou', type=float, default=0.4, help='Max IOU for bbox detection')

    # Jitter params
    parser.add_argument('--pose_thresh', type=float, default=1.0, help='Max change in pose to consider jitter')
    parser.add_argument('--orient_thresh', type=float, default=1.0, help='Max change in global orientation to consider jitter')
    parser.add_argument('--transl_thresh', type=float, default=2.0, help='Max change in translation to consider jitter')
    parser.add_argument('--shape_thresh', type=float, default=4.0, help='Max change in shape to consider jitter')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mano_layer_right = ManoLayer(mano_root='./_DATA/data/mano/', use_pca=False, flat_hand_mean=True).to(device)
    seq_name = args.img_folder.split('/')[-4]

    print(f"Model loaded on device: {device}")
    renderer = None

    # Load detector
    detector = create_detector()
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    model = model.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(os.path.join(args.out_folder, 'hand_meshes'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'hand_joints'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'hand_joints2d'), exist_ok=True)

    if args.visualize:
        os.makedirs(os.path.join(args.out_folder, 'hand_detections'), exist_ok=True)
        os.makedirs(os.path.join(args.out_folder, 'hand_rendered'), exist_ok=True)


    missing_right_frames, missing_left_frames = [], []
    found_right_frames, found_left_frames = [], []
    pred_right_orient, pred_right_pose, pred_right_shape, pred_right_transl = [], [], [], []
    pred_left_orient, pred_left_pose, pred_left_shape, pred_left_transl = [], [], [], []
    pred_left_mano, pred_right_mano = {}, {}

    min_frame_num = min(
        [int(os.path.splitext(p)[0]) for p in os.listdir(args.img_folder) if os.path.splitext(p)[-1] in [".png", ".PNG"]]
    )

    confidence_dict = {}

    for img_path in tqdm(sorted(Path(args.img_folder).rglob('*.png'))):
        frame_num_str = img_path.stem
        frame_num = int(frame_num_str) - min_frame_num
        img_cv2 = cv2.imread(str(img_path))

        
        image_area = img_cv2.shape[0] * img_cv2.shape[1]
        min_area = int(args.bb_area * image_area)
        max_area = int(0.2 * image_area)
        bboxes, is_right, kps, confs = detector(img_cv2, square=args.square, pad_factor=args.bb_pad)

        # plot keypoints and bounding box with conf on each keypoint and the average on the bounding box along with hand label right or left
        if args.visualize:
            visualize_detections(bboxes, is_right, kps, confs, img_cv2, frame_num_str, args)

        bboxes, is_right, kps, confs, conf_dict = filter_hands(bboxes, is_right, kps, confs, conf_thresh=args.bb_conf, min_area=min_area, max_area=max_area)
        # find index of right and left hands
        confidence_dict[frame_num] = conf_dict

        # filter hands
        if 1 in is_right and 0 in is_right:
            iou_val = iou(bboxes[0], bboxes[1])
            if iou_val > args.bb_iou:
                print(f" - IOU: {iou_val:.2f}")
                # remove the one with lower confidence
                if np.mean(confs[0]) > np.mean(confs[1]):
                    bboxes, is_right, kps, confs = [bboxes[0]], [is_right[0]], [kps[0]], [confs[0]]
                else:
                    bboxes, is_right, kps, confs = [bboxes[1]], [is_right[1]], [kps[1]], [confs[1]]

        if 1 not in is_right:
            missing_right_frames.append(frame_num)
        if 0 not in is_right:
            missing_left_frames.append(frame_num)

        if len(bboxes) != 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.max_batch_size, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []

            batch = next(iter(dataloader))            
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            if renderer is None:
                renderer = MeshPyTorch3DRenderer(model_cfg, model.mano.faces, device, render_res=img_size[0], focal_length=scaled_focal_length)

            # Render the result
            batch_size = batch['img'].shape[0]

            for n in range(batch_size):
                # Get filename from path img_path
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()
                pred_orient = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy()
                pred_orient_axis = matrix_to_axis_angle(out['pred_mano_params']['global_orient'][n].detach())
                pred_mano_pose = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy()
                pred_mano_pose_axis = matrix_to_axis_angle(out['pred_mano_params']['hand_pose'][n].detach())
                pred_mano_pose_axis = torch.cat([pred_orient_axis, pred_mano_pose_axis], dim=0).view(1, -1)
                pred_betas = out['pred_mano_params']['betas'][n].detach().cpu().numpy()

                is_right = batch['right'][n].cpu().numpy()

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0] 
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]
                
                if is_right:
                    pred_right_mano[frame_num] = (pred_mano_pose_axis[0].cpu(), out['pred_mano_params']['betas'][n].detach().cpu())
                else:
                    pred_left_mano[frame_num] = (pred_mano_pose_axis[0].cpu(), out['pred_mano_params']['betas'][n].detach().cpu())

                verts_mano, _ = mano_layer_right(pred_mano_pose_axis, out['pred_mano_params']['betas'][n].detach().unsqueeze(0))
                verts_mano[:, :, 0] = ((2 * is_right - 1) * verts_mano[:, :, 0])
                verts_mano = verts_mano[0] / 1000
                verts_mano += torch.tensor(cam_t).to(device)

                # Compute 2D joints
                camera_translation = cam_t.copy()
                joints += camera_translation
                joints_copy = joints.copy()
                joints_copy[:, :2] *= -1
                joints2d = project_3D_points(renderer.cam_int, joints_copy.reshape(1, -1, 3).copy())

                # Compute bounding box size based on 2D keypoints
                x_min, y_min = np.min(joints2d[0], axis=0)
                x_max, y_max = np.max(joints2d[0], axis=0)
                bbox_width, bbox_height = x_max - x_min, y_max - y_min
                area = bbox_width * bbox_height

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                if is_right:
                    pred_right_orient.append(pred_orient[0])
                    pred_right_pose.append(pred_mano_pose)
                    pred_right_shape.append(pred_betas)
                    pred_right_transl.append(cam_t)
                    found_right_frames.append(frame_num)
                else:
                    pred_left_orient.append(pred_orient[0])
                    pred_left_pose.append(pred_mano_pose)
                    pred_left_shape.append(pred_betas)
                    pred_left_transl.append(cam_t)
                    found_left_frames.append(frame_num)

                # # Save all meshes to disk
                if args.save_mesh:

                    np.savetxt(os.path.join(args.out_folder, f'hand_joints2d/{frame_num}_{int(is_right)}.xyz'), joints2d[0])
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.vertices[:, :2] *= -1
                    tmesh.export(os.path.join(args.out_folder, f'hand_meshes/{frame_num}_{int(is_right)}.obj'))

                    # save in .xyz format
                    np.savetxt(os.path.join(args.out_folder, f'hand_joints/{frame_num}_{int(is_right)}.xyz'), joints)

            # Render front view
            if len(all_verts) > 0 and args.visualize:
                cam_view = renderer.fast_render_rgb_frame_pytorch3d(all_verts, cam_t=all_cam_t, is_right=all_right)
                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1] / 255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                # print fps on image
                output_img = 255*input_img_overlay[:, :, ::-1]
                # flip image
                output_img = output_img.astype(np.uint8)
            else:
                output_img = img_cv2
        else:
            output_img = img_cv2

        if args.visualize:
            cv2.imwrite(os.path.join(args.out_folder, f'hand_rendered/{frame_num_str}.jpg'), output_img)
    
    # Save confidence dict using numpy
    np.save(os.path.join(args.out_folder, 'detection_confidences.npy'), confidence_dict)

    combined_missing_frames = []
    missing_right_frames, found_right_frames, pred_right_transl, pred_right_orient, pred_right_pose, pred_right_shape = \
        remove_jitter(missing_right_frames, found_right_frames, pred_right_transl, pred_right_orient, pred_right_pose, pred_right_shape, 
                    pose_thresh=args.pose_thresh, orient_thresh=args.orient_thresh, transl_thresh=args.transl_thresh, shape_thresh=args.shape_thresh)
    
    missing_left_frames, found_left_frames, pred_left_transl, pred_left_orient, pred_left_pose, pred_left_shape = \
        remove_jitter(missing_left_frames, found_left_frames, pred_left_transl, pred_left_orient, pred_left_pose, pred_left_shape,
                    pose_thresh=args.pose_thresh, orient_thresh=args.orient_thresh, transl_thresh=args.transl_thresh, shape_thresh=args.shape_thresh)

    if len(found_right_frames) > 0 and len(missing_right_frames) > 0:
        print(f"{seq_name}", f"Missing right frames: {missing_right_frames}")
        postprocess_sequence(missing_right_frames, found_right_frames, pred_right_transl, pred_right_orient, pred_right_pose, pred_right_shape, 
                            model, renderer, pred_right_mano, is_right=1, out_folder=args.out_folder)
        combined_missing_frames.extend(missing_right_frames)
    elif len(found_right_frames) > 0:
        pred_mano = dict(sorted(pred_right_mano.items()))
        # save the values as a tensor
        pose_params = torch.stack([v[0] for v in pred_mano.values()])
        shape_params = torch.stack([v[1] for v in pred_mano.values()])
        torch.save(pose_params, os.path.join(args.out_folder, f'pose_params_right.pt'))
        torch.save(shape_params, os.path.join(args.out_folder, f'shape_params_right.pt'))

    if len(found_left_frames) > 0 and len(missing_left_frames) > 0:
        print(f"{seq_name}", f"Missing left frames: {missing_left_frames}")
        postprocess_sequence(missing_left_frames, found_left_frames, pred_left_transl, pred_left_orient, pred_left_pose, pred_left_shape, 
                            model, renderer, pred_left_mano, is_right=0, out_folder=args.out_folder)
        combined_missing_frames.extend(missing_left_frames)
    elif len(found_left_frames) > 0:
        pred_mano = dict(sorted(pred_left_mano.items()))
        # save the values as a tensor
        pose_params = torch.stack([v[0] for v in pred_mano.values()])
        shape_params = torch.stack([v[1] for v in pred_mano.values()])
        # print(pose_params.shape, shape_params.shape)
        torch.save(pose_params, os.path.join(args.out_folder, f'pose_params_left.pt'))
        torch.save(shape_params, os.path.join(args.out_folder, f'shape_params_left.pt'))
        
    if len(combined_missing_frames) > 0:
        if args.visualize:
            rerender_frames(combined_missing_frames, args.out_folder, renderer, args.img_folder)

    # if 'ho3d' in seq_name:
    #     error_right, error_left, count_right, count_left = compare_with_gt(seq_name)
    # if 'arctic' in seq_name:
    #     error_right, error_left, count_right, count_left = compare_with_hold(seq_name)

    # print(error_right, error_left, count_right, count_left)
    # # save in a common .csv file with parameter configurations
    # final_out_path = 'hamer_error_summary.csv'
    # # create the files if it does not exist
    # if not os.path.exists(final_out_path):
    #     with open(final_out_path, 'w') as f:
    #         f.write('seq_name,square,bb_pad,bb_conf,bb_area,bb_iou,pose_thresh,orient_thresh,transl_thresh,shape_thresh,error_right,error_left,count_right,count_left\n')
    
    # # append to the file
    # with open(final_out_path, 'a') as f:
    #     f.write(f"{seq_name},{args.square},{args.bb_pad},{args.bb_conf},{args.bb_area},{args.bb_iou},{args.pose_thresh},{args.orient_thresh},{args.transl_thresh},{args.shape_thresh},{error_right},{error_left},{count_right},{count_left}\n")


if __name__ == '__main__':
    print('Starting HaMeR demo...')
    main()

