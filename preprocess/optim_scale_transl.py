import inspect
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from manopth.manolayer import ManoLayer
from pytorch3d.io import load_obj, save_obj, save_ply

from utils.colmap_readmodel import (
    Image,
    Point3D,
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from utils.optim_utils import (
    compute_loss,
    load_and_compute_grasp,
    project_points,
    visualize_grasping,
)

preprocess_dir = "ghost_build"

# Compatibility shims for deprecated numpy aliases used by downstream libs
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.str = np.str_
np.unicode = np.unicode_
np.object = np.object_
np.complex = np.complex_


@dataclass
class Paths:
    base: str
    colmap: str
    hand: str
    vis: str
    sfm_output: str
    aligned_hand: str
    aligned_mesh: str


@dataclass
class PriorMeshes:
    verts: Optional[torch.Tensor]
    faces: Optional[torch.Tensor]
    orig_verts: Optional[torch.Tensor] = None
    orig_faces: Optional[torch.Tensor] = None


@dataclass
class ManoParameters:
    pose: torch.nn.Parameter
    rot: torch.nn.Parameter
    shape: torch.nn.Parameter
    pose_left: Optional[torch.nn.Parameter]
    rot_left: Optional[torch.nn.Parameter]


@dataclass
class HandSequenceData:
    translations: List[torch.Tensor]
    left_translations: List[torch.Tensor]
    hand_proj_targets: List[torch.Tensor]
    hand_proj_targets_left: List[torch.Tensor]
    hand_verts_seq: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]
    hand_verts_seq_left: List[torch.Tensor]
    frames: List[Tuple[int, Image, torch.Tensor]]
    is_grasping_flags: List[int]
    hand_masks: List[np.ndarray]
    obj_masks: List[np.ndarray]
    img_dim: Tuple[int, int]


@dataclass
class OptimizationResult:
    scale: torch.Tensor
    right_translations: torch.Tensor
    left_translations: Optional[torch.Tensor]
    loss_dict_avg: Dict[str, float]
    dists_right: List[float]
    dists_left: List[float]


def build_paths(seq_name: str) -> Paths:
    base_path = f"../data/{seq_name}"
    hand_path = os.path.join(base_path, f"{preprocess_dir}/")
    return Paths(
        base=base_path,
        colmap=os.path.join(base_path, f"{preprocess_dir}/sfm"),
        hand=hand_path,
        vis=os.path.join(base_path, f"{preprocess_dir}/alignment_visuals"),
        sfm_output=os.path.join(base_path, f"{preprocess_dir}/sfm_rescaled"),
        aligned_hand=os.path.join(hand_path, "hand_joints_aligned"),
        aligned_mesh=os.path.join(hand_path, "hand_meshes_aligned"),
    )


def load_detection_confidences(paths: Paths) -> Optional[Dict[int, Dict[str, list]]]:
    conf_path = os.path.join(paths.base, f"{preprocess_dir}/detection_confidences.npy")
    if not os.path.exists(conf_path):
        return None

    detection_confidences = np.load(conf_path, allow_pickle=True)
    detection_confidences = {int(k): v for k, v in detection_confidences.item().items()}
    return {k: {side: v[side][0] if len(v[side]) > 0 else [] for side in ["left", "right"]} for k, v in detection_confidences.items()}


def ensure_directories(paths: Paths, visualize: bool) -> None:
    os.makedirs(paths.sfm_output, exist_ok=True)
    os.makedirs(paths.aligned_hand, exist_ok=True)
    os.makedirs(paths.aligned_mesh, exist_ok=True)
    if visualize:
        os.makedirs(paths.vis, exist_ok=True)


def determine_optimized_list(seq_name: str) -> List[str]:
    if seq_name in ["dfki_hand_02"]:
        return ["right"]
    if "arctic" in seq_name:
        return ["left", "right", "object"]
    return ["right", "object"]


def load_camera_and_prior(paths: Paths, optimized_list: List[str], device: torch.device, load_prior: bool) -> Tuple[torch.Tensor, Optional[Dict[int, Image]], Optional[Dict[int, Point3D]], Optional[torch.Tensor], Optional[torch.Tensor], PriorMeshes]:
    if "object" not in optimized_list:
        cam_intr = torch.tensor(
            [[914.0, 0.0, 959.5], [0.0, 914.0, 539.5], [0.0, 0.0, 1.0]],
            device=device,
        )
        return cam_intr, None, None, None, None, PriorMeshes(verts=None, faces=None)

    cameras = read_cameras_binary(os.path.join(paths.colmap, "cameras.bin"))
    images = read_images_binary(os.path.join(paths.colmap, "images.bin"))
    points3D = read_points3D_binary(os.path.join(paths.colmap, "points3D.bin"))

    point_cloud = torch.tensor([p.xyz for p in points3D.values()], dtype=torch.float32, device=device)
    filtered_point_cloud = torch.tensor(
        [p.xyz for p in points3D.values() if p.image_ids.shape[0] >= 30],
        dtype=torch.float32,
        device=device,
    )
    print("Point cloud shape:", point_cloud.shape, "Filtered point cloud shape:", filtered_point_cloud.shape)

    cam_params = np.array(cameras[list(cameras.keys())[0]].params)
    if cam_params.shape[0] == 4:
        fx, fy, cx, cy = cam_params[0], cam_params[1], cam_params[2], cam_params[3]
    else:
        fx, fy, cx, cy = cam_params[0], cam_params[0], cam_params[1], cam_params[2]
    cam_intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    prior_meshes = PriorMeshes(verts=None, faces=None)
    if load_prior:
        prior_obj_path = os.path.join(paths.base, f"{preprocess_dir}/prior/best_obj.obj")
        prior_mesh = load_obj(prior_obj_path)
        prior_meshes.verts = prior_mesh[0].to(device)
        prior_meshes.faces = prior_mesh[1].verts_idx

        best_orig_obj_path = os.path.join(paths.base, f"{preprocess_dir}/prior/best_orig_obj.obj")
        if os.path.exists(best_orig_obj_path):
            orig_prior_mesh = load_obj(best_orig_obj_path)
            prior_meshes.orig_verts = orig_prior_mesh[0].to(device)
            prior_meshes.orig_faces = orig_prior_mesh[1].verts_idx
    else:
        print("Not loading prior mesh")

    return cam_intr, images, points3D, point_cloud, filtered_point_cloud, prior_meshes


def load_mano_layers(device: torch.device) -> Tuple[ManoLayer, torch.Tensor, torch.Tensor]:
    mano_right = ManoLayer(mano_root="_DATA/data/mano", use_pca=False, flat_hand_mean=True).to(device)
    mano_left = ManoLayer(mano_root="_DATA/data/mano", use_pca=False, flat_hand_mean=True, side="left").to(device)
    faces_right = mano_right.th_faces.to(device)
    faces_left = mano_left.th_faces.to(device)
    return mano_right, faces_right, faces_left


def load_mano_parameters(paths: Paths, optimized_list: List[str], device: torch.device) -> ManoParameters:
    mano_pose_hamer = torch.load(os.path.join(paths.hand, "pose_params_right.pt")).to(device).to(torch.float32)
    shape_hamer = torch.load(os.path.join(paths.hand, "shape_params_right.pt"))[:1].to(device).to(torch.float32)
    mano_pose_params = torch.nn.Parameter(mano_pose_hamer[:, 3:], requires_grad=True)
    mano_rot_params = torch.nn.Parameter(mano_pose_hamer[:, :3], requires_grad=True)
    shape_params = torch.nn.Parameter(shape_hamer, requires_grad=True)

    if "left" in optimized_list:
        mano_pose_hamer_left = torch.load(os.path.join(paths.hand, "pose_params_left.pt")).to(device).to(torch.float32)
        mano_pose_left_params = torch.nn.Parameter(mano_pose_hamer_left[:, 3:], requires_grad=True)
        mano_rot_left_params = torch.nn.Parameter(mano_pose_hamer_left[:, :3], requires_grad=True)
    else:
        mano_pose_left_params = None
        mano_rot_left_params = None

    return ManoParameters(
        pose=mano_pose_params,
        rot=mano_rot_params,
        shape=shape_params,
        pose_left=mano_pose_left_params,
        rot_left=mano_rot_left_params,
    )


def prepare_hand_sequences(
    seq_name: str,
    paths: Paths,
    optimized_list: List[str],
    images: Optional[Dict[int, Image]],
    filtered_point_cloud: Optional[torch.Tensor],
    device: torch.device,
    mano_params: ManoParameters,
    faces_right: torch.Tensor,
    faces_left: torch.Tensor,
    visualize: bool,
) -> Tuple[HandSequenceData, float, float]:
    hand_mesh_files = sorted(
        glob(os.path.join(paths.hand, "hand_meshes", "*_1.obj")),
        key=lambda x: int(os.path.basename(x).split("_")[0]),
    )

    hand_data = HandSequenceData(
        translations=[],
        left_translations=[],
        hand_proj_targets=[],
        hand_proj_targets_left=[],
        hand_verts_seq=[],
        hand_verts_seq_left=[],
        frames=[],
        is_grasping_flags=[],
        hand_masks=[],
        obj_masks=[],
        img_dim=(0, 0),
    )

    scale_hand, scale_obj, img_dim = load_and_compute_grasp(
        seq_name,
        hand_mesh_files,
        optimized_list,
        images,
        filtered_point_cloud,
        paths.hand,
        device,
        hand_data.hand_masks,
        hand_data.obj_masks,
        hand_data.translations,
        hand_data.hand_proj_targets,
        hand_data.hand_verts_seq,
        hand_data.frames,
        hand_data.is_grasping_flags,
        hand_data.left_translations,
        hand_data.hand_proj_targets_left,
        hand_data.hand_verts_seq_left,
        mano_params.rot,
        mano_params.rot_left,
        faces_right,
        faces_left,
    )
    hand_data.img_dim = img_dim
    if visualize:
        visualize_grasping(preprocess_dir, paths.base, hand_data.is_grasping_flags)
    return hand_data, scale_hand, scale_obj


def create_scale_parameter(init_scale: float, apply_exp: bool, device: torch.device) -> torch.Tensor:
    if apply_exp:
        return torch.tensor([torch.log(torch.tensor(init_scale))], requires_grad=True, device=device, dtype=torch.float32)
    return torch.tensor([init_scale], requires_grad=True, device=device, dtype=torch.float32)


def build_optimizer(
    apply_exp: bool,
    init_scale: float,
    translations: List[torch.Tensor],
    left_translations: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, optim.Adam]:
    scale_log = create_scale_parameter(init_scale, apply_exp, device)
    params: List[torch.Tensor] = [scale_log] + translations + left_translations
    optimizer = optim.Adam(params, lr=0.05)
    return scale_log, optimizer


def compute_mano_outputs(
    mano_layer: ManoLayer, rot_params: torch.Tensor, pose_params: torch.Tensor, shape_params: torch.Tensor, translations: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    verts, joints = mano_layer(torch.cat((rot_params, pose_params), dim=1), shape_params.repeat(pose_params.shape[0], 1), translations)
    return verts / 1000, joints / 1000


def optimize_scale_and_translation(
    seq_name: str,
    optimized_list: List[str],
    apply_exp: bool,
    cam_intr: torch.Tensor,
    point_cloud: Optional[torch.Tensor],
    filtered_point_cloud: Optional[torch.Tensor],
    prior_meshes: PriorMeshes,
    mano_right: ManoLayer,
    faces_right: torch.Tensor,
    faces_left: torch.Tensor,
    mano_params: ManoParameters,
    hand_data: HandSequenceData,
    detection_confidences: Optional[Dict[int, Dict[str, list]]],
    scale_log: torch.Tensor,
    optimizer: optim.Optimizer,
    num_iters: int = 500,
) -> OptimizationResult:
    best_loss = float("inf")
    last_iter = 0
    best_scale = torch.exp(scale_log).detach().clone() if apply_exp else scale_log.detach().clone()
    best_right_transl = torch.cat(hand_data.translations, dim=0).detach().clone()
    best_left_transl = torch.cat(hand_data.left_translations, dim=0).detach().clone() if hand_data.left_translations else None
    dists_right: List[float] = []
    dists_left: List[float] = []
    loss_dict_avg: Dict[str, float] = {}

    for iteration in tqdm(range(num_iters)):
        optimizer.zero_grad()
        total_loss = 0.0
        loss_dict_avg = {}
        scale = torch.exp(scale_log) if apply_exp else scale_log

        all_translations = torch.cat(hand_data.translations, dim=0)
        all_left_translations = torch.cat(hand_data.left_translations, dim=0) if hand_data.left_translations else None

        verts_transl, joints_transl = compute_mano_outputs(mano_right, mano_params.rot, mano_params.pose, mano_params.shape, all_translations)

        if "left" in optimized_list and all_left_translations is not None and mano_params.pose_left is not None and mano_params.rot_left is not None:
            verts_transl_left, joints_transl_left = compute_mano_outputs(
                mano_right, mano_params.rot_left, mano_params.pose_left, mano_params.shape, all_left_translations
            )
            verts_transl_left[:, :, 0] = -verts_transl_left[:, :, 0]
            joints_transl_left[:, :, 0] = -joints_transl_left[:, :, 0]
            joints_combined = torch.cat((joints_transl, joints_transl_left), dim=1)
        else:
            verts_transl_left = None
            joints_transl_left = None
            joints_combined = joints_transl

        for idx, (_, _, faces, _, hand_id) in enumerate(hand_data.hand_verts_seq):
            frame_id, img_data, _ = hand_data.frames[idx]
            if "object" in optimized_list and filtered_point_cloud is not None and point_cloud is not None:
                rot = torch.tensor(qvec2rotmat(img_data.qvec), dtype=torch.float32, device=cam_intr.device)
                transl = torch.tensor(img_data.tvec, dtype=torch.float32, device=cam_intr.device) * scale
                scaled_point_cloud = torch.matmul(rot, filtered_point_cloud.T * scale).T + transl
                scaled_prior = torch.matmul(rot, prior_meshes.verts.T * scale).T + transl if prior_meshes.verts is not None else None
            else:
                scaled_point_cloud = None
                scaled_prior = None

            hand_mask_gt = hand_data.hand_masks[idx] / 255.0
            proj_2d = project_points(joints_combined[idx], cam_intr)
            grasping_flag = hand_data.is_grasping_flags[idx]
            is_grasping = (grasping_flag == 1 and hand_id == 1) or (grasping_flag == 2 and hand_id == 0)

            if is_grasping:
                active_verts = verts_transl_left[idx] if grasping_flag == 2 and verts_transl_left is not None else verts_transl[idx]
                dist_right = torch.cdist(verts_transl[idx], scaled_point_cloud) if scaled_point_cloud is not None else None
                min_dist_right = torch.min(dist_right) if dist_right is not None else torch.tensor(0.0, device=cam_intr.device)
                dists_right.append(min_dist_right.item())

                if verts_transl_left is not None:
                    dist_left = torch.cdist(verts_transl_left[idx], scaled_point_cloud) if scaled_point_cloud is not None else None
                    min_dist_left = torch.min(dist_left) if dist_left is not None else torch.tensor(0.0, device=cam_intr.device)
                    dists_left.append(min_dist_left.item())
            else:
                active_verts = torch.zeros_like(verts_transl[idx])

            if "left" in optimized_list and all_left_translations is not None and hand_data.hand_proj_targets_left:
                targets_2d = torch.cat((hand_data.hand_proj_targets[idx], hand_data.hand_proj_targets_left[idx]), dim=0)
                combined_translations = torch.cat((all_translations, all_left_translations), dim=1)
            else:
                targets_2d = hand_data.hand_proj_targets[idx]
                combined_translations = all_translations

            frame_conf = detection_confidences[frame_id] if detection_confidences is not None else None
            bb_box_thresh = 20000 if "arctic" in seq_name else 2000
            loss, loss_dict = compute_loss(
                proj_2d,
                targets_2d,
                active_verts,
                scaled_point_cloud,
                combined_translations,
                is_grasping,
                None,
                hand_mask_gt,
                scaled_prior,
                frame_conf,
                bb_box_thresh=bb_box_thresh,
            )

            loss_dict_avg = {k: loss_dict_avg.get(k, 0) + loss_dict[k] for k in loss_dict}
            total_loss += loss

        total_loss /= len(hand_data.hand_verts_seq)

        if total_loss < best_loss:
            last_iter = iteration
            best_loss = total_loss
            best_scale = scale.detach().clone()
            best_right_transl = all_translations.detach().clone()
            if "left" in optimized_list and all_left_translations is not None:
                best_left_transl = all_left_translations.detach().clone()

        if iteration - last_iter > 50:
            break

        for k, v in loss_dict_avg.items():
            loss_dict_avg[k] = v / len(hand_data.hand_verts_seq)

        total_loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            mean_transl = torch.mean(torch.cat(hand_data.translations, dim=0), dim=0).detach().cpu().numpy()
            print(f"Iter {iteration}: Loss = {total_loss.item():.4f}, Scale = {scale.item():.4f} Avg Translations = {mean_transl}")
            print("Losses:", loss_dict_avg)

    return OptimizationResult(
        scale=best_scale,
        right_translations=best_right_transl,
        left_translations=best_left_transl,
        loss_dict_avg=loss_dict_avg,
        dists_right=dists_right,
        dists_left=dists_left,
    )


def export_sfm_results(
    paths: Paths,
    optimized_list: List[str],
    frames: List[Tuple[int, Image, torch.Tensor]],
    images: Optional[Dict[int, Image]],
    points3D: Optional[Dict[int, Point3D]],
    point_cloud: Optional[torch.Tensor],
    prior_meshes: PriorMeshes,
    scale: torch.Tensor,
    is_grasping_flags: List[int],
) -> None:
    if "object" not in optimized_list or images is None or points3D is None or point_cloud is None:
        return

    scale_value = scale.item()
    new_images = {}
    for _, img_data, _ in frames:
        new_tvec = img_data.tvec * scale_value
        new_images[img_data.id] = Image(
            id=img_data.id,
            qvec=img_data.qvec,
            tvec=new_tvec,
            camera_id=img_data.camera_id,
            name=img_data.name,
            xys=img_data.xys,
            point3D_ids=img_data.point3D_ids,
        )
    write_images_binary(new_images, os.path.join(paths.sfm_output, "images.bin"))
    write_cameras_binary(read_cameras_binary(os.path.join(paths.colmap, "cameras.bin")), os.path.join(paths.sfm_output, "cameras.bin"))

    new_points3D = {}
    for i, k in enumerate(points3D.keys()):
        new_points3D[k] = Point3D(
            id=k,
            xyz=(point_cloud[i] * scale_value).detach().cpu().numpy(),
            rgb=points3D[k].rgb,
            error=points3D[k].error,
            image_ids=points3D[k].image_ids,
            point2D_idxs=points3D[k].point2D_idxs,
        )
    write_points3D_binary(new_points3D, os.path.join(paths.sfm_output, "points3D.bin"))

    if prior_meshes.verts is not None and prior_meshes.faces is not None:
        scaled_prior = prior_meshes.verts * scale_value
        save_obj(os.path.join(paths.base, f"{preprocess_dir}/prior/prior_obj_scaled.obj"), scaled_prior.detach().cpu(), prior_meshes.faces)

        if prior_meshes.orig_verts is not None and prior_meshes.orig_faces is not None:
            scaled_orig_prior = prior_meshes.orig_verts * scale_value
            save_obj(
                os.path.join(paths.base, f"{preprocess_dir}/prior/orig_prior_obj_scaled.obj"),
                scaled_orig_prior.detach().cpu(),
                prior_meshes.orig_faces,
            )

    is_grasping_flags_tensor = torch.tensor(is_grasping_flags, dtype=torch.int32, device=scale.device)
    torch.save(is_grasping_flags_tensor, os.path.join(paths.base, f"{preprocess_dir}/grasping_flags.pt"))


def export_aligned_hand_data(
    paths: Paths,
    optimized_list: List[str],
    cam_intr: torch.Tensor,
    mano_right: ManoLayer,
    faces_right: torch.Tensor,
    faces_left: torch.Tensor,
    mano_params: ManoParameters,
    hand_data: HandSequenceData,
    result: OptimizationResult,
    point_cloud: Optional[torch.Tensor],
    prior_meshes: PriorMeshes,
    visualize: bool,
) -> None:
    aligned_verts, aligned_joints = compute_mano_outputs(
        mano_right, mano_params.rot, mano_params.pose, mano_params.shape, result.right_translations
    )

    if "left" in optimized_list and result.left_translations is not None and mano_params.pose_left is not None and mano_params.rot_left is not None:
        aligned_verts_left, aligned_joints_left = compute_mano_outputs(
            mano_right, mano_params.rot_left, mano_params.pose_left, mano_params.shape, result.left_translations
        )
        aligned_verts_left[:, :, 0] = -aligned_verts_left[:, :, 0]
        aligned_joints_left[:, :, 0] = -aligned_joints_left[:, :, 0]
    else:
        aligned_verts_left = None
        aligned_joints_left = None

    right_translations_map: Dict[int, torch.Tensor] = {}
    left_translations_map: Dict[int, torch.Tensor] = {}
    selected_frames = {
        0,
        len(hand_data.frames) // 4,
        len(hand_data.frames) // 3,
        len(hand_data.frames) // 2,
        len(hand_data.frames) // 2 + len(hand_data.frames) // 4,
        len(hand_data.frames) // 2 + len(hand_data.frames) // 3,
        len(hand_data.frames) - 1,
    }

    for idx, (verts, joints, faces, transl, hand_id) in enumerate(hand_data.hand_verts_seq):
        frame_id, img_data, _ = hand_data.frames[idx]

        right_translations_map[frame_id] = result.right_translations[idx]
        aligned_verts_np = aligned_verts[idx].detach().cpu()
        aligned_joints_np = aligned_joints[idx].detach().cpu().numpy()
        np.savetxt(os.path.join(paths.aligned_hand, f"{frame_id}_{hand_id}.xyz"), aligned_joints_np)
        save_obj(os.path.join(paths.aligned_mesh, f"{frame_id}_{hand_id}.obj"), aligned_verts_np, faces)

        if "left" in optimized_list and aligned_verts_left is not None and aligned_joints_left is not None and result.left_translations is not None:
            left_translations_map[frame_id] = result.left_translations[idx]
            aligned_verts_left_np = aligned_verts_left[idx].detach().cpu()
            aligned_joints_left_np = aligned_joints_left[idx].detach().cpu().numpy()
            np.savetxt(os.path.join(paths.aligned_hand, f"{frame_id}_0.xyz"), aligned_joints_left_np)
            save_obj(os.path.join(paths.aligned_mesh, f"{frame_id}_0.obj"), aligned_verts_left_np, faces_left)

        if visualize and idx in selected_frames:
            img = np.zeros((hand_data.img_dim[0], hand_data.img_dim[1], 3), dtype=np.uint8)
            proj_2d = project_points(
                torch.cat((aligned_joints[idx], aligned_joints_left[idx]), dim=0) if aligned_joints_left is not None else aligned_joints[idx],
                cam_intr,
            )
            target_2d = (
                torch.cat((hand_data.hand_proj_targets[idx], hand_data.hand_proj_targets_left[idx]), dim=0)
                if hand_data.hand_proj_targets_left
                else hand_data.hand_proj_targets[idx]
            )

            for joint_idx in range(proj_2d.shape[0]):
                pr = proj_2d[joint_idx].detach().cpu().numpy().astype(np.int32)
                tg = target_2d[joint_idx].detach().cpu().numpy().astype(np.int32)
                cv2.circle(img, tuple(pr), 2, (0, 255, 0), -1)
                cv2.circle(img, tuple(tg), 2, (0, 0, 255), -1)

            cv2.imwrite(os.path.join(paths.vis, f"proj_{frame_id}_{hand_id}.jpg"), img)
            if "object" in optimized_list and point_cloud is not None:
                rot = torch.tensor(qvec2rotmat(img_data.qvec), dtype=torch.float32, device=cam_intr.device)
                transl = torch.tensor(img_data.tvec, dtype=torch.float32, device=cam_intr.device) * result.scale
                scaled_point_cloud = torch.matmul(rot, point_cloud.T * result.scale).T + transl
                save_ply(os.path.join(paths.vis, f"object_cam_{frame_id}.ply"), scaled_point_cloud.detach().cpu())

                if prior_meshes.verts is not None and prior_meshes.faces is not None:
                    scaled_prior = torch.matmul(rot, prior_meshes.verts.T * result.scale).T + transl
                    save_obj(os.path.join(paths.vis, f"prior_{frame_id}.obj"), scaled_prior.detach().cpu(), prior_meshes.faces)
                if prior_meshes.orig_verts is not None and prior_meshes.orig_faces is not None:
                    scaled_orig_prior = torch.matmul(rot, prior_meshes.orig_verts.T * result.scale).T + transl
                    save_obj(os.path.join(paths.vis, f"orig_prior_{frame_id}.obj"), scaled_orig_prior.detach().cpu(), prior_meshes.orig_faces)

            save_ply(os.path.join(paths.vis, f"hand_cam_{frame_id}_{hand_id}.ply"), aligned_verts[idx].detach().cpu())
            save_obj(os.path.join(paths.vis, f"hand_mesh_{frame_id}_{hand_id}.obj"), aligned_verts[idx].detach().cpu(), faces)
            if aligned_verts_left is not None:
                save_ply(os.path.join(paths.vis, f"hand_cam_{frame_id}_0.ply"), aligned_verts_left[idx].detach().cpu())
                save_obj(os.path.join(paths.vis, f"hand_mesh_{frame_id}_0.obj"), aligned_verts_left[idx].detach().cpu(), faces_left)

    right_translations_tensor = torch.cat([right_translations_map[k] for k in sorted(right_translations_map.keys())], dim=0)
    output_dir = os.path.dirname(paths.aligned_mesh)
    torch.save(right_translations_tensor, os.path.join(output_dir, "right_translations.pth"))

    if left_translations_map:
        left_translations_tensor = torch.cat([left_translations_map[k] for k in sorted(left_translations_map.keys())], dim=0)
        torch.save(left_translations_tensor, os.path.join(output_dir, "left_translations.pth"))


def main(seq_name: str, load_prior: bool = False, apply_exp: bool = True, visualize: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = build_paths(seq_name)

    detection_confidences = load_detection_confidences(paths)
    ensure_directories(paths, visualize)
    optimized_list = determine_optimized_list(seq_name)

    cam_intr, images, points3D, point_cloud, filtered_point_cloud, prior_meshes = load_camera_and_prior(paths, optimized_list, device, load_prior)
    mano_right, faces_right, faces_left = load_mano_layers(device)
    mano_params = load_mano_parameters(paths, optimized_list, device)
    hand_data, scale_hand, scale_obj = prepare_hand_sequences(
        seq_name, paths, optimized_list, images, filtered_point_cloud, device, mano_params, faces_right, faces_left, visualize
    )

    init_scale = (scale_hand * 2 / scale_obj) if "object" in optimized_list else 1.0
    scale_log, optimizer = build_optimizer(apply_exp, init_scale, hand_data.translations, hand_data.left_translations, device)

    result = optimize_scale_and_translation(
        seq_name,
        optimized_list,
        apply_exp,
        cam_intr,
        point_cloud,
        filtered_point_cloud,
        prior_meshes,
        mano_right,
        faces_right,
        faces_left,
        mano_params,
        hand_data,
        detection_confidences,
        scale_log,
        optimizer,
    )

    print("Optimization Done. Final Scale:", result.scale.item())
    if result.loss_dict_avg:
        print("Average Projection Loss:", result.loss_dict_avg.get("proj_loss"))
        print("Average Smoothness Loss:", result.loss_dict_avg.get("smoothness_loss"))
    print("Average Z Translation (Right):", torch.mean(result.right_translations[:, 2]).item())
    if result.dists_right:
        print("Average Distance to Object Surface (Right):", torch.mean(torch.tensor(result.dists_right)).item())

    if "left" in optimized_list and result.left_translations is not None:
        print("Average Z Translation (Left):", torch.mean(result.left_translations[:, 2]).item())
        if result.dists_left:
            print("Average Distance to Object Surface (Left):", torch.mean(torch.tensor(result.dists_left)).item())

    if "proximity_loss" in result.loss_dict_avg:
        print("Average Proximity Loss:", result.loss_dict_avg["proximity_loss"])

    export_sfm_results(paths, optimized_list, hand_data.frames, images, points3D, point_cloud, prior_meshes, result.scale, hand_data.is_grasping_flags)
    export_aligned_hand_data(
        paths,
        optimized_list,
        cam_intr,
        mano_right,
        faces_right,
        faces_left,
        mano_params,
        hand_data,
        result,
        point_cloud,
        prior_meshes,
        visualize,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align and scale hand meshes with respect to object points.")
    parser.add_argument("--seq_name", type=str, help="Sequence name (e.g., 'dfki_drill_03')")
    parser.add_argument("--load_prior", action="store_true", help="Whether to load prior mesh")
    parser.add_argument("--apply_exp", action="store_true", help="Whether to apply exponential scaling")
    parser.add_argument("--visualize", action="store_true", help="Save projected joints visualizations")
    args = parser.parse_args()
    main(args.seq_name, args.load_prior, args.apply_exp, args.visualize)
