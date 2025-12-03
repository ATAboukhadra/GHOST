import argparse
import os
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer,
    SoftSilhouetteShader, RasterizationSettings, BlendParams, PerspectiveCameras
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from utils.colmap_readmodel import qvec2rotmat, read_cameras_binary, read_images_binary, read_points3D_binary
from pytorch3d.io import save_obj, save_ply
# import matplotlib.pyplot as plt
# from optim_utils import filter_pc
# import pymeshlab
import trimesh
from pathlib import Path
# import shutil
# pip install rtree

def setup_renderer_low(cam_intr, image_size, device, scale=0.5):
    H, W = image_size
    Hs, Ws = int(H * scale), int(W * scale)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(Hs, Ws),
        blur_radius=1e-4,         # tiny but non-zero keeps grads stable
        faces_per_pixel=10,        # huge speedup
        bin_size=0,            # enable tiling (faster on big imgs)
    )
    
    K = torch.tensor([
        [cam_intr[0, 0] * scale, 0, cam_intr[0, 2] * scale, 0],
        [0, cam_intr[1, 1] * scale, cam_intr[1, 2] * scale, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    cams = PerspectiveCameras(K=K, device=device, in_ndc=False,
                              image_size=((Hs, Ws),))
    renderer_low = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cams, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return renderer_low, (Hs, Ws)

def setup_renderer(cam_intr, image_size, device):
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    K = torch.tensor([
        [cam_intr[0, 0], 0, cam_intr[0, 2], 0],
        [0, cam_intr[1, 1], cam_intr[1, 2], 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    cameras = PerspectiveCameras(K=K, device=device, in_ndc=False, image_size=(image_size, ))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return renderer, rasterizer

def load_mask(path):
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    return alpha

def visualize_projection_cv2(rendered, gt_mask, frame_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Convert tensors → numpy (H, W)
    gt_np = gt_mask.detach().cpu().numpy()
    rendered_np = rendered.detach().cpu().numpy()

    # Normalize to 0–255 uint8 if needed
    def to_uint8(x):
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / x.max()
        return (x * 255).astype(np.uint8)

    gt_u8 = to_uint8(gt_np)
    rendered_u8 = to_uint8(rendered_np)

    # ----- 1. GT mask (grayscale) -----
    gt_color = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out_dir, f"{frame_id:04d}_gt.png"), gt_color)

    # ----- 2. Rendered mask -----
    rendered_color = cv2.cvtColor(rendered_u8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out_dir, f"{frame_id:04d}_rendered.png"), rendered_color)

    # ----- 3. Overlay (jet heatmap on rendered over GT) -----
    # Jet colormap
    rendered_heat = cv2.applyColorMap(rendered_u8, cv2.COLORMAP_JET)
    gt_color = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)

    # Blend overlay (alpha = 0.5)
    overlay = cv2.addWeighted(gt_color, 1.0, rendered_heat, 0.5, 0)

    cv2.imwrite(os.path.join(out_dir, f"{frame_id:04d}_overlay.png"), overlay)

    print(f"✅ Saved GT, rendered, and overlay for frame {frame_id:04d}")

def export_transformed_obj_and_pc(mesh, img, frame_point_cloud, out_dir):
    # R_obj = torch.tensor(qvec2rotmat(img.qvec), dtype=torch.float32, device=point_cloud.device)
    # T_obj = torch.tensor(img.tvec, dtype=torch.float32, device=point_cloud.device)
    # frame_point_cloud = torch.matmul(R_obj, point_cloud.T).T + T_obj
    # print(frame_point_cloud.shape)
    frame_num = img.id - 1
    save_obj(os.path.join(out_dir, f"frame_{frame_num:04d}.obj"), mesh.verts_packed(), mesh.faces_packed())
    save_ply(os.path.join(out_dir, f"frame_{frame_num:04d}_point_cloud.ply"), frame_point_cloud)

def project_points(points, cam_intr):
    points_2d = torch.matmul(points, cam_intr.t())
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d

def plot_pointcloud_2d_cv(pc_cam, cam_intr, image_size, out_path, background=None):
    """
    Projects a 3D point cloud into 2D and draws it using OpenCV.
    - pc_cam: (N, 3) tensor in camera space
    - cam_intr: (3, 3) or (4, 4) tensor
    - image_size: (H, W)
    - background: optional image (H, W, 3) in BGR format
    """
    H, W = image_size
    cam_intr = cam_intr.to(pc_cam.device)

    # Keep only valid depth
    valid = pc_cam[:, 2] > 0
    pc_cam = pc_cam[valid]

    # Project to 2D
    pc_2d = project_points(pc_cam, cam_intr)
    u, v = pc_2d[:, 0], pc_2d[:, 1]

    # Filter image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds].round().long()
    v = v[in_bounds].round().long()

    # Create blank or use background
    if background is None:
        img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background
    else:
        img = background.copy()

    # Draw points
    for x, y in zip(u.cpu().numpy(), v.cpu().numpy()):
        cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)  # green

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"✅ Saved OpenCV 2D projection to {out_path}")


# --- helpers ---
def p3d_to_trimesh(meshes: Meshes) -> trimesh.Trimesh:
    V = meshes.verts_packed().detach().cpu().numpy()
    F = meshes.faces_packed().detach().cpu().numpy()
    return trimesh.Trimesh(vertices=V, faces=F, process=True)

def trimesh_to_p3d(tm: trimesh.Trimesh, device="cuda") -> Meshes:
    V = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device)
    F = torch.tensor(np.asarray(tm.faces),    dtype=torch.int64,  device=device)
    return Meshes(verts=[V], faces=[F])

def outer_surface_mesh(tm: trimesh.Trimesh, n_rays: int = 100_000) -> trimesh.Trimesh:
    """Keep only faces first-hit by rays shot from a sphere around the mesh."""
    tm = tm.copy().process()
    if not tm.is_watertight:
        tm.remove_unreferenced_vertices()

    dirs = np.random.normal(size=(n_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # unit vectors

    center = tm.bounds.mean(axis=0)
    radius = np.linalg.norm(tm.extents) * 3.0
    # radius = max(np.linalg.norm(tm.bounds - tm.bounds.mean(axis=0), axis=1)) * 1.5

    origins = center[None, :] + dirs * radius

    first = tm.ray.intersects_first(origins, -dirs)  # face indices or -1
    faces = np.unique(first[first != -1])

    if faces.size == 0:
        # fallback: convex hull if rays miss (rare)
        outer = tm.convex_hull
    else:
        outer = tm.submesh([faces], append=True)
    outer.remove_unreferenced_vertices()
    return outer

def main(seq_name, k=10, apply_exp=True, visualize=False):
    preprocess_dir = 'ghost_build'
    base_dir = f"../data/{seq_name}"
    cam_path = f"{base_dir}/{preprocess_dir}/sfm/"
        
    mask_dir = f"{base_dir}/{preprocess_dir}/obj_rgba"
    device = torch.device("cuda")

    cameras = read_cameras_binary(os.path.join(cam_path, "cameras.bin"))
    images = read_images_binary(os.path.join(cam_path, "images.bin"))
    points3D = read_points3D_binary(os.path.join(cam_path, "points3D.bin"))

    point_cloud = torch.tensor([p.xyz for p in points3D.values()], dtype=torch.float32, device=device)
    print(f"Original point cloud shape: {point_cloud.shape}")

    point_cloud = torch.tensor([p.xyz for p in points3D.values() if p.image_ids.shape[0] >= 30], dtype=torch.float32, device=device)
    # filter point cloud by removing black points
    print(f"Point cloud shape: {point_cloud.shape}")

    # Get image size from COLMAP camera intrinsics
    key = list(cameras.keys())[0]
    width = int(cameras[key].width)
    height = int(cameras[key].height)

    cam_intr = np.array(cameras[key].params)
    if cam_intr.shape[0] == 4:
        fx, fy, cx, cy = cam_intr[0], cam_intr[1], cam_intr[2], cam_intr[3]
    else:
        fx, fy, cx, cy = cam_intr[0], cam_intr[0], cam_intr[1], cam_intr[2]

    cam_intr = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

    image_size = (height, width)
    renderer_low, image_size_low = setup_renderer_low(torch.tensor(cam_intr, device=device), image_size, device, scale=1.0)

    best_loss = float('inf')

    frame_ids = sorted([int(os.path.basename(f).split('.')[0]) for f in os.listdir(mask_dir) if f.endswith('.png')])
    masks_full = torch.stack([torch.tensor(load_mask(f"{mask_dir}/{fid:04d}.png"),
                                    device=device) for fid in frame_ids], 0)  # [F,H,W]
    masks_low = torch.nn.functional.interpolate(masks_full[:,None],
                                                size=image_size_low, mode='bilinear')[:,0]  # [F,Hs,Ws]

    prev_meshes = {}

    # Create new file to log errors (best_object, best_loss)
    log_file = f"{base_dir}/{preprocess_dir}/prior/align_errors.txt"
    os.makedirs(f"{base_dir}/{preprocess_dir}/prior/", exist_ok=True)

    for i in range(0, k):
        try:
            mesh_path = next(Path('.').rglob(f"{base_dir}/{preprocess_dir}/openshape/openshape_text/txt_{i:02d}_*.obj")).resolve()
            print(f"Processing mesh {i}: {mesh_path}")
            mesh = load_objs_as_meshes([mesh_path], device=device)
        except Exception as e:
            print(f"Error loading mesh: {e}")
            continue

        orig_mesh_verts = mesh.verts_packed()
        orig_mesh_faces = mesh.faces_packed()

        # Check if this mesh has been processed before by checking vertex count
        if orig_mesh_verts.shape[0] in [v[0] for v in prev_meshes.values()] and orig_mesh_faces.shape[0] in [v[1] for v in prev_meshes.values()]:
            print(f"Skipping mesh {i} as it has already been processed.")
            continue

        prev_meshes[i] = (orig_mesh_verts.shape[0], orig_mesh_faces.shape[0])
        if not (10 <= orig_mesh_faces.shape[0] <= 1000000):
            print(f"⚠️ Skipping mesh {i} due to faces count: {orig_mesh_faces.shape[0]}")
            continue
        else:
            print(f"Mesh {i} has {orig_mesh_verts.shape[0]} vertices, and {orig_mesh_faces.shape[0]} faces.")

        # Apply outer surface mesh approximation
        print(f"Extracting outer mesh {i} from {mesh_path}")

        n_rays = min(int(1e8 / orig_mesh_faces.shape[0]), int(1e5))
        print(f"Mesh {i} is sampled with {n_rays} rays.")

        tm = p3d_to_trimesh(mesh)
        tm_outer = outer_surface_mesh(tm, n_rays=n_rays)
        mesh = trimesh_to_p3d(tm_outer, device=device)

        print(f"Outer mesh {i} has {mesh.verts_packed().shape[0]} vertices and {mesh.faces_packed().shape[0]} faces.")

        # save outer mesh
        if visualize:
            os.makedirs(f"{base_dir}/{preprocess_dir}/prior/openshape_outer", exist_ok=True)
            outer_mesh_path = f"{base_dir}/{preprocess_dir}/prior/openshape_outer/{i}.obj"
            trimesh.exchange.export.export_mesh(tm_outer, outer_mesh_path)

        verts_rgb = torch.ones(1, mesh.verts_packed().shape[0], 3, device=device)  # (1, V, 3)
        mesh.textures = texture = TexturesVertex(verts_features=verts_rgb)

        mesh_verts = mesh.verts_packed()  # (V, 3)
        mesh_faces = mesh.faces_packed()  # (F, 3)

        pc_range = point_cloud.max(dim=0).values - point_cloud.min(dim=0).values
        mesh_range = mesh_verts.max(dim=0).values - mesh_verts.min(dim=0).values
        init_scale = (pc_range / (mesh_range + 1e-6)).clamp(max=10.0, min=1e-3)
        # make initi_scal all the same value (minimum of the three)
        init_scale = torch.tensor([init_scale.min().item()] * 3, device=device)
        s = torch.nn.Parameter(init_scale.to(device).unsqueeze(0))  # (1, 3)
        log_s = torch.nn.Parameter(init_scale.log()[None].clone()) if apply_exp else torch.nn.Parameter(init_scale.clone()[None])
        q = torch.nn.Parameter(torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device))
        # r6d = torch.nn.Parameter(torch.zeros(1, 6, device=device, dtype=torch.float32))  # 6D rotation representation

        # init translation to the mean of the point cloud
        tvec = torch.nn.Parameter(point_cloud.mean(dim=0, keepdim=True).to(device))  # (1, 3)
        # optimizer = torch.optim.Adam([s, rvec, tvec], lr=0.05)
        optimizer = torch.optim.AdamW([
            {"params": [q],  "lr": 1e-2},
            {"params": [tvec],  "lr": 1e-2},
            {"params": [log_s], "lr": 1e-2},   # a bit higher than direct-s
        ], betas=(0.9, 0.99), eps=1e-8)

        # optimizer = torch.optim.Adam([r6d, tvec, log_s], lr=1e-2, betas=(0.9,0.99))
        max_iters = 1500
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        local_best = float('inf')
        for iter in tqdm(range(max_iters)):
            optimizer.zero_grad()
            total_iou_loss = 0
            # print(q, tvec, log_s)
            # Transformation params
            # R = axis_angle_to_matrix(rvec).unsqueeze(0).to(torch.float32)  # (1, 3, 3)
            R = quaternion_to_matrix(q).unsqueeze(0).to(torch.float32)  # (1, 3, 3)
            # R = rotation_6d_to_matrix(r6d)
            s = log_s.exp() if apply_exp else log_s

            verts_transformed = s[:, None, :] * mesh_verts  # scale
            verts_transformed = torch.bmm(verts_transformed, R) + tvec[:, None, :]  # rot + transl
            verts_transformed = verts_transformed.squeeze(0)

            # Sample N frames
            N = 10
            rand_gen = np.random.RandomState(seed=iter)  # Iteration number as seed
            frame_ids_sampled = rand_gen.choice(frame_ids, size=min(N, len(frame_ids)), replace=False)

            for frame_id in frame_ids_sampled:
                # check if len(images) > frame_id + 1
                img = images[frame_id] if 'dfki' not in base_dir else images[frame_id + 1]  # this used to be frame_id + 1 for hloc but in vggsfm it is frame_id
                R_colmap = torch.tensor(qvec2rotmat(img.qvec), device=device, dtype=torch.float32)
                T_colmap = torch.tensor(img.tvec, device=device, dtype=torch.float32)

                verts_cam = torch.matmul(R_colmap, verts_transformed.T).T + T_colmap
                mesh_frame = Meshes(verts=[verts_cam], faces=[mesh_faces], textures=texture)
                rendered = renderer_low(mesh_frame)[0, ..., 3].clamp(0, 1)  # alpha mask
                # Flip X and Y
                rendered_flip = rendered.flip(0).flip(1)  # (H, W)
                
                gt_mask = masks_low[frame_id]  # [H, W]

                intersection = (rendered_flip * gt_mask).sum()
                union = (rendered_flip + gt_mask - rendered_flip * gt_mask).sum() + 1e-6
                iou = intersection / union
                loss_iou = 1 - iou

                # Chamfer loss
                R_obj = R_colmap
                T_obj = T_colmap
                pc_cam = torch.matmul(R_obj, point_cloud.T).T + T_obj  # point cloud in camera space

                total_iou_loss += loss_iou

            # Chamfer loss in world space
            dist1 = torch.cdist(verts_transformed[None], point_cloud[None], p=2)
            dist2 = torch.cdist(point_cloud[None], verts_transformed[None], p=2)
            loss_chamfer = (dist1.min(dim=1).values.mean() + dist2.min(dim=1).values.mean()) / 2.0

            num_frames = len(frame_ids_sampled)
            if iter < 100:
                loss_avg = total_iou_loss / num_frames + 10 * loss_chamfer  # weight chamfer loss
            else:
                loss_avg = total_iou_loss / num_frames

                if loss_avg > 0.8:
                    print(f"⚠️ Skipping object {i} due to high loss: {loss_avg.item()}")
                    break

            # Normalize losses
            loss_avg.backward()
            optimizer.step()
            # scheduler.step()

            if iter > 50 and total_iou_loss / num_frames < best_loss:

                best_loss = total_iou_loss / num_frames
                best_verts = verts_transformed.clone()
                best_faces = mesh.faces_packed().clone()
                object_name = str(mesh_path).split('/')[-1].split('.')[0]
                print('Best object so far:', object_name)

                save_obj(f"{base_dir}/{preprocess_dir}/prior/best_obj.obj", best_verts, best_faces)

                orig_mesh_verts_transformed = s[:, None, :] * orig_mesh_verts  # scale
                orig_mesh_verts_transformed = torch.bmm(orig_mesh_verts_transformed, R) + tvec[:, None, :]  # rot + transl
                # Export original mesh
                save_obj(f"{base_dir}/{preprocess_dir}/prior/best_orig_obj.obj", orig_mesh_verts_transformed.squeeze(0), orig_mesh_faces)

                print(f"[Iter {iter+1:03d}] "
                    f"Rendering Loss (IoU) = {total_iou_loss.item() / num_frames:.4f}, "
                    f"Chamfer Loss = {loss_chamfer.item():.4f}, ")

                with open(log_file, 'a') as f:
                    f.write(f"{i}, {iter}, {frame_id}, {object_name}, {best_loss.item():.6f}\n")

                if args.visualize:
                    visualize_projection_cv2(rendered_flip, gt_mask, frame_id, f"{base_dir}/{preprocess_dir}/prior/{i}/projections")
                    export_transformed_obj_and_pc(mesh_frame, img, pc_cam, f"{base_dir}/{preprocess_dir}/prior/{i}/projections")

            if total_iou_loss / num_frames < local_best:
                local_best = total_iou_loss / num_frames
                last_change = iter                    

            if iter - last_change > 200:
                print(f"⚠️ Stopping early for object {i} due to no improvement.")
                break

    print("✅ Alignment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh alignment with COLMAP.")
    parser.add_argument(
        "--seq_name",
        type=str,
        required=True,
        help="Sequence name to process."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of objects to process."
    )
    parser.add_argument("--apply_exp", action="store_true", help="Whether to apply exponential scaling")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize projections")
    args = parser.parse_args()

    main(args.seq_name, args.k, args.apply_exp, args.visualize)
