import torch
import numpy as np
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.str = np.str_
np.unicode = np.unicode_
np.object = np.object_
np.complex = np.complex_

from manopth.manolayer import ManoLayer
from utils.gaussian_hand_helpers import initialize_gaussians_evenly, simulate_hand_texture, save_gaussians_as_ply, load_mesh_from_obj
from utils.transformation_utils import compute_face_transformation_optimized, update_gaussians, apply_transformations
from tqdm import tqdm
from pytorch3d.io import save_obj

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("sequence", type=str, help="Name of the sequence to process")
args = parser.parse_args()

sequences = [args.sequence]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pose_flat = torch.zeros(1, 48, requires_grad=True).to(device)
shape = torch.zeros(1, 10, requires_grad=True).to(device)
transl = torch.zeros(1, 3, requires_grad=True).to(device)
mano_layer = ManoLayer(mano_root='_DATA/data/mano', use_pca=False, flat_hand_mean=True).to(device)

max_num_gaussians_per_edge = 11  # Number of Gaussians per edge
preprocess_dir = 'ghost_build'
# Process each frame
for side in ['right', 'left']:

    is_right = int(side == 'right')
    
    for seq_name in sequences:
        if 'arctic' not in seq_name and side == 'left':
            continue

        root = f'../data/{seq_name}/{preprocess_dir}/'
        os.makedirs(os.path.join(root, 'gaussians'), exist_ok=True)
        os.makedirs(os.path.join(root, 'canonical'), exist_ok=True)
        
        shape_path = os.path.join(root, f"shape_params_right.pt")
        shape_params = torch.load(shape_path, map_location=device).to(torch.float32)[0].unsqueeze(0)

        canonical_verts, _ = mano_layer(pose_flat, shape_params, transl)
        canonical_verts = canonical_verts[0] / 1000  # Canonical (flat) hand vertices

        if side == 'left':
            canonical_verts[:, 0] = -canonical_verts[:, 0]

        # save it as obj
        faces = mano_layer.th_faces  # Faces of the mesh
        if side == 'left':
            # Mirror faces for left hand
            faces = faces[:, [2, 1, 0]]

        save_obj(f"canonical_{side}.obj", canonical_verts, faces)

        # Initialize gaussians in canonical space
        rescale_factor = 3
        all_gaussians = []

        for num_gaussians_per_edge in range(1, max_num_gaussians_per_edge):
            gaussians = initialize_gaussians_evenly(canonical_verts.detach(), faces, num_gaussians_per_edge, rescale_factor)
            print(f"Initialized {len(gaussians)} Gaussians")
            all_gaussians.append(gaussians)
        
        img_root = os.path.join(root, 'hand_rgba/')
        num_frames = len(os.listdir(img_root))
        seq_transformations = []
        failed = False

        for num_gaussians_per_edge in range(1, max_num_gaussians_per_edge):
            gaussians = all_gaussians[num_gaussians_per_edge - 1]
            num_gaussians = len(gaussians)
            features_dc, features_rest = simulate_hand_texture(num_gaussians)
            opacities = torch.ones(num_gaussians, device=device) * 0.1  # Slightly more opaque for dark skin
            canonical_path = os.path.join(root, f"canonical/gaussians_{side}_{num_gaussians_per_edge}.ply")
            save_gaussians_as_ply(gaussians, features_dc, features_rest, opacities, canonical_path)

        for frame_num in tqdm(range(0, num_frames)):

            frame_path = os.path.join(root, f"hand_meshes_aligned/{frame_num}_{is_right}.obj")
            if os.path.exists(frame_path):
                # Load deformed mesh
                loaded_verts, _ = load_mesh_from_obj(frame_path, device=device)
                if loaded_verts.shape[0] == 778:
                    deformed_verts = loaded_verts
            else:
                print(f"Frame {frame_num} not found.")
                failed = True
                break
                
            # Compute transformations
            transformations = compute_face_transformation_optimized(canonical_verts, deformed_verts, faces)
            # print(transformations)
            seq_transformations.append(transformations)
            updated_verts = apply_transformations(canonical_verts, transformations, faces)

            # Update Gaussians
            if frame_num % 10 == 0:
                num_gaussians_per_edge = 2
                num_gaussians_per_face = (num_gaussians_per_edge * (num_gaussians_per_edge - 1)) // 2 + num_gaussians_per_edge
                gaussians = all_gaussians[num_gaussians_per_edge-1]
                updated_gaussians = update_gaussians(gaussians, transformations, num_gaussians_per_face)
                # Save updated Gaussians as .ply
                num_gaussians = len(gaussians)
                features_dc, features_rest = simulate_hand_texture(num_gaussians)
                opacities = torch.ones(num_gaussians, device=device) * 0.1  # Slightly more opaque for dark skin    
                ply_file = os.path.join(root, f"gaussians/gaussians_frame_{frame_num}_{side}.ply")
                save_gaussians_as_ply(updated_gaussians, features_dc, features_rest, opacities, ply_file)
                # exit()
            # print(f"Frame {frame_num} processed and saved.")

        if not failed:
            seq_transformations = torch.stack(seq_transformations)
            output_path = os.path.join(root, f"{side}_transformations.pth")
            torch.save(seq_transformations, output_path)
        else:
            print(f"Failed to process {seq_name} {side}.")