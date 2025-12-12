# Adapted from HOLD

import os
import pycolmap
import h5py
import sys

from glob import glob
from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)
from pycolmap import IncrementalPipelineOptions, ImageSelectionMethod

def colmap_pose_est(seq_name, num_pairs=50, window_size=100):
    preprocess_dir = 'ghost_build'
    image_path = f"../data/{seq_name}/{preprocess_dir}/obj_rgb"
    output_path = f"../data/{seq_name}/{preprocess_dir}/sfm"

    images = Path(image_path)
    outputs = Path(output_path)
    num_images = len(glob(f"{image_path}/*"))

    assert (
        num_pairs <= num_images
    ), f"{num_pairs} should be less or equal to {num_images}"

    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs 
    features = outputs / "features.h5"
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    # feature_conf = extract_features.confs["superpoint_max"]
    feature_conf = {
        'output': 'feats-superpoint-n8000-rmax2500',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 8000,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 2500,
            'resize_force': False,
        },
    }

    matcher_conf = match_features.confs["superglue"]
    references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

    retrieval_path = extract_features.main(
        retrieval_conf, images, image_list=references, feature_path=features
    )
    # Check that each image got features extracted
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_pairs)

    # Load existing NetVLAD pairs
    existing_pairs = set()
    with open(sfm_pairs, "r") as f:
        for line in f:
            existing_pairs.add(line.strip())

    # instead of using pairs_from_retrieval, we use a sliding window to create pairs
    # pairs = []
    for i in range(num_images):
        for j in range(i + 1, i + 1 + window_size):
            if j >= num_images:
                break
            # Make a string of the pair example 0000.png 0001.png
            pair = f"{i:04d}.png {j:04d}.png"
            existing_pairs.add(pair)
            # pair = f"{j:04d}.png {i:04d}.png"
            # existing_pairs.add(pair)

    # save in sfm_pairs
    with open(sfm_pairs, "w") as f:
        for pair in sorted(existing_pairs):
            f.write(f"{pair}\n")
    print("Pairs", len(existing_pairs))
    # print("Saved pairs to", sfm_pairs)

    feature_path = extract_features.main(feature_conf, images, outputs)

    with h5py.File(feature_path, 'r') as f:
        keypoints_per_image = {k: f[k]['keypoints'].shape[0] for k in f.keys()}
        print("Min:", min(keypoints_per_image.values()))
        print("Max:", max(keypoints_per_image.values()))
        print("Mean:", sum(keypoints_per_image.values()) / len(keypoints_per_image))

    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        image_options={"camera_model": "PINHOLE"},
    )
