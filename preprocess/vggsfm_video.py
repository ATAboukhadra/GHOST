# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
import os
import shutil
    
from omegaconf import DictConfig, OmegaConf

# from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.runners.video_runner import VideoRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines


@hydra.main(config_path="cfgs/", config_name="video_demo")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VideoRunner is the main controller.

    VideoRunner assumes a sequential input of images.
    """

    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.preferred_linalg_library("magma")

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VideoRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
        prefix='ghost_build/obj_rgb'

    )

    sequence_list = test_dataset.sequence_list

    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    output_dir = batch[
        "scene_dir"
    ]  # which is also cfg.SCENE_DIR for DemoLoader
    print(output_dir)
    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = (
        batch["crop_params"] if batch["crop_params"] is not None else None
    )

    # Cache the original images for visualization, so that we don't need to re-load many times
    original_images = batch["original_images"]

    # Run VGGSfM
    # Both visualization and output writing are performed inside VGGSfMRunner
    predictions = vggsfm_runner.run(
        images,
        masks=masks,
        original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir,
        init_window_size=cfg.init_window_size,
        window_size=cfg.window_size,
        joint_BA_interval=cfg.joint_BA_interval,
    )

    print("Video Demo Finished Successfully")
    
    # mv the files from source to destination
    src_dir = os.path.join(output_dir, 'sparse/')
    dst_dir = os.path.join(output_dir, 'sfm/')
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Move all files & subdirectories
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        shutil.move(s, d)

    # Remove the original directory
    shutil.rmtree(src_dir)
    print(f"Moved {src_dir} to {dst_dir}")

    return True


if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
    

