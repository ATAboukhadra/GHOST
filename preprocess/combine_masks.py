
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sequence", type=str, help="Name of the sequence")
args = parser.parse_args()

seq_name = args.sequence
img_root = f'../data/{seq_name}/build/jpg/'

preprocess_dir = 'ghost_build'
hand_root_masks = f'../data/{seq_name}/{preprocess_dir}/hand_sam_masks/'
left_hand_root_masks = f'../data/{seq_name}/{preprocess_dir}/hand_sam_masks_left/'
obj_seg_masks = f'../data/{seq_name}/{preprocess_dir}/object_sam_masks/'

out_mask_dir = f'../data/{seq_name}/{preprocess_dir}/combined_masks/'
out_rgba_dir = f'../data/{seq_name}/{preprocess_dir}/combined_rgba/'
out_hand_rgba_dir = f'../data/{seq_name}/{preprocess_dir}/combined_hand_rgba/'
out_color_mask_dir = f'../data/{seq_name}/{preprocess_dir}/combined_color_masks/'
out_bkg_mask_dir = f'../data/{seq_name}/{preprocess_dir}/bkg_masks/'

os.makedirs(out_mask_dir, exist_ok=True)
os.makedirs(out_rgba_dir, exist_ok=True)
os.makedirs(out_hand_rgba_dir, exist_ok=True)
os.makedirs(out_color_mask_dir, exist_ok=True)
os.makedirs(out_bkg_mask_dir, exist_ok=True)

for frame in tqdm(sorted(os.listdir(hand_root_masks))):
    if not frame.endswith('.png'):
        continue

    sam_frame_name = frame.split('.')[0]

    # read masks
    hand_mask = cv2.imread(hand_root_masks + frame, cv2.IMREAD_GRAYSCALE)
    obj_mask = cv2.imread(obj_seg_masks + sam_frame_name + '.jpg.png', cv2.IMREAD_GRAYSCALE)

    if os.path.exists(left_hand_root_masks):
        left_hand_mask = cv2.imread(left_hand_root_masks + frame, cv2.IMREAD_GRAYSCALE)
    else:
        left_hand_mask = np.zeros_like(hand_mask)

    # combine hands
    total_hand_mask = cv2.bitwise_or(hand_mask, left_hand_mask)
    _, hand_bin = cv2.threshold(total_hand_mask, 127, 255, cv2.THRESH_BINARY)
    _, obj_bin = cv2.threshold(obj_mask, 127, 255, cv2.THRESH_BINARY)

    # border gap fix
    kernel = np.ones((7, 7), np.uint8)
    hand_dil = cv2.dilate(hand_bin, kernel, iterations=1)
    obj_dil = cv2.dilate(obj_bin, kernel, iterations=1)
    border_gap = cv2.bitwise_and(hand_dil, obj_dil)
    border_gap = cv2.bitwise_and(border_gap, cv2.bitwise_not(cv2.bitwise_or(hand_bin, obj_bin)))

    # assign border gap to object
    obj_bin = cv2.bitwise_or(obj_bin, border_gap)

    # full mask
    combined_mask = cv2.bitwise_or(hand_bin, obj_bin)

    # read original image
    img = cv2.imread(img_root + sam_frame_name + '.jpg')

    # save combined RGBA (hands + object)
    img_mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_mask[:, :, :3] = img
    img_mask[:, :, 3] = combined_mask
    rgba_path = os.path.join(out_rgba_dir, sam_frame_name + '.png')
    cv2.imwrite(rgba_path, img_mask)

    # save hands-only RGBA
    img_hand_mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_hand_mask[:, :, :3] = img
    img_hand_mask[:, :, 3] = hand_bin
    cv2.imwrite(out_hand_rgba_dir + sam_frame_name + '.png', img_hand_mask)

    # -----------------------------------------------
    # Create fun color overlay while keeping alpha
    # -----------------------------------------------
    rgba_img = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
    rgb_base = rgba_img[:, :, :3]
    alpha = rgba_img[:, :, 3]

    # fun pastel colors (B, G, R)
    COLOR_RIGHT_HAND = np.array([255, 153, 51], dtype=np.uint8)   # soft orange
    COLOR_LEFT_HAND  = np.array([203, 132, 255], dtype=np.uint8)  # soft lavender
    COLOR_OBJECT     = np.array([102, 255, 204], dtype=np.uint8)  # mint aqua

    # start with white
    color_mask = np.ones_like(rgb_base, dtype=np.uint8) * 255

    # assign class colors
    color_mask[obj_bin > 0] = COLOR_OBJECT
    color_mask[hand_mask > 127] = COLOR_RIGHT_HAND
    color_mask[left_hand_mask > 127] = COLOR_LEFT_HAND

    # blend RGB with colors where alpha > 0
    blended_rgb = rgb_base.copy()
    alpha_mask = (alpha > 0)
    blended_rgb[alpha_mask] = cv2.addWeighted(
        rgb_base[alpha_mask], 0.6, color_mask[alpha_mask], 0.4, 0
    )

    # stack back with original alpha
    blended_rgba = np.dstack((blended_rgb, alpha))

    # save final overlay
    cv2.imwrite(out_color_mask_dir + sam_frame_name + '_blended_rgba.png', blended_rgba)

    # -----------------------------------------------
    # Export background binary mask (1 = background, 0 = object)
    # -----------------------------------------------
    # Note: obj_bin is 255 where object exists
    background_mask = np.ones_like(obj_bin, dtype=np.uint8) * 255
    background_mask[obj_bin > 0] = 0
    cv2.imwrite(out_bkg_mask_dir + sam_frame_name + '.png', background_mask)

