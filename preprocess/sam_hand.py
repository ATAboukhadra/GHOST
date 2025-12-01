import os
import torch
import numpy as np
import cv2
import sys
# import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils.sam_utils import show_mask
from tqdm import tqdm

import numpy as np

def get_hand_bbox(joints_2d, expansion_factor=1.2):
    """
    Compute a bounding box from 2D projected hand joints.
    Expands by a factor to ensure full coverage.
    """
    min_x, min_y = np.min(joints_2d, axis=0)
    max_x, max_y = np.max(joints_2d, axis=0)

    # Expand the box slightly to include the full hand
    width = max_x - min_x
    height = max_y - min_y

    min_x -= width * (expansion_factor - 1) / 2
    max_x += width * (expansion_factor - 1) / 2
    min_y -= height * (expansion_factor - 1) / 2
    max_y += height * (expansion_factor - 1) / 2

    return int(min_x), int(min_y), int(max_x), int(max_y)

if len(sys.argv) != 5:
    print("Usage: python3 sam_hand_pixel.py <seq_name> <x> <y> <is_right>")
    print("is_right: 1 for right hand, 0 for left hand")
    print("Example: python3 sam_hand_pixel.py dfki_hand_02 1024 518 1")
    sys.exit(1)

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")

seq_name = sys.argv[1]
x = int(sys.argv[2])
y = int(sys.argv[3])
is_right = int(sys.argv[4])

video_dir = f'../data/{seq_name}/jpg/'
preprocess_dir = 'ghost_build'

output_dir = f'../data/{seq_name}/{preprocess_dir}/hand_bin_right/'
if not is_right:
    output_dir = f'../data/{seq_name}/{preprocess_dir}/hand_bin_left/'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir.replace('hand_bin', 'hand_rgba')), exist_ok=True)

hamer_joints_dir = f'../data/{seq_name}/{preprocess_dir}/hand_joints2d/'

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
ann_frame_idx = 0
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
# Read the pixel from the command line
clicked_data = {'clicked_point': (x, y)}
print(f"Clicked coordinates: {clicked_data['clicked_point']}")
# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[clicked_data['clicked_point'][0], clicked_data['clicked_point'][1]]], np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

for out_frame_idx, frame_name in tqdm(enumerate(frame_names)):
    seq = video_dir.split("/")[-2]
    ann_frame_idx = int(os.path.splitext(frame_name)[0])
    filename = os.path.join(hamer_joints_dir, f"{ann_frame_idx}_{is_right}.xyz")
    hand_joints = np.loadtxt(filename) # 21x2
    x_min, y_min, x_max, y_max = get_hand_bbox(hand_joints, 1.1)
    box = np.array([x_min, y_min, x_max, y_max], np.float32)

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # fill 4 digits zeros
        out_frame_idx_str = f"{out_frame_idx:04d}"

        # # mask out all white pixels outside the bounding box
        out_mask = out_mask[0]  # Shape becomes (1080, 1920)
        out_mask = (out_mask > 0).astype(np.uint8) * 255
        h, w = out_mask.shape
        mask_inside_box = np.zeros((h, w), dtype=np.uint8)
        mask_inside_box[y_min:y_max, x_min:x_max] = out_mask[y_min:y_max, x_min:x_max]
        out_mask = mask_inside_box[None, :, :]  # Shape back to (1, 1080, 1920)
        # print(out_mask.min(), out_mask.max())

        mask = show_mask((out_mask > 0), obj_id=out_obj_id, filename=os.path.join(output_dir, frame_name))
        mask_bin = (mask > 0).astype(np.uint8) * 255

        # save it rgba style by combining the mask as alpha channel
        image = cv2.imread(os.path.join(video_dir, frame_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = mask_bin[:, :, 0]

        # save the image with the mask as alpha channel
        cv2.imwrite(os.path.join(output_dir.replace('hand_bin', 'hand_rgba'), frame_name.replace('.jpg', '.png')), image)
