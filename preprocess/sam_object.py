import os
import numpy as np
import cv2
import sys
from PIL import Image
from tqdm import tqdm
# from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils.sam_utils import show_mask, export_jpg

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")

if len(sys.argv) < 3:
    print("Usage: python3 sam_object.py <seq_name> [+x,y|-x,y] ...")
    sys.exit(1)

seq_name = sys.argv[1]
click_args = sys.argv[2:]

points = []
labels = []

for click in click_args:
    if click.startswith('+') or click.startswith('-'):
        label = 1 if click[0] == '+' else 0
        try:
            x_str, y_str = click[1:].split(',')
            x, y = int(x_str), int(y_str)
            points.append([x, y])
            labels.append(label)
        except ValueError:
            print(f"Invalid format for click: {click}. Use +x,y or -x,y")
            sys.exit(1)
    else:
        print(f"Click must start with '+' or '-': {click}")
        sys.exit(1)

points = np.array(points, np.float32)
labels = np.array(labels, np.int32)
print(f"Collected clicks: {list(zip(labels, points))}")

video_dir = f'../data/{seq_name}/images/'
preprocess_dir = 'ghost_build'
output_dir = f'../data/{seq_name}/{preprocess_dir}/obj_bin/'

export_jpg(video_dir)

os.makedirs(output_dir, exist_ok=True)

video_dir = video_dir.replace('/images/', '/jpg/')
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
ann_frame_idx = 0
image = cv2.imread(os.path.join(video_dir, frame_names[ann_frame_idx]))
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

print('Adding points ..')
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
for out_frame_idx in tqdm(range(len(frame_names))):
    seq = video_dir.split("/")[-2]
    if out_frame_idx not in video_segments:
        cv2.imwrite(os.path.join(output_dir, f"{out_frame_idx:04d}.jpg.png"), np.zeros((480, 640), np.uint8))
        continue

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # fill 4 digits zeros
        out_frame_idx_str = f"{out_frame_idx:04d}"
        show_mask(out_mask, obj_id=out_obj_id, filename=os.path.join(output_dir, f"{out_frame_idx_str}.jpg"))

os.makedirs(f'../data/{seq_name}/{preprocess_dir}/obj_rgba/', exist_ok=True)
os.makedirs(f'../data/{seq_name}/{preprocess_dir}/obj_rgb/', exist_ok=True)

for sam_frame_name in frame_names:
    original_frame = cv2.imread(os.path.join(video_dir, sam_frame_name))
    mask_frame = cv2.imread(os.path.join(output_dir, sam_frame_name)+'.png', cv2.IMREAD_GRAYSCALE)
    rgb = cv2.bitwise_and(original_frame, original_frame, mask=mask_frame)
    sam_frame_name = sam_frame_name.replace('.jpg', '.png')

    cv2.imwrite(f'../data/{seq_name}/{preprocess_dir}/obj_rgb/{sam_frame_name}', rgb)
    rgba = cv2.cvtColor(original_frame, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_frame
    cv2.imwrite(f'../data/{seq_name}/{preprocess_dir}/obj_rgba/{sam_frame_name}', rgba)