import cv2
import numpy as np
import os
from PIL import Image

def show_mask(mask, obj_id=None, random_color=False, filename=None):
    # if random_color:
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]

    # save boolean mask as black and white image
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # save axis
    if filename is not None:
        mask_bin = (mask.reshape(h, w, 1) * 255).astype(np.uint8)
        mask_path = filename + '.png'
        cv2.imwrite(mask_path, mask_bin)
    else:
        cv2.imshow('mask', mask_image)

    return mask_image

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def export_jpg(video_dir):
    # iterate over directory and make a copy from every png to a jpg
    os.makedirs(video_dir.replace('/images/', '/jpg/'), exist_ok=True)
    min_frame_num = min(
        [int(os.path.splitext(p)[0]) for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".png", ".PNG"]]
    )
    for filename in os.listdir(video_dir):
        if filename.endswith('.png'):
            # change frame number to 0-index 
            frame_num = int(os.path.splitext(filename)[0]) - min_frame_num
            new_filename = f"{frame_num:04d}.png"
            img = Image.open(os.path.join(video_dir, filename))
            img.save(os.path.join(video_dir.replace('/images/', '/jpg/'), new_filename.replace('.png', '.jpg')))

