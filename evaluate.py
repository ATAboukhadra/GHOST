import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from utils.metric_utils import Evaluator  # assumes your class is saved as evaluator.py

def load_rgba_image(path):
    """Load image as float32 normalized to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img)

def evaluate_sequences(seq_names, exp_name):
    evaluator = Evaluator().cuda().eval()
    results = []

    for seq_name in seq_names:
        print(f"\n=== Evaluating sequence: {seq_name} ===")
        pred_dir = f"data/{seq_name}/output/{exp_name}/all_rendered_frames"
        gt_dir = f"data/{seq_name}/ghost_build/combined_rgba"
        
        frame_ids = sorted([
            f for f in os.listdir(pred_dir)
            if f.endswith('.png')
        ])
        
        seq_metrics = []

        for frame_file in tqdm(frame_ids, desc=f"{seq_name}"):
            pred_path = os.path.join(pred_dir, frame_file)
            gt_path = os.path.join(gt_dir, frame_file)

            if not os.path.exists(gt_path):
                print(f"Skipping frame {frame_file} (GT missing)")
                continue

            # Load predicted and GT RGBA
            pred = load_rgba_image(pred_path).cuda()
            gt = load_rgba_image(gt_path).cuda()

            rgb_pred = pred[..., :3].permute(2, 0, 1).unsqueeze(0)  # NCHW
            rgb_gt = gt[..., :3].permute(2, 0, 1).unsqueeze(0)
            mask_gt = gt[..., 3:].unsqueeze(0)  # NHWC

            with torch.no_grad():
                metrics = evaluator(rgb_pred, rgb_gt, mask_gt=mask_gt, get_both=False)
                seq_metrics.append(metrics)

        # Average per sequence
        if len(seq_metrics) == 0:
            continue
        mean_metrics = {k: torch.stack([m[k] for m in seq_metrics]).mean().item() for k in seq_metrics[0].keys()}
        results.append(mean_metrics)
        print(f" → Sequence average: {mean_metrics}")

    # Average across all sequences
    all_mean = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    print("\n=== Global Average Across Sequences ===")
    for k, v in all_mean.items():
        print(f"{k}: {v:.4f}")

    # Save as JSON
    import json
    out_path = f"metrics_summary_{exp_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_mean, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate rendered sequences against ground truth.")
    parser.add_argument(
        "sequences",
        nargs="+",
        help="One or more sequence names to evaluate",
    )
    parser.add_argument(
        "--exp-name",
        default="combined",
        help="Experiment name used under data/<seq>/output/",
    )
    args = parser.parse_args()

    evaluate_sequences(args.sequences, args.exp_name)
