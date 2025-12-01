"""
This script uses the following pipeline:
1. Person detection (every 'det_frequency' frame, otherwise track person)
2. Wholebody pose estimation including hands
3. Extract only hand keypoints
4. Compute bounding boxes for hands
5. Draw bounding boxes around hands

With det_frequency=10 (Line 56), FPS improves ~2.5x compared to the detection without tracking.

For CPU: pip install opencv-python rtmlib onnxruntime
For GPU: pip install opencv-python rtmlib onnxruntime-gpu
"""
import time

import cv2
import numpy as np

# import onnxruntime as ort

# sess_options = ort.SessionOptions()
# sess_options.intra_op_num_threads = 1  # or more, but keep explicit
# sess_options.inter_op_num_threads = 1

from rtmlib import Wholebody, PoseTracker, draw_bbox


def benchmark(detector, duration=5):
    """
    Benchmark the performance of a detector on dummy images.

    Args:
        detector (callable): The detector function to benchmark. It should take an image as input.
        duration (int): The duration in seconds for which to run the benchmark.

    Prints:
        Frames per second (FPS) achieved by the detector.
    """
    # Create a dummy image (fixed size, e.g., 640x480 with 3 channels)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initialize counters
    start_time = time.time()
    frames_processed = 0

    # Run the benchmark for the specified duration
    while time.time() - start_time < duration:
        # Run the detector on the dummy image
        _ = detector(dummy_image)
        frames_processed += 1

    # Calculate elapsed time and FPS
    elapsed_time = time.time() - start_time
    fps = frames_processed / elapsed_time

    print(f"Benchmark completed: {frames_processed} frames processed in {elapsed_time:.2f} seconds.")
    print(f"FPS: {fps:.2f}")


def create_detector():
    pose_model = PoseTracker(
        Wholebody,
        det_frequency=10,  # every 10 frames
        mode="performance",
        backend="onnxruntime",
        device="cuda",  # or "cuda" if onnxruntime-gpu is installed
    )
    # pose_model = Wholebody(
    #     det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
    #     det_input_size=(640, 640),
    #     pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",
    #     pose_input_size=(288, 384),
    #     backend="onnxruntime",
    #     device="cuda",
    # )

    def detector(frame, square=True, pad_factor=0.1):
        all_keypoints, all_scores = pose_model(frame)
        boxes = []
        is_right = []
        confidences = []
        kps = []

        num_hand_keypoints = 21
        l_start_id = 91
        r_start_id = 91 + num_hand_keypoints

        def pose_to_box(keypoints, scores):
            # Filter keypoints with valid scores
            valid_mask = scores > 0.1
            if not np.any(valid_mask):
                return None, []  # No valid keypoints
            
            valid_keypoints = keypoints[valid_mask]
            x = valid_keypoints[:, 0]
            y = valid_keypoints[:, 1]
            score = scores[valid_mask].mean()
            box = [x.min(), y.min(), x.max(), y.max()]

            # Make box square
            x_min, y_min, x_max, y_max = box
            # print(f"Hand box (before square): {x_min}, {y_min}, {x_max}, {y_max}")
            width = box[2] - box[0]
            height = box[3] - box[1]
            if square:
                if width > height:
                    # Adjust height to match width
                    diff = (width - height) / 2
                    y_min -= diff
                    y_max += diff
                else:
                    # Adjust width to match height
                    diff = (height - width) / 2
                    x_min -= diff
                    x_max += diff
            
            # Pad the box
            if pad_factor > 0:
                width = x_max - x_min
                height = y_max - y_min
                x_min -= width * pad_factor
                x_max += width * pad_factor
                y_min -= height * pad_factor
                y_max += height * pad_factor

            # print(f"Hand box (after square): {x_min}, {y_min}, {x_max}, {y_max}")
            box = [x_min, y_min, x_max, y_max]

            return box, score

        # Process left and right hand keypoints in a single loop
        for keypoints, scores in zip(all_keypoints, all_scores):
            # Extract left and right hand keypoints and scores
            l_keypoints = keypoints[l_start_id:l_start_id + num_hand_keypoints]
            r_keypoints = keypoints[r_start_id:r_start_id + num_hand_keypoints]
            det_freq = pose_model.det_frequency if hasattr(pose_model, 'det_frequency') else 1
            l_scores = scores[l_start_id:l_start_id + num_hand_keypoints] / det_freq
            r_scores = scores[r_start_id:r_start_id + num_hand_keypoints] / det_freq

            # Compute bounding boxes for left and right hands
            for h, (kpts, scs) in enumerate([(l_keypoints, l_scores), (r_keypoints, r_scores)]):
                box, score = pose_to_box(kpts, scs)
                # make sure the box is non degenerate
                if box is not None and (box[2] - box[0]) > 0 and (box[3] - box[1]) > 0:
                    boxes.append(box)
                    is_right.append(h == 1)
                    confidences.append(scs)
                    kps.append(kpts)


        return boxes, is_right, kps, confidences

    return detector


def demo(detector, camera_id):
    # Find and initialize the camera device
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("No camera found. Exiting.")
        exit(1)

    # Process the camera frames
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Perform hand detection
            boxes = detector(frame)

            # Draw bounding boxes on the frame
            draw_bbox(frame, boxes)

            # Display the frame
            cv2.imshow("Hand Detection", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting on user request.")
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# # Initialize the hand detector
# print("Initializing hand detector...")
# detector = create_detector()

# # Run a benchmark to measure the performance of the detector
# print("Running benchmark...")   
# benchmark(detector)

# # Ask if user wants to run a live demo or exit
# print("Would you like to run a live demo? (y/n)")
# response = input().lower()
# if response != "y":
#     print("Exiting.")
#     exit(0)


# # Run the live demo
# print("Running live demo with camera 0...")
# camera_id = 0
# demo(detector, camera_id)
