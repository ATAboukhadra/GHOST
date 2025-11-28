import argparse
import sys
from utils.colmap_utils import colmap_pose_est
sys.path = ["../submodules/"] + sys.path
# sys.path = ["../submodules/hloc/"] + sys.path # HLoc doesnt work without this
sys.path = ["../submodules/SuperGluePretrainedNetwork/"] + sys.path # HLoc doesnt work without this

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name")
    parser.add_argument(
        "--num_pairs",
        type=int,
        help="number of the frames that the model is searching for connections",
    )
    parser.add_argument("--window_size", type=int, help="window size for sliding window pairs")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    seq_name = args.seq_name
    num_pairs = args.num_pairs

    print("Processing sequence", seq_name)
    colmap_pose_est(seq_name, num_pairs, args.window_size)
