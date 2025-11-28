#!/bin/bash
# ------------------------------------------------------------------------
# GHOST Preprocessing Script (Single Sequence Version)
# ------------------------------------------------------------------------
# Usage:
#   bash run_single_sequence.sh \
#       --seq dfki_toy_02 \
#       --obj_points "+473,471 -200,350" \
#       --hands 1 \
#       --hand_pixels "1174,613" \
#       --prompt "cougar toy" \
#       --sfm vggsfm \
#       --window 30 \
#       --use_prior true
#
# ------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ghost
export PYTHONWARNINGS="ignore"
# ----------------------------- DEFAULTS -------------------------------

SEQ=""
OBJ_POINTS=""
NUM_HANDS=1
HAND_PIXELS=""
TEXT_PROMPT=""
SFM_METHOD="vggsfm"     # options: vggsfm | hloc
WINDOW_SIZE=30          # sliding window size for HLOC
USE_PRIOR=true          # whether to align with object prior
VISUALIZE=false         # optional flag

# --------------------------- PARSE ARGUMENTS --------------------------

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seq) SEQ="$2"; shift ;;
        --obj_points) OBJ_POINTS="$2"; shift ;;
        --hands) NUM_HANDS="$2"; shift ;;
        --hand_pixels) HAND_PIXELS="$2"; shift ;;
        --prompt) TEXT_PROMPT="$2"; shift ;;
        --sfm) SFM_METHOD="$2"; shift ;;           # vggsfm or hloc
        --window) WINDOW_SIZE="$2"; shift ;;
        --use_prior) USE_PRIOR="$2"; shift ;;
        --visualize) VISUALIZE=true ;;
        *) echo "❌ Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$SEQ" ]; then
    echo "❌ Error: --seq <sequence_name> is required"
    exit 1
fi

echo "------------------------------------------------"
echo "   GHOST Preprocessing — Single Sequence Mode   "
echo "------------------------------------------------"
echo "Sequence:        $SEQ"
echo "Object Points:   $OBJ_POINTS"
echo "Hands:           $NUM_HANDS"
echo "Hand Pixels:     $HAND_PIXELS"
echo "Text Prompt:     $TEXT_PROMPT"
echo "SFM Method:      $SFM_METHOD"
echo "Window Size:     $WINDOW_SIZE"
echo "Use Prior:       $USE_PRIOR"
echo "Visualize:       $VISUALIZE"
echo "------------------------------------------------"

seq_start=$(date +%s)

# ----------------------------- STEP 1 -----------------------------------
echo "1️⃣  SAM Object Segmentation"
python sam_object.py "$SEQ" $OBJ_POINTS

# ----------------------------- STEP 2 -----------------------------------
echo "2️⃣  Structure-from-Motion"

if [[ "$SFM_METHOD" == "vggsfm" ]]; then
    echo " - Using VGG-SfM"
    python vggsfm_video.py SCENE_DIR=../data/"$SEQ" init_window_size=$WINDOW_SIZE window_size=$WINDOW_SIZE \
                        camera_type="SIMPLE_PINHOLE" query_method="sp+sift" #max_query_pts=512
else
    echo " - Using HLOC with sliding window: $WINDOW_SIZE"
    # python hloc_estimation.py \
    #     --seq "$SEQ" \
    #     --window $WINDOW_SIZE \
    #     --no_vis
fi

# ----------------------------- STEP 3 -----------------------------------
echo "3️⃣  HAMER Hand Fitting"
# python hamer_video.py --model wilor \
#     --img_folder "../data/${SEQ}/build/image/" \
#     --out_folder "../data/${SEQ}/gs_preprocessing/" \
#     --save_mesh

# ----------------------------- STEP 4 -----------------------------------
echo "4️⃣  SAM Hand Segmentation"

IFS=',' read -r h1x h1y h2x h2y <<< "$HAND_PIXELS"

if [[ "$NUM_HANDS" -eq 2 ]]; then
    echo " - Running 2-hand segmentation"
    # python sam_hand_pixel.py "$SEQ" "$h1x" "$h1y" 1
    # python sam_hand_pixel.py "$SEQ" "$h2x" "$h2y" 0
else
    echo " - Running 1-hand segmentation"
    # python sam_hand_pixel.py "$SEQ" "$h1x" "$h1y" 1
fi

# ----------------------------- STEP 5 -----------------------------------
echo "5️⃣  Combining masks"
# python combine_masks.py "$SEQ"

# ----------------------------- STEP 6 -----------------------------------
if [[ -n "$TEXT_PROMPT" ]]; then
    echo "6️⃣  Retrieving prior using text prompt"
    echo " - Prompt: $TEXT_PROMPT"
    # python retrieve.py "$SEQ" --text "$TEXT_PROMPT" --topk 20
fi

# ----------------------------- STEP 7 -----------------------------------
if [[ "$USE_PRIOR" == true ]]; then
    echo "7️⃣  Aligning point cloud with geometric prior"
    # python align_object.py --seq_name "$SEQ"
else
    echo "7️⃣  Skipping prior alignment"
fi

# ----------------------------- STEP 8 -----------------------------------
echo "8️⃣  Refining scale and MANO translations"
# python refine_scale_ho.py --seq_name "$SEQ" --apply_exp

# ----------------------------- STEP 9 -----------------------------------
echo "9️⃣  Animating hand Gaussians"
python animate_hand_gaussian.py "$SEQ"

seq_end=$(date +%s)
duration=$((seq_end - seq_start))

echo "------------------------------------------------"
echo "🎉 Finished $SEQ in $((duration/60))m $((duration%60))s"
echo "------------------------------------------------"