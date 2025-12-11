#!/bin/bash
sequence=$1
num_hands=$2
hand="both"
if [[ "${num_hands}" -eq 1 ]]; then
    hand="right"
fi
python ./train.py \
    --source_path data/"${sequence}"/ghost_build/ \
    --model_path  data/"${sequence}"/output/combined/ \
    --object_path data/"${sequence}"/output/object/point_cloud/iteration_30000/point_cloud.ply \
    --images combined_rgba \
    --use_object_mask \
    --hand "${hand}" \
    --gaussians_per_edge 10 \
    --resolution 1 \
    --data_device cuda \
    --sh_degree 3 \
    --iterations 30000 \
    --densify_until_iter 15000 \
    --test_iterations 15000 30000 \
    --save_iterations 15000 30000 \
    --eval \
    --optimize_mano # Change this to optimize transl only
