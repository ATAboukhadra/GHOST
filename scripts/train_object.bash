#!/bin/bash
sequence=$1

python ./train.py \
    --source_path data/"${sequence}"/ghost_build/ \
    --model_path data/"${sequence}"/output/object/ \
    --images obj_rgba \
    --use_object_mask \
    --hand none \
    --resolution 1 \
    --data_device cuda \
    --sh_degree 3 \
    --iterations 30000 \
    --densify_until_iter 15000 \
    --test_iterations 15000 30000 \
    --save_iterations 15000 30000 \
    --lambda_background 0.3 \
    --eval \
    --background_ignore_mask hand_rgba \
    # --use_obj_prior \
    # --lambda_geo 10.0 \
    # --tau_out 0.02 \
    # --tau_fill 0.00001 \
    # --w_fill 0.5 \
    # --w_out 0.5 \
    # --run_id 100 \