#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=42000 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch configs/transfusion_nusc_voxel_LC.py --no-validate
