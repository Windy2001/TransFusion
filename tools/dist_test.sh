#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m torch.distributed.launch --nproc_per_node=4 --master_port=43000 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch configs/transfusion_nusc_voxel_LC.py \
    work_dirs/transfusion_nusc_voxel_LC/latest.pth --eval mAP --show --show-dir result_dirs
