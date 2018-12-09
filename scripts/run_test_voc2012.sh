#!/usr/bin/env bash
# run at root dir of the project

#01
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/test_net.py --config-file "configs/pascal_voc/e2e_faster_rcnn_R_50_C4_1x_4_gpu_voc_2012.yaml"
