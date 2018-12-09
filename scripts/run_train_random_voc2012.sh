#!/usr/bin/env bash
# run at root dir of the project

#01
export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/train_net.py --config-file "configs/pascal_voc/e2e_faster_rcnn_R_50_C4_1x_4_gpu_voc_2012.yaml" \
    MODEL.WEIGHT "" \
    OUTPUT_DIR: results/voc2012_train_val_random_test \
    SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 2
