#!/usr/bin/env bash
# run at root dir of the project

#01
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/train_net.py --config-file "configs/quick_schedules/e2e_faster_rcnn_R_50_C4_quick.yaml" \
    OUTPUT_DIR results/coco_e2e_faster_rcnn_R_50_C4_quick \
    SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8
