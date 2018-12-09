#!/usr/bin/env bash
# run at root dir of the project

#01
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/train_net.py --config-file "configs/quick_schedules/e2e_faster_rcnn_X_101_32x8d_FPN_quick.yaml" \
    OUTPUT_DIR results/e2e_faster_rcnn_X_101_32x8d_FPN_quick_with_pretrained_detector \
    SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 \
    MODEL.WEIGHT "catalog://Caffe2Detectron/COCO/36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x"
