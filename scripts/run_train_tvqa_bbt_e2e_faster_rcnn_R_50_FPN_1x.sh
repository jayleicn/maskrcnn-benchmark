#!/usr/bin/env bash
# run at root dir of the project

#01
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/train_net.py --config-file "configs/tvqa_bbt/e2e_faster_rcnn_R_50_FPN_1x_tvqa_bbt.yaml" \
    OUTPUT_DIR results/run_train_tvqa_bbt_e2e_faster_rcnn_R_50_FPN_1x \
    SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8
