#!/usr/bin/env bash

LOG=logs/unc/lscm_p2345
mkdir -p ${LOG}
now=$(date +"%Y%m%d_%H%M%S")

python -u trainval_model.py \
-g 0 \
-m train \
-d unc \
-t train \
-f ckpts/unc/lscm_p2345 2>&1 | tee ${LOG}/train_$now.txt

python -u trainval_model.py \
-g 0 \
-m test \
-d unc \
-t val \
-i 700000 \
-f ckpts/unc/lscm_p2345 2>&1 | tee ${LOG}/test_val_$now.txt
