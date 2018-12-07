#!/usr/bin/env bash

MODEL_NUM=2399
python eval.py \
    --logtostderr \
    --pipeline_config_path=./robot_plate.config \
    --checkpoint_dir=./anni_exports/model_${MODEL_NUM} \
    --eval_dir=eval/