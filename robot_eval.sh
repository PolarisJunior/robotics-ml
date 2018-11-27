#!/usr/bin/env bash

python eval.py \
    --logtostderr \
    --pipeline_config_path=./robot_plate.config \
    --checkpoint_dir=./ \
    --eval_dir=eval/