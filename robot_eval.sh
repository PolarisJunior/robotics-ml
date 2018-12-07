#!/usr/bin/env bash

python eval.py \
    --logtostderr \
    --pipeline_config_path=./robot_plate.config \
    --checkpoint_dir=./anni_exports/model_2399 \
    --eval_dir=eval/