#!/usr/bin/env bash

MODEL_NUM=1922
DIR_PREFIX=anni
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=./robot_plate.config
TRAINED_CKPT_PREFIX=./${DIR_PREFIX}_checkpoints/model.ckpt-${MODEL_NUM}
EXPORT_DIR=./${DIR_PREFIX}_exports/model_${MODEL_NUM}/
python ./export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
