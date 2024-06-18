#!/bin/sh
set -x

exp_dir=$1
config=$2


model_dir=${exp_dir}/model

export PYTHONPATH=.
python -u run/train_3d_backbone.py \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${model_dir}/model_last.pth.tar \
  2>&1 | tee ${exp_dir}/train_3d_backbone-$(date +"%Y%m%d_%H%M").log