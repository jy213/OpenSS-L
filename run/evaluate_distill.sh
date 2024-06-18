#!/bin/sh
set -x

exp_dir=$1
config=$2
feature_type=$3

mkdir -p ${exp_dir}
result_dir=${exp_dir}/result_eval

export PYTHONPATH=.
python -u run/evaluate_distill.py \
  --config=${config} \
  feature_type ${feature_type} \
  save_folder ${result_dir} \
  2>&1 | tee -a ${exp_dir}/evaluate_distill-$(date +"%Y%m%d_%H%M").log